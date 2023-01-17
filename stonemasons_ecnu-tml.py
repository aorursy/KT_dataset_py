import torch as ch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder, ImageNet
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json
import pdb
clf = (models.resnet50, 224)
ch.set_default_tensor_type('torch.cuda.FloatTensor')
IMAGENET_PATH = '../input/ilsvrc2012val/ILSVRC2012-val/'
def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

def gd_prior_step(x, g, lr):
    return x + lr*g

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj
def make_adversarial_examples(image, true_label, args, model_to_fool, IMAGENET_SL):
    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup
    prior_size = IMAGENET_SL if not args.tiling else args.tile_size
    upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
    total_queries = ch.zeros(args.batch_size)
    prior = ch.zeros(args.batch_size, 3, prior_size, prior_size)
    dim = prior.nelement()/args.batch_size
    prior_step = gd_prior_step if args.mode == 'l2' else eg_step
    image_step = l2_image_step if args.mode == 'l2' else linf_step
    proj_maker = l2_proj if args.mode == 'l2' else linf_proj
    proj_step = proj_maker(image, args.epsilon)
    print(image.max(), image.min())

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    def normalized_eval(x):
        x_copy = x.clone()
        x_copy = ch.stack([F.normalize(x_copy[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
                        for i in range(args.batch_size)])
        return model_to_fool(x_copy)

    L = lambda x: criterion(normalized_eval(x), true_label)
    losses = L(image)

    # Original classifications
    orig_images = image.clone()
    orig_classes = normalized_eval(image).argmax(1).cuda()
    correct_classified_mask = (orig_classes == true_label).float()
    total_ims = correct_classified_mask.sum()
    not_dones_mask = correct_classified_mask.clone()
    print(correct_classified_mask[:50].cpu())
    t = 0
    while not ch.any(total_queries > args.max_queries):
        t += args.gradient_iters*2
        if t >= args.max_queries:
            break
        if not args.nes:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration*ch.randn_like(prior)/(dim**0.5) 
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(args.fd_eta*args.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)
        else:
            prior = ch.zeros_like(image)
            for _ in range(args.gradient_iters):
                exp_noise = ch.randn_like(image)/(dim**0.5) 
                est_deriv = (L(image + args.fd_eta*exp_noise) - L(image - args.fd_eta*exp_noise))/args.fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

            # Preserve images that are already done, 
            # Unless we are specifically measuring gradient estimation
            prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1)), args.image_lr)
        image = proj_step(new_im)
        image = ch.clamp(image, 0, 1)
        if args.mode == 'l2':
            if not ch.all(norm(image - orig_images) <= args.epsilon + 1e-3):
                pdb.set_trace()
        else:
            if not (image - orig_images).max() <= args.epsilon + 1e-3:
                pdb.set_trace()

        ## Continue query count
        total_queries += 2*args.gradient_iters*not_dones_mask
        not_dones_mask = not_dones_mask*((normalized_eval(image).argmax(1) == true_label).float())

        ## Logging stuff
        new_losses = L(image)
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
        if args.log_progress:
            print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))

        if current_success_rate == 1.0:
            break

    return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': success_mask.cpu().numpy()
    }

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        return self.params[x.lower()]
    
args = dict()
args['tiling'] = True
args['nes'] = False
args['log_progress'] = True
args['total_images'] = 10000
defaults = {
    "fd_eta": 0.01,
    "max_queries": 10000,
    "image_lr": 0.5,
    "mode": "l2",
    "online_lr": 0.1,
    "exploration": 0.01,
    "epsilon": 5.0,
    "batch_size": 500,
    "gradient_iters": 1,
    "total_images": 10000,
    "tile_size": 50
}    

defaults.update(args)
args = Parameters(defaults)

args.params

def main(args, model_to_fool, dataset_size):
    dataset = ImageFolder(IMAGENET_PATH, 
                    transforms.Compose([
                        transforms.Resize(dataset_size),
                        transforms.CenterCrop(dataset_size),
                        transforms.ToTensor(),
                    ]))
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size)
    total_correct, total_adv, total_queries = 0, 0, 0
    for i, (images, targets) in enumerate(dataset_loader):
        if i*args.batch_size >= args.total_images:
            break
        res = make_adversarial_examples(images.cuda(), targets.cuda(), args, model_to_fool, dataset_size)
        ncc = res['num_correctly_classified'] # Number of correctly classified images (originally)
        num_adv = ncc * res['success_rate'] # Success rate was calculated as (# adv)/(# correct classified)
        queries = num_adv * res['average_queries'] # Average queries was calculated as (total queries for advs)/(# advs)
        total_correct += ncc
        total_adv += num_adv
        total_queries += queries

    print("-"*80)
    print("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
            aq=total_queries/total_adv,
            succ=total_adv/total_correct))
    print("-"*80)
model_type = clf[0]
model_to_fool = model_type(pretrained=True).cuda()
model_to_fool = DataParallel(model_to_fool)
model_to_fool.eval()
with ch.no_grad():
    main(args, model_to_fool, clf[1])