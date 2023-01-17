import fastai

from fastai.utils.mem import gpu_mem_get_free_no_cache

from fastai.vision import *

from fastai.callbacks import *

from fastai.vision.gan import *

from torchvision.models import vgg16_bn

from torch.autograd import Variable

import torch

import pathlib

import os

import PIL

import warnings

use_cpu = False   

warnings.filterwarnings('ignore')



# Set to True to use just a fraction of the data so that everything will run much faster (though the results will be a lot worst)

use_sample = False
# config

size = 320           # image size

wd = 1e-3            # weight decay

y_range = (-3.,3.)   # min/max values of the generator image output



if use_sample:

    data_dir = pathlib.Path("/kaggle/input/pneumotoraxnoisy/data/siim-acr-noisy-sample/")

else:

    data_dir = pathlib.Path("../input/pneumotoraxnoisy/data/siim-acr-noisy")



img_noisy = data_dir/'img_noisy'            # noisy images

img_clean = data_dir/'img_clean'            # clean images



img_clean_test = data_dir/'img_clean_test'  # clean images test

img_noisy_test = data_dir/'img_noisy_test'  # noisy images test



img_gen2 = pathlib.Path('./img_gen2')              # generator images

img_gen4 = pathlib.Path('./img_gen4')              # generator images 



img_gen2.mkdir(exist_ok=True)

img_gen4.mkdir(exist_ok=True)



if use_cpu:

    fastai.torch_core.defaults.device = 'cpu'

    bs = 12

else:

    free = gpu_mem_get_free_no_cache()

    # set batch size depending on the available GPU RAM 

    if free > 8200: 

        bs = 12

    else:           

        bs = 6

    print(f"Using bs={bs}, size={size}, have {free}MB of GPU RAM free")
img_clean_example = open_image('../input/pneumotoraxnoisy/data/siim-acr-noisy/img_clean/ID_f04064220.jpg').resize(size)

img_noisy_example = open_image('../input/pneumotoraxnoisy/data/siim-acr-noisy/img_noisy/ID_f04064220.jpg').resize(size)



print("Clean vs Noisy")

show_image(img_clean_example, figsize=(14,11))

show_image(img_noisy_example, figsize=(14,11))

plt.show()
# Perceptual Loss



# base loss

base_loss = F.l1_loss



def gram_matrix(x):

    n,c,h,w = x.size()

    x = x.view(n, c, -1)

    return (x @ x.transpose(1,2))/(c*h*w)





if use_cpu:

    vgg_m = vgg16_bn(True).features.eval()

else:

    vgg_m = vgg16_bn(True).features.cuda().eval()



requires_grad(vgg_m, False)

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]



class PerceptualLoss(nn.Module):

    def __init__(self, m_feat, layer_ids, layer_wgts):

        super().__init__()

        self.m_feat = m_feat

        self.loss_features = [self.m_feat[i] for i in layer_ids]

        self.hooks = hook_outputs(self.loss_features, detach=False)

        self.wgts = layer_wgts

        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))

              ] + [f'gram_{i}' for i in range(len(layer_ids))]



    def make_features(self, x, clone=False):

        self.m_feat(x)

        return [(o.clone() if clone else o) for o in self.hooks.stored]

    

    def forward(self, input, target):

        out_feat = self.make_features(target, clone=True)

        in_feat = self.make_features(input)

        self.feat_losses = [base_loss(input,target)]

        self.feat_losses += [base_loss(f_in, f_out)*w

                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3

                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))

        return sum(self.feat_losses)

    

    def __del__(self): self.hooks.remove()

        



perc_loss = PerceptualLoss(vgg_m, blocks[2:5], [5,15,2])
# MSE Loss

mse_loss = MSELossFlat()



# MAE Loss

# mae_loss = F.l1_loss
def content_loss(pred, target):

    coeff = 1  # TODO: find the optimum value

    return mse_loss(pred, target) + (coeff * perc_loss(pred, target))
def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)]).cuda()

    return gauss/gauss.sum()



def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).cuda()

    return window



# structural similarity measure

def SSIM(img1, img2):

     

    (_, channel, _, _) = img1.size()

    window_size = 11

    window = create_window(window_size, channel)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel).cuda()

    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel).cuda()



    mu1_sq = mu1.pow(2)

    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2



    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel).cuda() - mu1_sq

    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel).cuda() - mu2_sq

    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel).cuda() - mu1_mu2



    C1 = 0.01 ** 2

    C2 = 0.03 ** 2



    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()



# peak signal-to-noise ratio

def PSNR(img1, img2):

    mse = torch.mean( (img1/3. - img2/3.) ** 2 )

    if mse == 0:

        return 100

    PIXEL_MAX = 1

    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
arch_gen = models.resnet34

loss_gen = content_loss



def get_gen_data(path, bs, size, split=0.2, labels=img_clean):

    data_gen = (ImageImageList.from_folder(path)

                .split_by_rand_pct(split, seed=1)            

                .label_from_func(lambda x: labels/x.name)                              

                .transform(get_transforms(max_zoom=2), size=size, tfm_y=True)  # TODO: refractor and customize

                .databunch(bs=bs).normalize(imagenet_stats, do_y=True))



    data_gen.c = 3

    

    return data_gen
data_gen = get_gen_data(img_noisy, bs, size//2)  # start the training with half the image size
data_gen.show_batch(3, image_size=1)
def create_gen_learner(data):

    return unet_learner(data, arch_gen, wd=wd, blur=True, norm_type=NormType.Weight, self_attention=True, 

                        y_range=y_range, loss_func=loss_gen, metrics=[SSIM, PSNR])
learn_gen = create_gen_learner(data_gen)

learn_gen.model_dir = "/kaggle/working"
learn_gen.lr_find()

learn_gen.recorder.plot()
# step 1

learn_gen.fit_one_cycle(7, max_lr=1e-4, pct_start=0.7)
learn_gen.save('gen-pre1')

learn_gen.show_results(rows=1, imgsize=5)
# Losses

learn_gen.recorder.plot_losses()
# Metrics (peak signal-to-noise ratio, structural similarity measure)

learn_gen.recorder.plot_metrics()
# step 2

learn_gen.unfreeze()

learn_gen.fit_one_cycle(7, max_lr=slice(1e-6,1e-4), pct_start=0.6)

learn_gen.save('gen-pre2')
# step 3

learn_gen.data = get_gen_data(img_noisy, bs, size)  # scale-up the image size x2 (size/2 --> size)

learn_gen.freeze()

gc.collect()  # garbage collector to reclaim GPU memory
learn_gen.fit_one_cycle(7, max_lr=1e-4, pct_start=0.6)
# Losses

learn_gen.recorder.plot_losses()
learn_gen.unfreeze()

learn_gen.fit_one_cycle(7, max_lr=slice(1e-7,1e-5), pct_start=0.3)
learn_gen.model_dir = "/kaggle/working"

learn_gen.save('gen-pre4')

learn_gen.show_results(rows=4, imgsize=8)
data_gen, learn_gen = None, None

gc.collect()
data_gen = get_gen_data(img_noisy, bs, size)

learn_gen = create_gen_learner(data_gen)

learn_gen.model_dir = "/kaggle/working"

learn_gen.load('gen-pre4')

print()
data_test = get_gen_data(img_noisy_test, bs, size, split=0.99, labels=img_clean_test)

learn_gen.data = data_test
print("loss | SSIM | PSNR")

learn_gen.validate()
learn_gen.show_results(dl=data_test, rows=3, imgsize=8)
# reclaim GPU memory

learn_gen, data_gen, data_test = None, None, None

gc.collect()
def get_critic_data(fake_data, bs, size):

    data_crit = (ImageList.from_folder(data_dir, include=[fake_data, 'img_clean'])

                .split_by_rand_pct(0.2, seed=1)

                .label_from_folder(classes=[fake_data, 'img_clean'])

                .transform(get_transforms(max_zoom=1.3), size=size)  # TODO: refractor and customize

                .databunch(bs=bs).normalize(imagenet_stats))



    data_crit.c = 3

    

    return data_crit
# loss

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())



# architecture

arch_critic = gan_critic(n_channels=3, nf=128, n_blocks=3, p=0.15)
def create_critic_learner(data):

    return Learner(data, arch_critic, metrics=accuracy_thresh_expand, loss_func=loss_critic, wd=wd)
lamb = 50.  # lambda

batch_size = 4
data_critic = get_critic_data('img_noisy', bs=batch_size, size=size)

data_gen = get_gen_data(img_noisy, bs=batch_size, size=size)



learn_gen = create_gen_learner(data_gen)

learn_critic = create_critic_learner(data=data_critic)



learn_gen.model_dir = "/kaggle/working"

learn_critic.model_dir = "/kaggle/working"



switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)

learn = GANLearner.from_learners(learn_gen, learn_critic, weights_gen=(1.,lamb), show_img=False, switcher=switcher,

                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd, gen_first=True)

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
lr = 1e-3

learn.fit(20, lr)
lr = 1e-4

learn.fit(20, lr)
# learn_gen and learn_critic still reference to the same models that were use to create the GANs

learn_gen.save('gan_gen')

learn_critic.save('gan_critic')
learn.show_results(rows=1, imgsize=10)
# reclaim GPU memory

learn_critic , data_critic, learn_gen, data_gen = None, None, None, None

gc.collect()
data_gen = get_gen_data(img_noisy, bs, size)

learn_gen = create_gen_learner(data_gen)

learn_gen.model_dir = "/kaggle/working"

learn_gen.load('gan_gen')  # <-- load the final generator from the GANs

print()
data_test = get_gen_data(img_noisy_test, bs, size, split=0.99, labels=img_clean_test)

learn_gen.data = data_test

gc.collect()
print("loss | SSIM | PSNR")

learn_gen.validate()
learn_gen.show_results(dl=data_test, rows=4, imgsize=10)