import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import requests
from io import StringIO
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
!wget https://thesource.com/wp-content/uploads/2015/02/Pablo_Picasso1.jpg -O /kaggle/working/style.jpg
!wget http://mymasterpiecesmile.com/wp-content/uploads/2018/11/Masterpiece-Smiles-Clinton-TN-Dentist.jpg -O /kaggle/working/content.jpg
def im_crop_center(img):
    img_width, img_height = img.size
    w = h = min(img_width, img_height)
    left, right = (img_width - w) / 2, (img_width + w) / 2
    top, bottom = (img_height - h) / 2, (img_height + h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    return img.crop((left, top, right, bottom))
def image_loader(name):
    image = Image.open(name)
    image = im_crop_center(image,)
    
    image = loader(image).unsqueeze(0)
    return image.to(device, dtype = torch.float)
unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title = None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# plt.figure()
# imshow(style_img)
# plt.figure()
# imshow(content_img)

img = image_loader("/kaggle/working/content.jpg")
imshow(img)
img1 = image_loader("/kaggle/working/style.jpg")
imshow(img1)
style_img = img1
content_img = img
assert style_img.size() == content_img.size()
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c*d)
    g = torch.mm(features, features.t())
    
    return g.div(a*b*c*d)
class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_features).detach()
    def forward(self, input):
        g = gram_matrix(input)
        self.loss = F.mse_loss(g, self.target)
        return input
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
    def forward(self, img):
        return (img - self.mean) / self.std
cl_def = ['conv_4']
sl_def = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, n_mean, n_std, st_img, ct_img, ctl = cl_def, stl = sl_def):
    cnn = copy.deepcopy(cnn)
    norm = Normalization(n_mean, n_std).to(device)
    ct_losses = []
    st_losses = []
    
    model = nn.Sequential(norm)
    
    i=0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace = False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer')
            break
        model.add_module(name, layer)
        
        if name in ctl:
            target = model(ct_img).detach()
            ct_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), ct_loss)
            ct_losses.append(ct_loss)
        
        if name in stl:
            target_st = model(st_img).detach()
            st_loss = StyleLoss(target_st)
            model.add_module("style_loss_{}".format(i), st_loss)
            st_losses.append(st_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, st_losses, ct_losses

input_img = img
plt.figure()
imshow(input_img)
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
cnn = models.vgg19(pretrained=True).features.to(device).eval()
output = run_style_transfer(cnn, cnn_norm_mean, cnn_norm_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output)

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()