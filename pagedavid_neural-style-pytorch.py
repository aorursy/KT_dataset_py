import torch
import torch.nn.functional as F
import torchvision.models
from copy import deepcopy
!mkdir ~/.torch
!mkdir ~/.torch/models
!cp ../input/vgg16/vgg16.pth ~/.torch/models/vgg16.pth
!mv ~/.torch/models/vgg16.pth ~/.torch/models/vgg16-397923af.pth
vgg16 = torchvision.models.vgg16(pretrained=True)
class ContentLoss(torch.nn.Module):
    
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.target = None
        self.mode = ''
        
    def forward(self, x):
        if self.mode == 'loss':
            self.loss = F.mse_loss(x, self.target, size_average=True)
        elif self.mode == 'learn':
            self.target = x.detach()
        return x
    
class StyleLoss(torch.nn.Module):
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.target = None
        self.mode = ''
        
    def gram(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        return torch.matmul(features,
                            features.transpose(1, 2)).div(b * c * h * w)
        
        
    def forward(self, x):
        if self.mode == 'loss':
            self.loss = torch.mean((self.gram(x) - self.gram(self.target)).pow(2))
        elif self.mode == 'learn':
            self.target = x.detach()
        return x
class Normalization(torch.nn.Module):
    
    def __init__(self):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        self.mean = mean.view(-1, 1, 1).type(torch.FloatTensor).cuda()
        std = torch.tensor([0.229, 0.224, 0.225])
        self.std = mean.view(-1, 1, 1).type(torch.FloatTensor).cuda()

    def forward(self, x):
        x = x.float() - self.mean.float()
        return x / self.std

class LossNetwork(torch.nn.Module):
    
    def __init__(self, vgg16):
        super(LossNetwork, self).__init__()
        
        self.content_losses = list()
        self.style_losses = list()
        
        cnn = deepcopy(vgg16).features.eval()
        i = 0
        
        model = torch.nn.Sequential(
            Normalization()
        )
        
        content_layers = ['relu_5', 'relu_8']
        style_layers = ['relu_3', 'relu_5', 'relu_8', 'relu_11']
        
        for layer in cnn:
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            if isinstance(layer, torch.nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = torch.nn.ReLU(inplace=False)
            if isinstance(layer, torch.nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                layer = torch.nn.AvgPool2d(kernel_size=2,
                                           stride=2)
            model.add_module(name, layer)
            
            if name in content_layers:
                c_loss = ContentLoss().cuda()
                model.add_module('con_loss_{}'.format(i), c_loss)
                self.content_losses.append(c_loss)
            
            if name in style_layers:
                s_loss = StyleLoss().cuda()
                model.add_module('sty_loss_{}'.format(i), s_loss)
                self.style_losses.append(s_loss)
            
            self.model = model.cuda()
    
    def forward(self, x, c_tar, s_tar):
        for i in self.content_losses:
            i.mode = 'learn'
        for i in self.style_losses:
            i.mode = ''
        self.model(c_tar.double())
        
        for i in self.content_losses:
            i.mode = ''
        for i in self.style_losses:
            i.mode = 'learn'
        self.model(s_tar.double())
        
        for i in self.content_losses:
            i.mode = 'loss'
        for i in self.style_losses:
            i.mode = 'loss'
        self.model(x)
        c_loss_sum = 0
        s_loss_sum = 0
        
        for i in self.content_losses:
            c_loss_sum += i.loss
            
        for i in self.style_losses:
            s_loss_sum += i.loss
            
        return c_loss_sum, s_loss_sum
import skimage.io
import numpy as np
c = skimage.io.imread('../input/st_trans_img/chicago.jpg').astype(np.float32).transpose([2, 0, 1]) / 255.0
s = skimage.io.imread('../input/st_trans_img/wave_ss.jpg').astype(np.float32).transpose([2, 0, 1]) / 255.0
c = torch.from_numpy(c).unsqueeze(0).cuda()
s = torch.from_numpy(s).unsqueeze(0).cuda()
net = LossNetwork(vgg16).cuda()
n = torch.distributions.Normal(torch.tensor([0.5]), torch.tensor([0.01]))
image = torch.tensor(n.sample((1, 3, 474, 712)).view([1, 3, 474, 712]), requires_grad=True, device='cuda')
optimizer = torch.optim.Adam([image], lr=0.05)
result = [[], []]
for i in range(3000):
    c_loss, s_loss = net(image, c, s)
    n_loss = F.mse_loss(image[:, :, 1:, :], image[:, :, : -1, :], size_average=True)\
        + F.mse_loss(image[:, :, :, 1:], image[:, :, :, : -1], size_average=True)
    loss = c_loss + 50000000 * s_loss + 0.01 * n_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        _l = loss
        print('itr: {} loss: {}'.format(i, _l))
        result[0].append(i)
        result[1].append(_l)
    
    if i % 1000 == 0 and i != 0:
        optimizer.param_groups[0]['lr'] /= 10
        print('optimizer lr decayed')
skimage.io.imsave('{}.png'.format(i),
                  np.clip(image.cpu().data.numpy()[0].transpose([1, 2, 0]), 0.0, 1.0))