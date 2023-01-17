import os

import sys

import numpy as np

import torch

import torchvision

import torch.utils.data as data

from scipy.misc import imsave

import matplotlib

import matplotlib.pyplot as plt 

import os.path

import pandas as pd

import torchvision.transforms as transforms

#from inception_v3 import *



from PIL import Image



from torch import autograd

from torch.autograd.gradcheck import zero_gradients

#from helpers import *



import hashlib

import io



%matplotlib inline
IMG_EXTENSIONS = ['.png', '.jpg']



class LeNormalize(object):

    """Normalize to -1..1 in Google Inception style

    """

    def __call__(self, tensor):

        for t in tensor:

            t.sub_(0.5).mul_(2.0)

        return tensor





def default_inception_transform(img_size):

    tf = transforms.Compose([

        transforms.Scale(img_size),

        transforms.CenterCrop(img_size),

        transforms.ToTensor(),

        LeNormalize(),

    ])

    return tf





def find_inputs(folder, filename_to_target=None, types=IMG_EXTENSIONS):

    inputs = []

    for root, _, files in os.walk(folder, topdown=False):

        for rel_filename in files:

            base, ext = os.path.splitext(rel_filename)

            if ext.lower() in types:

                abs_filename = os.path.join(root, rel_filename)

                target = filename_to_target[rel_filename.split('.')[0]] if filename_to_target else 0

                inputs.append((abs_filename, target))

    return inputs





class Dataset(data.Dataset):



    def __init__(

            self,

            root,

            target_file='../input/nips-2017-adversarial-learning-development-set/images.csv',

            transform=None):

        

        if target_file:

            target_file_path = target_file #os.path.join(root, target_file)

            target_df = pd.read_csv(target_file_path)#, header=None)

            target_df["TargetClass"] = target_df["TargetClass"].apply(int)

            #print(target_df["ImageId"], target_df["TargetClass"])

            f_to_t = dict(zip(target_df["ImageId"], target_df["TargetClass"] - 1))  # -1 for 0-999 class ids

        else:

            f_to_t = dict()



        imgs = find_inputs(root, filename_to_target=f_to_t)

        if len(imgs) == 0:

            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"

                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))



        self.root = root

        self.imgs = imgs

        self.transform = transform



    def __getitem__(self, index):

        path, target = self.imgs[index]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:

            img = self.transform(img)

        if target is None:

            target = torch.zeros(1).long()

        return img, target



    def __len__(self):

        return len(self.imgs)



    def set_transform(self, transform):

        self.transform = transform



    def filenames(self, indices=[], basename=False):

        if indices:

            if basename:

                return [os.path.basename(self.imgs[i][0]) for i in indices]

            else:

                return [self.imgs[i][0] for i in indices]

        else:

            if basename:

                return [os.path.basename(x[0]) for x in self.imgs]

            else:

                return [x[0] for x in self.imgs]



class OneShotDataset(data.Dataset):



    def __init__(

            self,

            filename,

            transform=None):



        self.filename = filename

        self.transform = transform



    def __getitem__(self, index):

        path = self.filename

        target = None

        img = Image.open(path).convert('RGB')

        if self.transform is not None:

            img = self.transform(img)

        if target is None:

            target = torch.zeros(1).long()

        return img, target



    def __len__(self):

        return 1
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo





__all__ = ['Inception3', 'inception_v3']





model_urls = {

    # Inception v3 ported from TensorFlow

    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',

}





def inception_v3(pretrained=False, **kwargs):

    r"""Inception v3 model architecture from

    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    if pretrained:

        if 'transform_input' not in kwargs:

            kwargs['transform_input'] = True

        model = Inception3(**kwargs)

        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))

        return model



    return Inception3(**kwargs)







class Inception3(nn.Module):



    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):

        super(Inception3, self).__init__()

        self.aux_logits = aux_logits

        self.transform_input = transform_input

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)

        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)

        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)

        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        self.Mixed_5b = InceptionA(192, pool_features=32)

        self.Mixed_5c = InceptionA(256, pool_features=64)

        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)

        self.Mixed_6c = InceptionC(768, channels_7x7=160)

        self.Mixed_6d = InceptionC(768, channels_7x7=160)

        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:

            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)

        self.Mixed_7c = InceptionE(2048)

        self.fc = nn.Linear(2048, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                import scipy.stats as stats

                stddev = m.stddev if hasattr(m, 'stddev') else 0.1

                X = stats.truncnorm(-2, 2, scale=stddev)

                values = torch.Tensor(X.rvs(m.weight.data.numel()))

                m.weight.data.copy_(values)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def forward(self, x):

        if self.transform_input:

            x = x.clone()

            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5

            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5

            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        # 299 x 299 x 3

        x = self.Conv2d_1a_3x3(x)

        # 149 x 149 x 32

        x = self.Conv2d_2a_3x3(x)

        # 147 x 147 x 32

        x = self.Conv2d_2b_3x3(x)

        # 147 x 147 x 64

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # 73 x 73 x 64

        x = self.Conv2d_3b_1x1(x)

        # 73 x 73 x 80

        x = self.Conv2d_4a_3x3(x)

        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # 35 x 35 x 192

        x = self.Mixed_5b(x)

        # 35 x 35 x 256

        x = self.Mixed_5c(x)

        # 35 x 35 x 288

        x = self.Mixed_5d(x)

        # 35 x 35 x 288

        x = self.Mixed_6a(x)

        # 17 x 17 x 768

        x = self.Mixed_6b(x)

        # 17 x 17 x 768

        x = self.Mixed_6c(x)

        # 17 x 17 x 768

        x = self.Mixed_6d(x)

        # 17 x 17 x 768

        x = self.Mixed_6e(x)

        # 17 x 17 x 768

        if self.training and self.aux_logits:

            aux = self.AuxLogits(x)

        # 17 x 17 x 768

        x = self.Mixed_7a(x)

        # 8 x 8 x 1280

        x = self.Mixed_7b(x)

        # 8 x 8 x 2048

        x = self.Mixed_7c(x)

        # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)

        # 1 x 1 x 2048

        x = F.dropout(x, training=self.training)

        # 1 x 1 x 2048

        x = x.view(x.size(0), -1)

        # 2048

        x = self.fc(x)

        # 1000 (num_classes)

        if self.training and self.aux_logits:

            return x, aux

        return x





class InceptionA(nn.Module):



    def __init__(self, in_channels, pool_features):

        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)



        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)

        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)



        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)



        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)



    def forward(self, x):

        branch1x1 = self.branch1x1(x)



        branch5x5 = self.branch5x5_1(x)

        branch5x5 = self.branch5x5_2(branch5x5)



        branch3x3dbl = self.branch3x3dbl_1(x)

        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)



        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)



        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionB(nn.Module):



    def __init__(self, in_channels):

        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)



        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)



    def forward(self, x):

        branch3x3 = self.branch3x3(x)



        branch3x3dbl = self.branch3x3dbl_1(x)

        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)



        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)



        outputs = [branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionC(nn.Module):



    def __init__(self, in_channels, channels_7x7):

        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)



        c7 = channels_7x7

        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)

        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))

        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))



        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)

        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))

        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))



        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)



    def forward(self, x):

        branch1x1 = self.branch1x1(x)



        branch7x7 = self.branch7x7_1(x)

        branch7x7 = self.branch7x7_2(branch7x7)

        branch7x7 = self.branch7x7_3(branch7x7)



        branch7x7dbl = self.branch7x7dbl_1(x)

        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)

        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)

        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)

        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)



        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)



        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionD(nn.Module):



    def __init__(self, in_channels):

        super(InceptionD, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)



        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)



    def forward(self, x):

        branch3x3 = self.branch3x3_1(x)

        branch3x3 = self.branch3x3_2(branch3x3)



        branch7x7x3 = self.branch7x7x3_1(x)

        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)

        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)

        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)



        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]

        return torch.cat(outputs, 1)





class InceptionE(nn.Module):



    def __init__(self, in_channels):

        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)



        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)

        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))

        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))



        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)

        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)

        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))

        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))



        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)



    def forward(self, x):

        branch1x1 = self.branch1x1(x)



        branch3x3 = self.branch3x3_1(x)

        branch3x3 = [

            self.branch3x3_2a(branch3x3),

            self.branch3x3_2b(branch3x3),

        ]

        branch3x3 = torch.cat(branch3x3, 1)



        branch3x3dbl = self.branch3x3dbl_1(x)

        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch3x3dbl = [

            self.branch3x3dbl_3a(branch3x3dbl),

            self.branch3x3dbl_3b(branch3x3dbl),

        ]

        branch3x3dbl = torch.cat(branch3x3dbl, 1)



        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)



        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionAux(nn.Module):



    def __init__(self, in_channels, num_classes):

        super(InceptionAux, self).__init__()

        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)

        self.conv1 = BasicConv2d(128, 768, kernel_size=5)

        self.conv1.stddev = 0.01

        self.fc = nn.Linear(768, num_classes)

        self.fc.stddev = 0.001



    def forward(self, x):

        # 17 x 17 x 768

        x = F.avg_pool2d(x, kernel_size=5, stride=3)

        # 5 x 5 x 768

        x = self.conv0(x)

        # 5 x 5 x 128

        x = self.conv1(x)

        # 1 x 1 x 768

        x = x.view(x.size(0), -1)

        # 768

        x = self.fc(x)

        # 1000

        return x





class BasicConv2d(nn.Module):



    def __init__(self, in_channels, out_channels, **kwargs):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        return F.relu(x, inplace=True)
class AttackIterative:



    def __init__(

            self,

            max_epsilon=16, norm=float('inf'), step_alpha=None, 

            num_steps=None, cuda=True, debug=False):



        self.eps = 2.0 * max_epsilon / 255.0

        self.num_steps = num_steps or 10

        self.norm = norm

        if not step_alpha:

            if norm == float('inf'):

                self.step_alpha = self.eps / self.num_steps

            else:

                # Different scaling required for L2 and L1 norms to get anywhere

                if norm == 1:

                    self.step_alpha = 500.0  # L1 needs a lot of (arbitrary) love

                else:

                    self.step_alpha = 1.0

        else:

            self.step_alpha = step_alpha

        self.loss_fn = torch.nn.CrossEntropyLoss()

        if cuda:

            self.loss_fn = self.loss_fn.cuda()

        self.debug = debug



    def non_target_attack(self, non_target_model, x, targets, batch_idx=0):

        input_var = autograd.Variable(x, requires_grad=True)

        targets_var = autograd.Variable(targets)

        eps = self.eps

        step_alpha = self.step_alpha



        step = 0

        while step < self.num_steps:

            zero_gradients(input_var)

            output = non_target_model(input_var)

            if not step:

                # for non-targeted, we'll move away from most likely

                targets_var.data = output.data.max(1)[1]

            loss = self.loss_fn(output, targets_var)

            loss.backward()



            # normalize and scale gradient

            if self.norm == 2:

                normed_grad = step_alpha * input_var.grad.data / l2_norm(input_var.grad.data)

            elif self.norm == 1:

                normed_grad = step_alpha * input_var.grad.data / l1_norm(input_var.grad.data)

            else:

                # infinity-norm

                normed_grad = step_alpha * torch.sign(input_var.grad.data)



            # perturb current input image by normalized and scaled gradient

            step_adv = input_var.data + normed_grad



            # calculate total adversarial perturbation from original image and clip to epsilon constraints

            total_adv = step_adv - x

            total_adv = torch.clamp(total_adv, -eps, eps)

            

            if self.debug:

                print('Non-Targeted --', 'batch:', batch_idx, 'step:', step, total_adv.mean(), total_adv.min(), total_adv.max())

                sys.stdout.flush()



            # apply total adversarial perturbation to original image and clip to valid pixel range

            input_adv = x + total_adv

            input_adv = torch.clamp(input_adv, -1.0, 1.0)

            input_var.data = input_adv

            step += 1



        return input_adv.permute(0, 2, 3, 1).cpu().numpy()

    

    def target_attack(self, target_model, x, targets, batch_idx=0):

        input_var = autograd.Variable(x, requires_grad=True)

        targets_var = autograd.Variable(targets)

        eps = self.eps

        step_alpha = self.step_alpha



        step = 0

        while step < self.num_steps:

            zero_gradients(input_var)

            output = target_model(input_var)

            loss = self.loss_fn(output, targets_var)

            loss.backward()



            # normalize and scale gradient

            if self.norm == 2:

                normed_grad = step_alpha * input_var.grad.data / l2_norm(input_var.grad.data)

            elif self.norm == 1:

                normed_grad = step_alpha * input_var.grad.data / l1_norm(input_var.grad.data)

            else:

                # infinity-norm

                normed_grad = step_alpha * torch.sign(input_var.grad.data)



            # perturb current input image by normalized and scaled gradient

            step_adv = input_var.data - normed_grad



            # calculate total adversarial perturbation from original image and clip to epsilon constraints

            total_adv = step_adv - x

            total_adv = torch.clamp(total_adv, -eps, eps)

            

            if self.debug:

                print('Targeted --', 'batch:', batch_idx, 'step:', step, total_adv.mean(), total_adv.min(), total_adv.max())

                sys.stdout.flush()



            # apply total adversarial perturbation to original image and clip to valid pixel range

            input_adv = x + total_adv

            input_adv = torch.clamp(input_adv, -1.0, 1.0)

            input_var.data = input_adv

            step += 1



        return input_adv.permute(0, 2, 3, 1).cpu().numpy()

        

    def run(self, non_target_model, target_model, x, true_targets, fake_targets, batch_idx=0):

        non_target_pred = self.non_target_attack(non_target_model, x, true_targets, batch_idx=0)

        target_pred = self.target_attack(target_model, x, fake_targets, batch_idx=0)

        return (non_target_pred, target_pred)
def make_prediction(model, output_file):

    dataset = OneShotDataset(

            output_file,

            transform=default_inception_transform(args["img_size"]))

    loader = data.DataLoader(

        dataset,

        batch_size=1)

    # one shot

    for _batch_idx, (tensor, _target) in enumerate(loader): 

        input_var = autograd.Variable(tensor, requires_grad=True)

        zero_gradients(input_var)

        output = model(input_var)

        _, preds = torch.max(output.data, 1)

    return preds
def make_md5(img_file):

    m = hashlib.md5()

    img = Image.open(img_file)

    with io.BytesIO() as memf:

        img.save(memf, 'PNG')

        data = memf.getvalue()

        m.update(data)

    return m.hexdigest()



def display_attacks(attacks, fake_targets, cols):

    f, axs = plt.subplots(nrows=len(attacks),ncols=3,figsize=(15,22))

    cat_df = pd.read_csv('../input/nips-2017-adversarial-learning-development-set/categories.csv')

    

    for ax, col in zip(axs[0], cols):

        ax.annotate(col, xy=(0.5, 1.1), xytext=(0, 1),

                xycoords='axes fraction', textcoords='offset points',

                size='large', ha='center', va='baseline')

    

    for i, row in enumerate(axs):

        for j, col in enumerate(row):

            img, label, md5 = attacks[i][j]

            col.imshow(img)

            target = cat_df.iloc[label-1][1].split(",")[0]

            title = 'Labeled: '+target

            if j == 2:

                fake_target = cat_df.iloc[fake_targets[i]-1][1].split(",")[0]

                title += '. Target: '+fake_target

            col.set_title(title)

            col.annotate('MD5: '+md5, xy=(0.5, -0.12), xytext=(0, 1),

                xycoords='axes fraction', textcoords='offset points',

                size='large', ha='center', va='baseline')

    plt.show()
import torchvision.models

import matplotlib.image as mpimg

def run_iterative_attack(args, attack):

    assert args["input_dir"]



    dataset = Dataset(

        args["input_dir"],

        transform=default_inception_transform(args["img_size"]))



    loader = data.DataLoader(

        dataset,

        batch_size=args["batch_size"],

        shuffle=False)



    # train from scratch

    non_target_gen_model = inception_v3(pretrained=False, transform_input=True)

    target_gen_model     = inception_v3(pretrained=False, transform_input=True)

    if args["cuda"]:

        non_target_gen_model = non_target_gen_model.cuda()

        target_gen_model     = target_gen_model.cuda()



    # pick up feature hierarchy from checkpoint

    if args["checkpoint_path"] is not None and os.path.isfile(args["checkpoint_path"]):

        checkpoint = torch.load(args["checkpoint_path"])

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:

            non_target_gen_model.load_state_dict(checkpoint['state_dict'])

            target_gen_model.load_state_dict(checkpoint['state_dict'])

        else:

            non_target_gen_model.load_state_dict(checkpoint)

            target_gen_model.load_state_dict(checkpoint)

    else:

        print("Error: No checkpoint found at %s." % args["checkpoint_path"])



    non_target_gen_model.eval()

    target_gen_model.eval()

    

    # pretrained discriminatory model

    dis_model = inception_v3(pretrained=True)

    dis_model.eval()

    

    attacks = []                        

    # run both non-targeted and targeted attacks

    for batch_idx, (input, true_targets) in enumerate(loader):    

        start_index = args["batch_size"] * batch_idx

        indices = list(range(start_index, start_index + input.size(0)))



        # spawn 4 random classes. For one batch (size=4), unlikely fake==true (4/1000=1/250)

        fake_targets = np.random.randint(1, 1001 + 1, size=4)

        fake_targets = torch.from_numpy(fake_targets)

        

        if args["cuda"]:

            input = input.cuda()

            true_targets = true_targets.cuda()

            fake_targets = fake_targets.cuda()



        (non_target_adv, target_adv) = attack.run(non_target_gen_model, target_gen_model, 

                                                  input, true_targets, fake_targets, batch_idx)

        

        for i, (filename, non_target_o, target_o) in enumerate(

            zip(dataset.filenames(indices, basename=True), non_target_adv, target_adv)):

            # get and save non-targeted adversary image, label, and hash

            non_target_img = (non_target_o + 1.0) * 0.5

            non_target_output_file = os.path.join(args["output_dir"], "non_target_" + filename)

            imsave(non_target_output_file, non_target_img, format='png')     

            non_target_hash  = make_md5(non_target_output_file)

            non_target_label = make_prediction(dis_model, non_target_output_file)[0]

            

            # get and save targeted adversary image, label, and hash

            target_img = (target_o + 1.0) * 0.5

            target_output_file = os.path.join(args["output_dir"], "target_" + filename)

            imsave(target_output_file, target_img, format='png')     

            target_hash  = make_md5(target_output_file)

            target_label = make_prediction(dis_model, target_output_file)[0]

            

            # get original image, label, and hash

            og_img = mpimg.imread(args["input_dir"]+filename)

            og_file = os.path.join(args["input_dir"], filename)

            og_label = make_prediction(dis_model, og_file)[0]

            og_hash  = make_md5(og_file)

            attacks.append(((og_img, og_label, og_hash), 

                            (non_target_img, non_target_label, non_target_hash), 

                            (target_img, target_label, target_hash)))

              

        # only one batch for the notebook

        display_attacks(attacks, fake_targets, cols=["Original","Non-Targeted", "Targeted"])

        break 
args = {}

args["targeted"]=True

args["input_dir"]='../input/nips-2017-adversarial-learning-development-set/images'

args["max_epsilon"]=5 # serves the purpose of the demo

args["norm"]=0 

args["step_alpha"]=0.01

args["num_steps"]=10 

args["debug"]=False

args["img_size"]=299

args["batch_size"]=4

args["checkpoint_path"]='pytorch-nips2017-attack-example-master/inception_v3_google-1a9a5a14.pth' # Need this file...

args["no_gpu"]=False

args["output_dir"]="output/"

args["cuda"]=False # quick run on laptop
attack = AttackIterative(

        max_epsilon=args["max_epsilon"],

        norm=args["norm"],

        step_alpha=args["step_alpha"],

        num_steps=args["num_steps"],

        cuda=args["cuda"],

        debug=args["debug"])



#print(torchvision.models.__dict__)

run_iterative_attack(args, attack)





# see https://github.com/Cpruce/Notebooks/blob/master/BattleJazzGANs.ipynb