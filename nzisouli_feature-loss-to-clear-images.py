# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

import fastai

from fastai.vision import *

from fastai.callbacks import *

from fastai.vision.gan import *

from torchvision.models import vgg16_bn, vgg16
path = Path("/kaggle/input/dance-images")

path_cl= path/"clear"

path_bl= path/"blurry"
bs,size=4, 512

arch = models.resnet34
src = ImageImageList.from_folder(path_bl).split_by_rand_pct(0.1, seed=42)
def get_data(bs,size):

    data = (src.label_from_func(lambda x: path_cl/x.name)

           .transform(get_transforms(max_zoom=0), size=size, tfm_y=True)

           .databunch(bs=bs, num_workers=0).normalize(imagenet_stats, do_y=True))



    #data.c = 3

    return data
data = get_data(bs,size)

data.show_batch(4)
t = data.valid_ds[0][1].data

t = torch.stack([t,t])
def gram_matrix(x):

    n,c,h,w = x.size()

    x = x.view(n, c, -1)

    return (x @ x.transpose(1,2))/(c*h*w)
base_loss = F.mse_loss


imgnet = vgg16(True).features.cuda().eval()

requires_grad(imgnet, False)
#index of layers just before MaxPool

blocks = [i-1 for i,o in enumerate(children(imgnet))

          if (isinstance(o,nn.MaxPool2d) or (isinstance(o,nn.Conv2d) and o.stride==(2,2))) and i>0]

blocks, [imgnet[i] for i in blocks]
class FeatureLoss(nn.Module):

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
feat_loss = FeatureLoss(imgnet, blocks[2:5], [5,1,11])
y_range = (-3.,3.)

learn = unet_learner(data, arch, loss_func=feat_loss, callback_fns=LossMetrics,

                     blur=True, norm_type=NormType.Weight, model_dir="/tmp/model/")

gc.collect();
learn.lr_find()

learn.recorder.plot()
gc.collect()
lr = 3e-3

learn.fit_one_cycle(10, lr)
gc.collect()

learn.unfreeze()

learn.fit_one_cycle(10, slice(1e-4,lr))

learn.show_results(rows=1, imgsize=5)
bs, size = 1, 1024

data = get_data(bs, size)

learn.data = data

gc.collect()
learn.freeze()

learn.fit_one_cycle(10, lr)
gc.collect()

learn.unfreeze()

learn.fit_one_cycle(10, slice(1e-4,lr))

learn.show_results(imgsize=5)
gc.collect()

learn.save("feature_loss")