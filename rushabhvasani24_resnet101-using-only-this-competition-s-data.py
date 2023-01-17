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
from fastai.vision import *

from torchvision import transforms

from fastai.vision.image import *

import torchvision
Df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
dev = torch.device("cuda:0")

data_bunch = (ImageList.from_df(Df,"../input/aptos2019-blindness-detection/train_images",cols = 0, suffix = '.png').

              split_none().

              label_from_df(1,label_cls = FloatList).

              transform(get_transforms(do_flip = True, flip_vert = True), size = 512).

              databunch(bs = 8).

              normalize(imagenet_stats))
model = torchvision.models.resnet101(pretrained=False)

model.load_state_dict(torch.load("../input/pytorch-models/resnet101-5d3b4d8f.pth"))

num_features = model.fc.in_features

model.fc = nn.Linear(2048, 1)

model = model.to(dev)
learn= Learner(data_bunch,model, loss_func = mse, metrics=mse, model_dir="../../../models")
learn.fit_one_cycle(5,2.5e-4)
learn.fit_one_cycle(5,2.5e-4)
learn.unfreeze()
learn.fit_one_cycle(5,2.5e-5)
learn.fit_one_cycle(5,1e-6)
Tf = partial(Image.apply_tfms,tfms=get_transforms(do_flip=True, flip_vert = True)[0][1:]+get_transforms(do_flip=True, flip_vert = True)[1],size = 512)
sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
for i in range(len(sub.id_code)):

    s = 0

    img = open_image("../input/aptos2019-blindness-detection/test_images/"+sub.id_code[i]+'.png')

    for i in range(64):

        Img = Tf(img)

        p = learn.predict(Img)

        s+=p[1]

    s = s/64.0

    if s<0.5:

        sub.diagnosis[i]=0

    elif s>=0.4 and s<1.4:

        sub.diagnosis[i]=1

    elif s>=1.4 and s <2.4:

        sub.diagnosis[i]=2

    elif s>=2.4 and s <3.4:

        sub.diagnosis[i]=3

    else:

        sub.diagnosis[i]=4

sub.to_csv("submission.csv",index=False)
sub