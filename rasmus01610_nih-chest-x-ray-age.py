from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from fastai.vision import *
from fastai import *
from torch import nn
import torchvision
PATH = "../input/"
data_csv = pd.read_csv(PATH + 'Data_Entry_2017.csv'); data_csv.head()
images = glob(os.path.join('..', 'input', 'images*', '*', '*.png'))
df = pd.DataFrame({'fn': images, 'age': data_csv["Patient Age"].astype(float)}, columns=['fn','age'])
df = df[df["age"]<100][:1000]
tfms = get_transforms(do_flip=False)
md3 = (ImageItemList.from_df(df,".")
       .random_split_by_pct(0.2, seed=1337)
      .label_from_df()
      .transform(tfms, size=224)
      .databunch(bs=16, num_workers=0)
      .normalize(imagenet_stats))
md3.show_batch(3)
max_age = df["age"].max(); max_age
min_age = df["age"].min(); min_age
class AgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(models.resnet34().children())[:-2]
        layers += [AdaptiveConcatPool2d(), Flatten()]
        layers += [nn.Linear(1024,16), nn.ReLU(), nn.Linear(16,1)]
        self.agemodel = nn.Sequential(*layers)
    def forward(self, x):
        x = self.agemodel(x).squeeze()
        return torch.sigmoid(x) * (max_age - min_age) + min_age

model = AgeModel()
learner = Learner(md3, model, loss_func = F.l1_loss)
learner.split([model.agemodel[2], model.agemodel[-3]])
learner.freeze_to(-1)
learner.fit_one_cycle(4)
learner.show_results(rows=3)
df["age"].mean()
