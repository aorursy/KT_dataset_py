!pip install --upgrade fastai2 > /dev/null

!pip install --upgrade fastcore > /dev/null

!pip install pretrainedmodels > /dev/null
from fastai2.vision.all import *

import pretrainedmodels
panda_block = DataBlock(blocks=(ImageBlock, CategoryBlock),

                        splitter=RandomSplitter(),

                        get_x=ColReader('image_id',pref=Path('../input/panda-challenge-512x512-resized-dataset'),suff='.jpeg'),

                        get_y=ColReader('isup_grade'),

                        item_tfms=Resize(256),

                        batch_tfms=aug_transforms()

                       )
train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
train_df.head()
dls = panda_block.dataloaders(train_df,bs=16)
dls.show_batch()
m = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')

children = list(m.children())

head = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), 

                                  nn.Linear(children[-1].in_features,200))

model = nn.Sequential(nn.Sequential(*children[:-2]), head)
learn = Learner(dls,model,splitter=default_split,metrics=[accuracy,CohenKappa(weights='quadratic')])
learn.freeze()

learn.lr_find()
learn.freeze()

learn.fit_one_cycle(5,6e-3)
learn.save('stage-1-512.pth')
learn.load('stage-1-512.pth')
learn.unfreeze()

learn.lr_find()
learn.unfreeze()

learn.fit_one_cycle(5,slice(1e-6,5e-5))
learn.save('stage-2-512.pth')
learn.recorder.plot_loss()
learn.export()