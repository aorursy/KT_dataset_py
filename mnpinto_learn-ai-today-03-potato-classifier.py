# Install fastai2 (note that soon fastai2 will be officially released as fastai)

!pip install fastai2
from fastai2.vision.all import *
path = Path('/kaggle/input/fruits/fruits-360/')

train_fld = 'Training'
train_files = [o.ls() for o in (path/train_fld).ls() if 'Potato' in o.name]

train_files = [l for o in train_files for l in o]

len(train_files)
train_df = pd.DataFrame({'file': train_files, 'id': [o.parent.name for o in train_files]})

train_df.head()
dblock = DataBlock(blocks     = (ImageBlock, CategoryBlock),

                   get_x      = ColReader('file'),

                   get_y      = ColReader('id'),

                   splitter   = RandomSplitter(),

                   item_tfms  = Resize(64),

                   batch_tfms = [Normalize.from_stats(*imagenet_stats), *aug_transforms()])
dls = dblock.dataloaders(train_df)
dls.show_batch()
class BasicCNN(Module):

    def __init__(self, n_classes):

        self.conv1      = nn.Conv2d(3, 32, kernel_size=3)

        self.conv2      = nn.Conv2d(32, 64, kernel_size=3)

        self.avgpool    = nn.AdaptiveAvgPool2d(1)

        self.fc         = nn.Linear(64, n_classes)



    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = self.avgpool(x)

        x = self.fc(x.view(-1, x.shape[1]))

        return x
model = BasicCNN(n_classes=dls.c)

learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy])
learn.summary()
learn.fit_one_cycle(30, lr_max=3e-3, wd=1e-2)
plt.figure(figsize=(10,5), dpi=150)

learn.recorder.plot_loss()

plt.xlabel('iterations')

plt.ylabel('loss')