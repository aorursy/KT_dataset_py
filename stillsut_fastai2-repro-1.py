%load_ext autoreload

%autoreload 2



import os, sys

import pandas as pd



my_env = os.environ.get('USER', 'KAGGLE')

b_kaggle = (my_env == 'KAGGLE')



if b_kaggle:

    os.system('pip install fastai2')



from fastai2.vision.all import *



import fastai2

fastai2.__version__
path = untar_data(URLs.MNIST_TINY)

dl = ImageDataLoaders.from_folder(path, num_workers=0)

learn = cnn_learner(dl, resnet18, pretrained=True, )

with learn.no_logging(): learn.fit_one_cycle(1)

interp = ClassificationInterpretation.from_learner(learn)
# this is good, the vocab is strings

learn.dls.vocab
# therefore this function works

interp.print_classification_report()
df = pd.read_csv(path / 'labels.csv')

df['fname'] = df['name']
dblock = DataBlock(

                blocks=(ImageBlock, CategoryBlock),

                get_x = ColReader('fname', pref=path),

                get_y=ColReader('label'),

                   )



dls2 = dblock.dataloaders(df, num_workers=1)
learn = cnn_learner(dls2, resnet18, pretrained=True, )

with learn.no_logging(): learn.fit_one_cycle(1)

interp = ClassificationInterpretation.from_learner(learn)
# this is bad, the vocab is integers

learn.dls.vocab
# therefore this fails

interp.print_classification_report()
df.dtypes
if b_kaggle:

    os.system('pip install git+git://github.com/sutt/fastai2.git')
df = pd.read_csv(path / 'labels.csv')

df['fname'] = df['name']



dblock = DataBlock(

                blocks=(ImageBlock, CategoryBlock),

                get_x = ColReader('fname', pref=path),

                get_y=ColReader('label'),

                   )



dls2 = dblock.dataloaders(df, num_workers=1)



learn = cnn_learner(dls2, resnet18, pretrained=True, )

with learn.no_logging(): learn.fit_one_cycle(1)

interp = ClassificationInterpretation.from_learner(learn)
# this is bad, the vocab is integers

learn.dls.vocab
# therefore this fails

interp.print_classification_report()