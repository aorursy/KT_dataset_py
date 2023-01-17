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

df = pd.read_csv(path / 'labels.csv')

df['fname'] = df['name']

df.head(2)
cr_label = ColReader('label')

cr_name  = ColReader('name')

cr_fname = ColReader('fname')



row0 = df.iloc[0,:]

row0
cr_label(row0), cr_name(row0), cr_fname(row0)
getattr(row0, 'name'), getattr(row0, 'fname')
row0.name, row0.fname
tmp = pd.DataFrame({'a':[1,2],'b':[3,4]})

tmp0 = tmp.iloc[0,:]

tmp0
tmp0.name
row0['name']
row0['fname'], row0['label']
dblock = DataBlock(

                blocks=(ImageBlock, CategoryBlock),

                get_x = ColReader('name', pref=path),

                get_y=ColReader('label'),

                   )



dls2 = dblock.dataloaders(df, num_workers=1)
# This works because we use `fname` 

# instead of `name` for get_x



dblock = DataBlock(

                blocks=(ImageBlock, CategoryBlock),

                get_x = ColReader('fname', pref=path),

                get_y=ColReader('label'),

                   )



dl = dblock.dataloaders(df, num_workers=1)



len(dl.train.items)