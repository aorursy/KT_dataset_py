import pandas as pd



from fastai.vision import *

from fastai.metrics import error_rate

from scipy.io import loadmat
!wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz

!tar xvzf car_devkit.tgz
unpack_double = lambda x: x[0][0]

unpack_single = lambda x: x[0]



def prepare_df(path, singles = [], doubles = []):

    annos = loadmat(path)['annotations']

    df = pd.DataFrame(annos[0])

    

    for item in singles:

        df[item] = df[item].apply(unpack_single)

    

    for item in doubles:

        df[item] = df[item].apply(unpack_double)



    return df[singles + doubles]



train = prepare_df('devkit/cars_train_annos', ['fname'], ['class'])

test = prepare_df('devkit/cars_test_annos', ['fname'])



train['name'] = train['fname'].apply(lambda name: '../input/cars_train/cars_train/' + name)

test['name'] = test['fname'].apply(lambda name: '../input/cars_test/cars_test/' + name)



del train['fname']

del test['fname']



train['class'] = train['class'].apply(lambda x: x-1)



meta = pd.DataFrame(loadmat('devkit/cars_meta.mat')['class_names'][0], columns=['class_name'])

meta['class_name'] = meta['class_name'].apply(unpack_single)



train['label'] = train['class'].apply(lambda x: meta.iloc[x].class_name)

del train['class']
train.head()
data = ImageDataBunch.from_df('', train, ds_tfms=get_transforms(), size=224, bs=64).normalize(imagenet_stats)

data