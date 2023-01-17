%matplotlib inline

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs

from os.path import join, exists, expanduser

from tqdm import tqdm

from sklearn.metrics import log_loss, accuracy_score

from keras.preprocessing import image

from keras.applications import xception
LABELS = "../input/dog-breed-identification/labels.csv"



train_df = pd.read_csv(LABELS)

#return top 16 value counts and convert into list

plt.figure(figsize=(13, 6))

train_df['breed'].value_counts().plot(kind='bar')

plt.show()



top_breeds = sorted(list(train_df['breed'].value_counts().head(16).index))

train_df = train_df[train_df['breed'].isin(top_breeds)]



print(top_breeds)
INPUT_SIZE = 224

NUM_CLASSES = 16

SEED = 1987

data_dir = '../input/dog-breed-identification'

labels = pd.read_csv(join(data_dir, 'labels.csv'))

sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))

print(len(listdir(join(data_dir, 'train'))), len(labels))

print(len(listdir(join(data_dir, 'test'))), len(sample_submission))
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

labels = labels[labels['breed'].isin(selected_breed_list)]

labels['target'] = 1

#labels['rank'] = labels.groupby('breed').rank()['id']

labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

np.random.seed(seed=SEED)

rnd = np.random.random(len(labels))

train_idx = rnd < 0.8

valid_idx = rnd >= 0.8

y_train = labels_pivot[selected_breed_list].values

ytr = y_train[train_idx]

yv = y_train[valid_idx]
def read_img(img_id, train_or_test, size):

    """Read and resize image.

    # Arguments

        img_id: string

        train_or_test: string 'train' or 'test'.

        size: resize the original image.

    # Returns

        Image as numpy array.

    """

    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)

    img = image.img_to_array(img)

    return img
cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

    

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
INPUT_SIZE = 299

POOLING = 'avg'

x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, img_id in tqdm(enumerate(labels['id'])):

    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))

    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))

    x_train[i] = x

print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))
Xtr = x_train[train_idx]

Xv = x_train[valid_idx]

print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)

train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)

valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)

print('Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))

print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)

logreg.fit(train_x_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))

valid_probs = logreg.predict_proba(valid_x_bf)

valid_preds = logreg.predict(valid_x_bf)

print('Validation Xception LogLoss {}'.format(log_loss(yv, valid_probs)))

print('Validation Xception Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))
  