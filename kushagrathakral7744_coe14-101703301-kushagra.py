# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from tqdm import tqdm

from keras.applications import inception_resnet_v2

from keras.preprocessing import image

from mpl_toolkits.axes_grid1 import ImageGrid

from keras.preprocessing.image import ImageDataGenerator

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)

!cp ../input/keras-pretrained-models/inception_resnet_v2* ~/.keras/models/
classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

n_classes = len(classes)



sample_per_class = 200

seed = 1987

data_dir = '../input/plant-seedlings-classification/'

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))



sample_submission.head()
for each_class in classes:

    print('{} {} images'.format(each_class, len(os.listdir(os.path.join(train_dir, each_class)))))
train = []

for category_id, category in enumerate(classes):

    for file in os.listdir(os.path.join(train_dir, category)):

        train.append(['train/{}/{}'.format(category, file), category_id, category])

train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])

train.head(2)

train.shape
train = pd.concat([train[train['category'] == c][:sample_per_class] for c in classes])

train = train.sample(frac=1)

train.index = np.arange(len(train))

train.head(2)

train.shape
test = []

for file in os.listdir(test_dir):

    test.append(['test/{}'.format(file), file])

test = pd.DataFrame(test, columns=['filepath', 'file'])

test.head(2)

test.shape
def read_img(filepath, size):

    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)

    img = image.img_to_array(img)

    return img
np.random.seed(seed=seed)

rnd = np.random.random(len(train))

train_idx = rnd < 0.8

valid_idx = rnd >= 0.8

ytr = train.loc[train_idx, 'category_id'].values

yv = train.loc[valid_idx, 'category_id'].values

len(ytr), len(yv)

INPUT_SIZE = 299

POOLING = 'avg'

x_train = np.zeros((len(train), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, file in tqdm(enumerate(train['file'])):

    img = read_img(file, (INPUT_SIZE, INPUT_SIZE))

    x = inception_resnet_v2.preprocess_input(np.expand_dims(img.copy(), axis=0))

    x_train[i] = x

print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr = x_train[train_idx]

Xv = x_train[valid_idx]

print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))

inception_bottleneck = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling=POOLING)

train_x_bf = inception_bottleneck.predict(Xtr, batch_size=32, verbose=1)

valid_x_bf = inception_bottleneck.predict(Xv, batch_size=32, verbose=1)

print('Inception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))

print('Inception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)

logreg.fit(train_x_bf, ytr)

valid_probs = logreg.predict_proba(valid_x_bf)

valid_preds = logreg.predict(valid_x_bf)

print('Validation Xception Accuracy {}'.format(accuracy_score(yv, valid_preds)))

x_test = np.zeros((len(test), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, filepath in tqdm(enumerate(test['filepath'])):

    img = read_img(filepath, (INPUT_SIZE, INPUT_SIZE))

    x = inception_resnet_v2.preprocess_input(np.expand_dims(img.copy(), axis=0))

    x_test[i] = x

print('test Images shape: {} size: {:,}'.format(x_test.shape, x_test.size))



test_x_bf = inception_bottleneck.predict(x_test, batch_size=32, verbose=1)

print('Xception test bottleneck features shape: {} size: {:,}'.format(test_x_bf.shape, test_x_bf.size))

test_preds = logreg.predict(test_x_bf)

test['category_id'] = test_preds

test['species'] = [classes[c] for c in test_preds]

test[['file', 'species']].to_csv('submission.csv', index=False)