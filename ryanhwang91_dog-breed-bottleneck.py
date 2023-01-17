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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, models, layers, optimizers, metrics
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from keras.applications import xception
from os import listdir, makedirs
from os.path import join, exists, expanduser
import pandas as pd
import datetime as dt
np.random.seed(3)
tf.compat.v1.set_random_seed(3)
!ls ../input/keras-pretrained-models/
start = dt.datetime.now()
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
!ls ~/.keras/models
!ls ../input/dog-breed-identification
INPUT_SIZE = 224
NUM_CLASSES = 5
SEED = 1987
data_dir = '../input/dog-breed-identification'
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print(len(listdir(join(data_dir, 'train'))), len(labels))
print(len(listdir(join(data_dir, 'test'))), len(sample_submission))
print(labels)
NUM_CLASSES = 20
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
print(labels)
print(selected_breed_list)
img_path = os.path.join(data_dir,'train/')
labels = labels.assign(img_path = lambda x : img_path + x['id']+'.jpg')
labels = labels.assign(file_name = lambda x : x['id']+'.jpg')
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(labels[['id','img_path','file_name']], labels['breed'], test_size=0.2, random_state=42)
print(X_train.head())
print(y_train.head())
# train = pd.DataFrame({"X":X_train['id'],"X_path":X_train['img_path'],"Y":y_train})
train = pd.DataFrame({"X":X_train['id'],"X_path":X_train['img_path'],"X_filename":X_train['file_name'],"Y":y_train})
train.head()
test = pd.DataFrame({"X":X_test['id'],"X_path":X_test['img_path'],"X_filename":X_test['file_name'],"Y":y_test})
test.head()
train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                   vertical_flip=True,                                   
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                   rotation_range=5,
                                   shear_range=0.7,
                                   zoom_range=1.2,
                                  fill_mode='nearest')

# train_generator = train_datagen.flow_from_directory(
#        '../input/dog-breed-identification/train',
#        target_size=(150, 150),
#        batch_size=50,
#        class_mode='categorical')

train_generator = train_datagen.flow_from_dataframe(train, x_col="X_path",
                                                    y_col="Y",
                                                    target_size=(299, 299),
                                                    class_mode='categorical',
                                                    batch_size=32,
                                                    seed=33,
                                                    interpolation='nearest',)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(test, x_col="X_path",
                                                    y_col="Y",
                                                    target_size=(299, 299),
                                                    class_mode='categorical',
                                                    batch_size=32,
                                                    seed=33,
                                                    interpolation='nearest',)


# test_generator = test_datagen.flow_from_directory(
#        '../input/dog-breed-identification/test',
#        target_size=(150, 150),
#        batch_size=50,
#        class_mode='categorical')
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
from tqdm import tqdm
from keras.preprocessing import image
INPUT_SIZE = 299
POOLING = 'avg'
x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
for i, img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))
labels['target'] = 1
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]
Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))
xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
print('Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))
print(Xtr)
print(Xv)
print(ytr)
print(yv)
print((ytr * range(NUM_CLASSES)).sum(axis=1))
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_x_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)
print('Validation Xception LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation Xception Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))

from keras.models import load_model
import pickle


saved_model = pickle.dumps(logreg)
log_from_pickle = pickle.loads(saved_model)
log_from_pickle.predict(valid_x_bf)
log_from_pickle.predict_proba(valid_x_bf)
# import sklearn.external.joblib as extjoblib
import joblib
joblib.dump(logreg, 'god_log.pkl') 
log_from_joblib = joblib.load('god_log.pkl') 
log_from_joblib.predict(valid_x_bf)
crawling = pd.read_csv('../input/crawling-data1/fin_pure_data1_891.csv',encoding='cp949')
crawling.head()
os.listdir('../input/crawlingdata1/')
import re
matchObj = re.search('\d{4}-\d{5}', '░°░φ╣°╚ú-┴ª┴╓-┴ª┴╓-2019-06205 ║╨╜╟╡╚╡┐╣░╗τ┴°.jpg')
matchObj.group()
crawling.info()
crawling[['notice_num']]

crawling['keys'] = crawling['notice_num'].str.slice(start=6, stop=16) # 인덱스 사이 값 반환
crawling.head()
data_dir = '../input/img-renamed/crawling_photo/'
def read_img(img_id, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img
x_pred = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

img = read_img('2019-00108', (INPUT_SIZE, INPUT_SIZE))

x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
x_pred[0] = x
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))
train_x_bf = xception_bottleneck.predict(x_pred, batch_size=32, verbose=1)
pred_dog = log_from_joblib.predict(train_x_bf)
selected_breed_list[pred_dog[0].astype(int)]
print(crawling[crawling['keys']=='2019-00108'])
valid_breeds = (yv * range(NUM_CLASSES)).sum(axis=1)
error_idx = (valid_breeds != valid_preds)
for img_id, breed, pred in zip(labels.loc[valid_idx, 'id'].values[error_idx],
                                [selected_breed_list[int(b)] for b in valid_preds[error_idx]],
                                [selected_breed_list[int(b)] for b in valid_breeds[error_idx]]):
    fig, ax = plt.subplots(figsize=(5,5))
    img = read_img(img_id, 'train', (299, 299))
    ax.imshow(img / 255.)
    ax.text(10, 250, 'Prediction: %s' % pred, color='w', backgroundcolor='r', alpha=0.8)
    ax.text(10, 270, 'LABEL: %s' % breed, color='k', backgroundcolor='g', alpha=0.8)
    ax.axis('off')
    plt.show()                                                    
