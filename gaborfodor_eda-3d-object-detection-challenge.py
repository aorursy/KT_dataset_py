%matplotlib inline

import os

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import datetime as dt

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import warnings

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')

pd.options.display.max_columns = 99

sns.set_palette(sns.color_palette('tab20', 20))
start = dt.datetime.now()
base = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'

dirs = os.listdir(base)

print(dirs)
train = pd.read_csv(base + 'train.csv')

sample_submission = pd.read_csv(base + 'sample_submission.csv')

print(f'train: {train.shape}, sample submission: {sample_submission.shape}')

train.head(2)

sample_submission.head(2)
# Let's check the parsing of prediction strings. Each object should have 8 params

max([len(ps.split(' ')) % 8 for ps in train.PredictionString.values])
object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z',

                  'width', 'length', 'height', 'yaw', 'class_name']

objects = []

for sample_id, ps in tqdm(train.values[:]):

    object_params = ps.split()

    n_objects = len(object_params)

    for i in range(n_objects // 8):

        x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])

        objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])

train_objects = pd.DataFrame(

    objects,

    columns = object_columns

)
for col in object_columns[2:-1]:

    train_objects[col] = train_objects[col].astype('float')

train_objects['confidence'] = 1.0
train_objects.groupby('sample_id').count()[['object_id']].hist()

plt.title('Number of objects per sample')

plt.show();
train_objects.shape

train_objects.head()

train_objects.describe()
fig, ax = plt.subplots(ncols=3)

sns.distplot(train_objects.center_x, ax = ax[0])

sns.distplot(train_objects.center_y, ax = ax[1])

sns.distplot(train_objects.center_z, ax = ax[2])

plt.suptitle('X, y, z coord distribution')

plt.show();
fig, ax = plt.subplots(ncols=3)

sns.distplot(train_objects.width, ax = ax[0])

sns.distplot(train_objects.length, ax = ax[1])

sns.distplot(train_objects.height, ax = ax[2])

plt.suptitle('Width, length, height distribution')

plt.show();
class_cnt = train_objects.groupby('class_name').count()[['object_id']].sort_values(by='object_id', ascending=False).reset_index()

class_cnt['p'] = class_cnt.object_id / class_cnt.object_id.sum() 

class_cnt
train_objects.groupby('class_name').mean()
x, y, z, w, l, h, yaw = train_objects[[

    'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']].mean()

mean_prediction_string = ' '.join(map(str, [0.9, x, y, z, 10*w, 10*l, h, yaw, 'car']))

sample_submission['PredictionString'] = mean_prediction_string 

sample_submission.to_csv('submission.csv', index=False)
sample_submission.shape

sample_submission.head()
for f in os.listdir(base + 'train_data'):

    print(f)

    try:

        df = pd.read_json(base + 'train_data/' + f)

        df.shape

        df.head()

        df.nunique()

    except Exception as e:

        print(e)
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))