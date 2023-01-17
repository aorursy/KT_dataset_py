import os

print(os.listdir('../input/global-wheat-detection/'))
import pandas as pd

train_csv = pd.read_csv('../input/global-wheat-detection/train.csv')

train_csv.head()
train_csv.tail()
train_csv.shape
train_csv.isnull().any().any()
train_csv.info()
print(train_csv.width.unique())

print(train_csv.height.unique())
unique = train_csv['source'].unique()

print(unique)
train_csv.image_id.value_counts()
train_csv.groupby("source").image_id.count()
nunique = train_csv.image_id.nunique()

print(nunique)
train_csv.groupby("source").image_id.value_counts()
#Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (18, 8)

plt.rcParams['figure.figsize'] = (15, 10)

sns.countplot(train_csv['source'], palette = 'hsv')

plt.title('Distribution of Source', fontsize = 20)

plt.legend()

plt.show()
labels = ['ethz_1', 'arvalis_1', 'rres_1','arvalis_3','usask_1','arvalis_2','inrae_1 ']

plt.rcParams['figure.figsize'] = (7, 7)

plt.pie(train_csv['source'].value_counts(),labels=labels,explode = [0.0,0.0,0.05,0.05,0.2,0.2,0.2], autopct = '%.2f%%')

plt.title('Source', fontsize = 21)

plt.axis('off')

plt.legend(loc='lower center', bbox_to_anchor=(1, 1))

plt.show()
from ast import literal_eval



def get_bbox_area(bbox):

    bbox = literal_eval(bbox)

    return bbox[2] * bbox[3]

train_csv['bbox_area'] = train_csv['bbox'].apply(get_bbox_area)

train_csv['bbox_area'].value_counts().hist(bins=33)
train_dir = '../input/global-wheat-detection/train'

test_dir = '../input/global-wheat-detection/test'



print('total train images:', len(os.listdir(train_dir)))

print('total test images:', len(os.listdir(test_dir)))
import matplotlib.image as mpimg



pic_index = 100

train_files = os.listdir(train_dir)





next_train = [os.path.join(train_dir, fname) 

                for fname in train_files[pic_index-4:pic_index]]



for i, img_path in enumerate(next_train):

  img = mpimg.imread(img_path)

  plt.imshow(img)

  plt.axis('Off')

  plt.show()
from pandas_profiling import ProfileReport

profile = ProfileReport(train_csv, title='Report',progress_bar = False);

profile.to_widgets()
my_submission = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)