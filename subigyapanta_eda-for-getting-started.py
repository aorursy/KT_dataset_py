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
import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt

import seaborn as sb
input_dir = '/kaggle/input'

isic_dir = os.path.join(input_dir, 'siim-isic-melanoma-classification')

jpg_dir = os.path.join(isic_dir, 'jpeg')

work_dir = os.path.abspath(os.getcwd())
train_csv = pd.read_csv(os.path.join(isic_dir, 'train.csv'))

test_csv = pd.read_csv(os.path.join(isic_dir, 'test.csv'))



train_csv.head()
test_csv.head()
def preprocess(data):

    data['sex'] = data['sex'].astype('category')

    data['site'] = data['anatom_site_general_challenge'].astype('category')

    data['diagnosis'] = data['diagnosis'].astype('category')

    data['benign_malignant'] = data['benign_malignant'].astype('category')

    

    return data
train_csv = preprocess(train_csv)
train_csv.info()
total = train_csv.shape[0]

print('Missing sex info: ', total - train_csv[train_csv.sex.notnull()].shape[0])

print('Missing age info: ', total - train_csv[train_csv.age_approx.notnull()].shape[0])

print('Missing site info: ', total - train_csv[train_csv.site.notnull()].shape[0])
train_csv['has_missing'] = train_csv.apply(lambda row: row.isna().any(), axis=1)
train_csv.head()
sb.distplot(train_csv[train_csv.has_missing==True]['target'], label='Missing', kde=False)

sb.distplot(train_csv[train_csv.has_missing==False]['target'], label='Full', kde=False)

plt.legend(loc='best')

plt.show()
missing_malignant = train_csv[(train_csv.has_missing==True) & (train_csv.target==1)].shape[0]

missing_benign = train_csv[(train_csv.has_missing==False) & (train_csv.target==1)].shape[0]



print('Malignant with missing values: ', missing_malignant)

print('Benign with missing values: ', missing_benign)
unique_patients = train_csv['patient_id'].unique().shape[0] 

unique_images = train_csv['image_name'].unique().shape[0]

total = train_csv['patient_id'].shape[0]



print('Unique patients: ', unique_patients)

print('Unique images: ', unique_images)

print('Total: ', total)
# for train and test

common_images = test_csv['image_name'].isin(train_csv['image_name']).value_counts()

common_patients = test_csv['patient_id'].isin(train_csv['patient_id']).value_counts()

print('Common images ---> ', common_images)

print('Common Patients ---> ', common_patients)

print('Total test: ', test_csv.shape[0])
sb.distplot(train_csv["patient_id"].value_counts())

plt.title("Patient ID distribution")
# For numeric columns

plt.rcParams['figure.figsize'] = 10,4

train_csv['age_approx'].hist()

plt.show()
# For non-numeric columns

plt.rcParams['figure.figsize'] = 16,12

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

train_csv['sex'].value_counts().plot(kind='bar', ax=ax1, title='Sex')

train_csv['anatom_site_general_challenge'].value_counts().plot(kind='bar', ax=ax2, title='Site')

train_csv['diagnosis'].value_counts().plot(kind='bar', ax=ax3, title='Diagnosis')

train_csv['benign_malignant'].value_counts().plot(kind='bar', ax=ax4, title='Benign/Malignant')

plt.tight_layout() 

plt.show()
# plt.rcParams['figure.figsize'] = 6,4



plt.rcParams['figure.figsize'] = 16,6

fig, (ax1, ax2) = plt.subplots(1,2)

sb.countplot(x='sex', hue='benign_malignant', data=train_csv, ax=ax1)

sb.countplot(x='anatom_site_general_challenge', hue='benign_malignant', data=train_csv, ax=ax2)

plt.tight_layout()

plt.show()
sex_site_result = train_csv.groupby(['sex', 'site', 'benign_malignant']).agg(occurence=('target','count'))

sex_site_result = sex_site_result.reset_index()

sex_site_result
sex_site_result.info()
sb.scatterplot(x='site', y='occurence', hue='benign_malignant', style='sex', data=sex_site_result)

plt.show()
sb.scatterplot(x='age_approx', y='site', hue='benign_malignant', style='sex', alpha=0.6, markers=['P', 'X'], data=train_csv)

plt.show()
# sb.pairplot(train_csv[['age_approx','benign_malignant']], hue="benign_malignant")

sb.distplot(train_csv[train_csv.target==0]['age_approx'], label='Benign')

sb.distplot(train_csv[train_csv.target==1]['age_approx'], label='Malignant')

plt.legend(loc='best')

plt.show()
sb.distplot(train_csv[train_csv.sex=='male']['age_approx'], label='Male')

sb.distplot(train_csv[train_csv.sex=='female']['age_approx'], label='Female')

plt.legend(loc='best')

plt.show()
sb.distplot(train_csv[(train_csv.sex=='male') & (train_csv.target==0)]['age_approx'], label='Male Benign')

sb.distplot(train_csv[(train_csv.sex=='male') & (train_csv.target==1)]['age_approx'], label='Male Malignant')

sb.distplot(train_csv[(train_csv.sex=='female') & (train_csv.target==0)]['age_approx'], label='Female Benign')

sb.distplot(train_csv[(train_csv.sex=='female') & (train_csv.target==1)]['age_approx'], label='Female Malignant')

plt.legend(loc='best')

plt.show()
benign = train_csv[train_csv.benign_malignant=='benign']

malignant = train_csv[train_csv.benign_malignant=='malignant']
bimg = plt.imread(jpg_dir + '/train/' + benign.iloc[0]['image_name'] + '.jpg')

plt.imshow(bimg)
mimg = plt.imread(jpg_dir + '/train/' + malignant.iloc[0]['image_name'] + '.jpg')

plt.imshow(mimg)
def show_rgb_histogram(image, ax=None):

    if ax is None:

        plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.5, label='Red')

        plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5, label='Green')

        plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5, label='Blue')

        plt.xlabel('Intensity')

        plt.ylabel('Count')

        plt.legend(loc='best')

        plt.show()

    else:

        ax.hist(image[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.5, label='Red')

        ax.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5, label='Green')

        ax.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5, label='Blue')

        ax.set_xlabel('Intensity')

        ax.set_ylabel('Count')

        ax.legend(loc='best')
show_rgb_histogram(bimg)
show_rgb_histogram(mimg)
random_benign = benign.sample(6)

random_malignant = malignant.sample(6)
plt.rcParams['figure.figsize'] = 10,6

for index, row in random_benign.iterrows():

    img = plt.imread(jpg_dir + '/train/' + row['image_name'] + '.jpg')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img)

    show_rgb_histogram(img, ax2)

    plt.tight_layout()

    plt.show()
for index, row in random_malignant.iterrows():

    img = plt.imread(jpg_dir + '/train/' + row['image_name'] + '.jpg')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img)

    show_rgb_histogram(img, ax2)

    plt.tight_layout()

    plt.show()