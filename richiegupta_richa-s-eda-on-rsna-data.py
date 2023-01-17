# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns 

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline 

np.warnings.filterwarnings('ignore')
!ls ../input/rsna-pneumonia-detection-challenge/
df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

print(df.iloc[0])
print(df.iloc[10])

import pydicom

patientId = df['patientId'][10]

dcm_file = '../input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % patientId

dcm_data = pydicom.read_file(dcm_file)



print(dcm_data)
data_class = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

data_tlables = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')



print(data_class.iloc[20])

print(data_tlables.iloc[20])
def get_feature_distribution(data, feature):

    # Get the count for each label

    label_counts = data[feature].value_counts()



    # Get total number of samples

    total_samples = len(data)



    # Count the number of items in each class

    print("Feature: {}".format(feature))

    for i in range(len(label_counts)):

        label = label_counts.index[i]

        count = label_counts.values[i]

        percent = int((count / total_samples) * 10000) / 100

        print("{:<30s}:   {} or {}%".format(label, count, percent))



get_feature_distribution(data_class, 'class')
train_class_dataf = data_tlables.merge(data_class, left_on='patientId', right_on='patientId', how='inner')

train_class_dataf.sample(10)

train_class_dataf.describe()
get_feature_distribution(train_class_dataf, 'class')
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))

missing_data(train_class_dataf)
sns.countplot(data_class['class'],order = data_class['class'].value_counts().index, palette='Set3')
