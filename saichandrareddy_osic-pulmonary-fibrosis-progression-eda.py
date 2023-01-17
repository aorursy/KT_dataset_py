# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
data.head()
#Setting the Patients column as the Index



data = data.set_index(['Patient'])
data.info()
data.corr()
data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

data['SmokingStatus'] = le.fit_transform(data['SmokingStatus'])
import seaborn as sns

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(15, 10))

sns.heatmap(data.corr(), annot=True, 

           cbar_kws={"orientation": "horizontal"})

plt.show()
val = data.isna().sum().values

lab = data.columns



plt.bar(lab, val)

plt.xlabel('Labels')

plt.ylabel("Empty or not")

plt.title("Null values")

plt.tight_layout()

plt.show()
vals = data['Age'].value_counts().values

labs = data['Age'].unique()



figu = plt.figure(figsize=(15, 5))

fig = sns.barplot(labs,vals) 

plt.xlabel('Ages')

plt.ylabel("Total Pateints")

plt.title("Total number of patients with the particular age")

plt.tight_layout()

plt.show()


vals = data['Sex'].value_counts().values

labs = ['male', 'Female']

sns.barplot(vals, labs)

plt.xlabel('Labels')

plt.ylabel("Number of males or females")

plt.title("Sex Count")

plt.tight_layout()

plt.show()
vals = data['Sex'].value_counts().values

labs = ['male', 'female']



explode = []



for i in range(len(vals)):

    if max(vals) == vals[i]:

        explode.append(0.1)

    else:

        explode.append(0)

        



plt.pie(vals, explode, labs, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.title("Sex Count")

plt.tight_layout()

plt.show()
vals = data['SmokingStatus'].unique()

vals
vals = data['SmokingStatus'].value_counts().values

labs = ['Ex-smoker', 'Never smoked', 'Currently smokes']



fig = plt.figure(figsize=(7, 7))



explode = []



for i in range(len(vals)):

    if max(vals) == vals[i]:

        explode.append(0.1)

    else:

        explode.append(0)

        



plt.pie(vals, explode, labs, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.title("Sex Count")

plt.tight_layout()

plt.show()
vals = data['SmokingStatus'].value_counts().values

labs = ['Ex-smoker', 'Never smoked', 'Currently smokes']





fig = plt.figure(figsize=(12, 5))

sns.barplot(vals, labs)



plt.xlabel('Labels')

plt.ylabel("Number of males or females")

plt.title("Sex Count")

plt.tight_layout()

plt.show()
path = '../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430'



images = [path + '/' + img for img in os.listdir(path) if img.endswith('dcm')]
import pydicom

w=10

h=10

fig=plt.figure(figsize=(14, 8))

columns = 3

rows = 3

for i in range(1, columns*rows+1):

    ds = pydicom.dcmread(images[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='hsv') 

plt.show()
fvc_female = []

fvc_male = []



for i in range(len(data)):

    

    if data['Sex'][i] == 1 :

        fvc_female.append(data['FVC'][i])

    elif data['Sex'][i] == 0:

        fvc_male.append(data['FVC'][i])

    else:

        pass
plt.scatter(fvc_female, list(range(len(fvc_female))), c='r', label='Male')

plt.scatter(fvc_male,list(range(len(fvc_male))), c='y', label='Female')

plt.xlabel("FVC")

plt.ylabel("Range")

plt.title("Forced vital capacity (FVC)")

plt.legend()

plt.show()
print(f"The Max week of the patient {max(data['Weeks'])}")

print(f"The Max week of the patient {min(data['Weeks'])}")
data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
data.head()
data.shape
#Setting the Patients column as the Index



data = data.set_index(['Patient'])
path = '../input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264'



images = [path + '/' + img for img in os.listdir(path) if img.endswith('dcm')]
ds = pydicom.dcmread(images[0])

val = ds.pixel_array

img = np.array(val, dtype='f')

img
import pydicom

import cv2

w=10

h=10

fig=plt.figure(figsize=(14, 8))

columns = 3

rows = 3

for i in range(1, columns*rows+1):

    ds = pydicom.dcmread(images[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='hsv') 

plt.show()
print(f"The Max week of the patient {max(data['Weeks'])}")

print(f"The Max week of the patient {min(data['Weeks'])}")
train_Data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
vals = train_Data['Patient'].value_counts()[:25]

labs = train_Data['Patient'].unique()[:25]



fig = plt.figure(figsize=(12, 5))

sns.barplot(vals, labs)



plt.xlabel('Count')

plt.ylabel("Patients IDS")

plt.title("Unqiue patient Count out of duplicate")

plt.tight_layout()

plt.show()

nodupData = train_Data.drop_duplicates(subset = 'Patient', keep='first')
nodupData.set_index('Patient', inplace=True)
nodupData


Unique_patients = list(train_Data['Patient'].unique())





def getting_group(groupID):

    

    return train_Data.groupby('Patient').get_group(groupID)



print(getting_group(Unique_patients[0]).plot())

print(getting_group(Unique_patients[1]).plot())
test_Data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
Unique_patients == list(test_Data['Patient'].unique())
test_Data
samplt_Data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
#samplt_Data['Patient_Week'].unique()