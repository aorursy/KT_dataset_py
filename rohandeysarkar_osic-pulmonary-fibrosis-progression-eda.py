import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="ticks", color_codes=True)

import pydicom
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
folder_path = "../input/osic-pulmonary-fibrosis-progression/"



train_df = pd.read_csv(folder_path + '/train.csv')

test_df = pd.read_csv(folder_path + '/test.csv')
train_df.head()
train_df.info()
train_df.isnull().sum()
# train_df['Patient'].value_counts().shape[0]

print(f"No. of patients are {train_df['Patient'].count()} with {train_df['Patient'].value_counts().shape[0]} unique patient ids")
train_patient_id = set(train_df['Patient'].unique())

test_patient_id = set(test_df['Patient'].unique())



train_patient_id.intersection(test_patient_id)
columns = list(train_df.columns)

print(f'The colums are {columns}')
train_df['Patient'].value_counts().max()
folders = []

files = []



path = "../input/osic-pulmonary-fibrosis-progression/train"



for _, dirnames, filenames in os.walk(path):

    files.append(len(filenames))

    folders.append(len(dirnames))



print(f"No. of patients/folders:  {sum(folders)}")

print(f"No. of images/files:  {sum(files)}")

print(f"AVG images/files of patient:  {round(np.mean(files))}")

print(f"MAX images/files of a patient:  {round(np.max(files))}")
patient_df = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()



patient_df.head()
train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'



available_images = []



for patient_id in patient_df['Patient']:

    available_images.append(len(os.listdir(train_dir + patient_id)))

    

patient_df["available_images"] = available_images



patient_df.head()
patient_df["SmokingStatus"].value_counts()
plt.figure(figsize=(8, 6))

sns.countplot(x='SmokingStatus', data=patient_df)

plt.title('Smoking Status')

plt.show()
train_df['Weeks'].value_counts()
n_bins = int(np.sqrt(len(train_df["Weeks"])))



plt.figure(figsize=(12, 8))

sns.distplot(train_df["Weeks"],bins=n_bins, kde=False)

plt.title('Week Distribution')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(train_df["Age"], hist=False, kde_kws=dict(lw=6, ls="--"))

# sns.countplot(x="Age", data=train_df, order=train_df['Age'].value_counts().index)

plt.title("Age count")

plt.show()
plt.figure(figsize=(8, 6))

sns.catplot(x='Sex', kind='count', data=train_df)

plt.title('Sex Count')
train_df['FVC'].value_counts()
plt.figure(figsize=(12, 6))

sns.boxplot(train_df['FVC'])

plt.show()
n_bins = int(np.sqrt(len(train_df["FVC"])))



plt.figure(figsize=(12, 6))

sns.distplot(train_df["FVC"], bins=n_bins, kde=False)

plt.title('Distribution of the FVC')

plt.show()
n_bins = int(np.sqrt(len(train_df["Percent"])))



plt.figure(figsize=(12, 6))

sns.distplot(train_df["Percent"], bins=n_bins, kde=False)

plt.title('Distribution of the Percent')

plt.show()
plt.figure(figsize=(12, 6))

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

plt.xlabel('Age (years)')

plt.ylabel('Density')

plt.title('Age vs Sex')
temp_df = patient_df.groupby(['Sex', 'SmokingStatus'])['Sex'].count().unstack(['Sex'])

temp_df.plot.bar(rot=0, figsize=(12, 6))

plt.title('Sex vs SmokingStatus')

plt.show()
plt.figure(figsize=(12, 6))

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)

plt.xlabel('Age (years)')

plt.ylabel('Density')

plt.title('Age vs SmokingStatus');
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df["FVC"], y=train_df["Percent"],size=10)

plt.title("Joint plot FVC vs Percent")

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df["FVC"], y=train_df["Percent"], kind='kde',size=10)

plt.title("Joint plot FVC vs Percent")

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df["FVC"], y=train_df["Age"],size=10)

plt.title("Joint plot FVC vs Percent")

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df["FVC"], y=train_df["Age"], kind='kde',size=10)

plt.title("Joint plot FVC vs Percent")

plt.show()
corr = train_df.corr()

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corr, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5) 
# Normalization

train_df['Age'] = (train_df['Age'] - train_df['Age'].min() ) / ( train_df['Age'].max() - train_df['Age'].min() )

train_df['FVC'] = (train_df['FVC'] - train_df['FVC'].min() ) / ( train_df['FVC'].max() - train_df['FVC'].min() )

train_df['Weeks'] = (train_df['Weeks'] - train_df['Weeks'].min() ) / ( train_df['Weeks'].max() - train_df['Weeks'].min() )

train_df['Percent'] = (train_df['Percent'] - train_df['Percent'].min() ) / ( train_df['Percent'].max() - train_df['Percent'].min() )
imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"

print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))



fig=plt.figure(figsize=(12, 12))

columns = 4

rows = 5

imglist = os.listdir(imdir)

for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='gray')

plt.show()