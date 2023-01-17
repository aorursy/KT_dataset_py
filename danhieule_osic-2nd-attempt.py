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
import pydicom
import seaborn as sns
import matplotlib.pyplot as plt
import px

from colorama import Fore, Back, Style

# Set Color Palettes for the notebook
custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']
sns.palplot(sns.color_palette(custom_colors))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
#Load data
train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
train.head()
train.shape
# How many unique values of rows (identifier is patient)
train.columns
# Missing values??
train.isnull().sum()
def unique_values_of_each_col(df):
    for col in df.columns:
        print(f"{col} has {len(df[col].unique())} unique values")
unique_values_of_each_col(train)
train.info()
path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"
numFiles = 0
numFolders = 0
files = []
for x, dirnames, filenames in os.walk(path):
    numFiles += len(filenames)
    numFolders += len(dirnames)
    files.append(len(filenames))

print("number of files: ", numFiles)
print("number of folders: ", numFolders)
print(f"there are {np.mean(files)} in average per patient")
print(f"there are {np.max(files)} max images for a patient")
f, axes = plt.subplots(2, 2, figsize = (7, 7), sharex = False, sharey=False)
sns.distplot(train["Weeks"], ax = axes[0, 0])
sns.distplot(train["FVC"], ax = axes[0, 1])
sns.distplot(train["Percent"], ax = axes[1, 0])
sns.distplot(train["Age"], ax = axes[1, 1])
sns.catplot(x="SmokingStatus", y="FVC", kind="bar", data=train)
sns.catplot(x="Sex", y="FVC", kind="bar", data=train)
# This is how to get values for a specific value in another column
train.loc[train['SmokingStatus'] == 'Ex-smoker', 'Age']
plt.figure(figsize=(16, 6))
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)

# Labeling of plot
plt.xlabel('Age (years)');
plt.ylabel('Density');
plt.title('Distribution of Ages over SmokingStatus');
plt.figure(figsize=(16, 6))
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Ex-smoker', 'FVC'], label = 'Ex-smoker',shade=True)
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Never smoked', 'FVC'], label = 'Never smoked',shade=True)
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Currently smokes', 'FVC'], label = 'Currently smokes', shade=True)

# Labeling of plot
plt.xlabel('FVC');
plt.ylabel('Density');
plt.title('Distribution of FVC over SmokingStatus');
import plotly.express as px
fig = px.scatter(train, x="Weeks", y="FVC", color="Age")
fig.show()
fig = px.scatter(train, x="Weeks", y="FVC", color="SmokingStatus")
fig.show()
fig = px.scatter(train, x="FVC", y="Percent", color='SmokingStatus')
fig.show()
#One patient to lineplot
patient = train[train.Patient == 'ID00228637202259965313869']
fig = px.line(patient, x="Weeks", y="FVC", color='SmokingStatus')
fig.show()
plt.hist(train["Sex"])
sns.countplot(data=train, x="SmokingStatus", hue="Sex")
first_patient_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/"
images = os.listdir(first_patient_path)
num = []
for image in images:
    num.append(int(image.split(".")[0]))
print(sorted(num))
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    filename = first_patient_path + "/" + str(num[i]) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap='jet')
plt.show()
def dump(obj):
    dic = dict()
    for attr in dir(obj):
        dic[attr] = getattr(obj, attr)
    return dic

sample_dcm = pydicom.dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/2.dcm")
sample_dic = dump(sample_dcm)
sample_dic
plt.imshow(sample_dcm.pixel_array, cmap ="jet")
