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

import seaborn as sns




df_train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')



df_train.head()
df_train.info()
df_train.shape
df_train.head()
df_train['Hospital_type_code'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Hospital_type_code", data=df_train)

plt.show()

df_train['City_Code_Hospital'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="City_Code_Hospital", data=df_train)

plt.show()
df_train['Hospital_region_code'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Hospital_region_code", data=df_train)

plt.show()
df_train['Department'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Department", data=df_train)

plt.show()
df_train['Ward_Type'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Ward_Type", data=df_train)

plt.show()
df_train['Ward_Facility_Code'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Ward_Facility_Code", data=df_train)

plt.show()
df_train['Bed Grade'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Bed Grade", data=df_train)

plt.show()
df_train['Type of Admission'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Type of Admission", data=df_train)

plt.show()
df_train['Severity of Illness'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Severity of Illness", data=df_train)

plt.show()
df_train['Age'].astype('category').value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Age", data=df_train)

plt.show()
sns.distplot(df_train['Admission_Deposit'])

plt.show()
df_train.groupby('Age')['Severity of Illness'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Age", hue="Severity of Illness", data=df_train)

plt.show()
df_train.groupby('Age')['Type of Admission'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Age", hue="Type of Admission", data=df_train)

plt.show()
df_train.groupby('Age')['Department'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Age", hue="Department", data=df_train)

plt.show()
df_train.groupby('Age')['Stay'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Age", hue="Stay", data=df_train)

plt.show()
df_train.groupby('Type of Admission')['Stay'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Type of Admission", hue="Stay", data=df_train)

plt.show()
df_train.groupby('Severity of Illness')['Stay'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Severity of Illness", hue="Stay", data=df_train)

plt.show()
df_train.groupby('Department')['Stay'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x="Department", hue="Stay", data=df_train)

plt.show()