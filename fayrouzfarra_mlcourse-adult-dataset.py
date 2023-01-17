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
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")

my_filepath = ('../input/mlcourse/adult_train.csv')

df=pd.read_csv(my_filepath)

df
df.shape
df.describe()
df.isnull().sum()
df['Workclass'].value_counts()
for col in df.columns:

    print(col, len(df[col].unique()))
for col in df.columns:

    print(col, df[col].value_counts())

    print('....................')
for col in ['Workclass','Occupation', 'Country']:

    df.fillna(df[col].value_counts().index[0], inplace=True)     #df.fillna(df[col].mode()[0], inplace=True)
df.isna().sum()
df.head()
df.Target.value_counts()
plt.figure(figsize=(10,6))

sns.countplot(x= 'Target',hue='Sex',data = df)
sns.pairplot(data=df)
#plt.figure(figsize = (10,6))

good_job = df.sort_values(by='Hours_per_week', ascending=False)[:10][['Age','Martial_Status','Hours_per_week']]

good_job
df.Sex.value_counts()
sns.set(style='dark')

plt.figure(figsize=(20,16))

sns.barplot(x=df['Martial_Status'], y=df['Age'])

plt.title('Relationship between Age and Martial Stauts')