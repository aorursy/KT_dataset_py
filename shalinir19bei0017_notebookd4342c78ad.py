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
#uploading test and train data set using pandas

import pandas as pd



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



#To find the no.of rows and coloumns in the test and train data set

test.shape

train.shape



#To see if there is any missing data set in the rows

train.info()

test.info()



#import action 2

import matplotlib.pyplot as plt

import seaborn as sns



#combining train and test data set



traintestdata= [train,test]

for dataset in traintestdata:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')

    

train['Title'].value_counts()

test['Title'].value_counts()



#Mapping each feature to a numerical:



Mappingtitle = {"master":3, "Dr":3, "Rev": 3, "col":3, "Major":3, "Mlle":3, "Countess":3, "Ms":3, "lady":3,"Jonkheer":3,"Don":23, "Dona":3, "Mme":3, "Capt":3, "Sir":3,"Mr":0, "Miss":1, "Mrs":2}

for dataset in traintestdata:

    dataset['Title']=dataset['Title'].map(Mappingtitle)

train.head()

test.head()

#mapping the sex

Mappingsex={"male":0,"female":1}

for dataset in traintestdata:

    dataset['Sex']=dataset['Sex'].map(Mappingsex)

train.head()

test.head()
#Pclass

#for filling the missing values on Pclass

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['class 1','class 2', 'class 3']

df.plot(kind='bar',stacked=True, figsize=(10,5))

for dataset in traintestdata:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head()