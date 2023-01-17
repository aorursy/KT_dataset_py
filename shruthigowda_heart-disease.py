# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

heart = pd.read_csv("../input/heart.csv")
import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
tr=pd.read_csv('../input/heart.csv')
## first 5 rows of data

tr.head()
## Number of rows and columns of data

tr.shape
## column names

tr.columns
## Number of distinct observations

tr.nunique()
## find missing values in each column

tr.isna().sum()
## Getting the statistical summary of the different columns

tr.describe()
print(tr['target'].value_counts())

tr[tr['target']==1].shape

tr['target'].value_counts().plot.bar()
print(tr['sex'].value_counts())
tr[(tr['target']==1) & (tr['sex']==1)].shape
tr[(tr['target']==1) & (tr['sex']==0)].shape
tr.hist(column='thalach')
tr[tr['target']==1].sort_values(by='thalach',ascending=False).head()
tr.corr()
#data=tr
#df=pd.DataFrame(data)
#df['age']=df['age'].astype('category').cat.codes

#df['sex']=df['sex'].astype('category').cat.codes

#df['cp']=df['cp'].astype('category').cat.codes

#df['trestbps']=df['trestbps'].astype('category').cat.codes

#df['chol']=df['chol'].astype('category').cat.codes

#df['fbs']=df['fbs'].astype('category').cat.codes

#df['restecg']=df['restecg'].astype('category').cat.codes

#df['thalach']=df['thalach'].astype('category').cat.codes

#df['exang']=df['exang'].astype('category').cat.codes

#df['oldpeak']=df['oldpeak'].astype('category').cat.codes

#df['slope']=df['slope'].astype('category').cat.codes

#df['ca']=df['ca'].astype('category').cat.codes

#df['thal']=df['thal'].astype('category').cat.codes

#df['target']=df['target'].astype('category').cat.codes
df[df.columns[0:]].corr()['target'][:]
tr.describe()
tr.hist(column='age')
tr[((tr['age']) & (tr['target']==1))]
tr['cp'].value_counts().plot.pie()
tr[tr['target']==0].sort_values(by='thalach',ascending=False).head(1)