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
df1=pd.read_csv('/kaggle/input/student-grade-prediction/student-mat.csv')

df1.head(20)
df1.isnull().sum()
df1.describe()
df1.corr()
import seaborn as sns
sns.heatmap(df1.corr())
# shows number of student of different age prefer to go-out

import matplotlib as plt

df1.groupby('age')['goout'].count().plot.bar()
# shows number of failure student in respective region

df1.groupby('address')['failures'].count().plot.bar()
df1.columns
(df1[df1['age']==16]['failures']).value_counts().plot.bar()
(df1[df1['age']==19]['failures']).value_counts().plot.bar()
dmap={'yes':1,'no':0}

df1['paid']=df1['paid'].map(dmap)

df1['schoolsup']=df1['schoolsup'].map(dmap)

df1['famsup']=df1['famsup'].map(dmap)

df1['romantic']=df1['romantic'].map(dmap)

df1['internet']=df1['internet'].map(dmap)
x=df1.drop(['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','paid','activities','nursery','higher','internet','famrel','Dalc','Walc','health','traveltime','schoolsup','failures','G1','G3','freetime'],axis=1)

y=df1['internet']



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
cl=LogisticRegression()

cl.fit(x_train,y_train)
y_pred=cl.predict(x_test)

cl.score(x_test,y_test)
from sklearn import metrics

cm=metrics.confusion_matrix(y_test,y_pred)

cm
from sklearn.feature_selection import chi2
chi_scores=chi2(x,y)
chi_scores
p_values=pd.Series(chi_scores[1],index=x.columns)

p_values.plot.bar()