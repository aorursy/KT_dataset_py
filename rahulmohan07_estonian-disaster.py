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
df = pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

df.head()
df.describe
df.dtypes
df.isnull().sum()
temp = ['Country','Firstname','Lastname','Sex','Category']

for i in temp:

    print('************ Value Count in', i, '************')

    print(df[i].value_counts())

    print('')
df.shape
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.countplot(x=df['Survived'],hue='Category',data=df)
import plotly.graph_objects as go



labels = df.Country.value_counts().index

values = df.Country.value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
pd.crosstab(df['Sex'],df['Category'])
df.groupby("Sex")['Age'].mean().plot.bar()
males_survived= df['Sex'][(df['Sex']=='M') & (df['Survived']==1)].count()

        

total_males =  df['Sex'][df['Sex']=='M'].count()

print("Percentage of Males who survived = ",(males_survived/total_males)*100)
females_survived= df['Sex'][(df['Sex']=='F') & (df['Survived']==1)].count()

        

total_females =  df['Sex'][df['Sex']=='F'].count()

print("Percentage of Females who survived = ",(females_survived/total_females)*100)
df = pd.get_dummies(df)

x = df.drop(['Survived'],axis=1)

y = df['Survived']
from sklearn.model_selection import train_test_split as tts 

train_x,test_x,train_y,test_y = tts(x,y,test_size=0.1,random_state=1,stratify=y)
from sklearn.linear_model import LogisticRegression as LogReg

from sklearn.metrics import roc_auc_score as ras
logreg = LogReg()

logreg.fit(train_x,train_y)
train_predict = logreg.predict_proba(train_x)

train_predict
train_preds = train_predict[:,1]

train_preds

for i in range(0,len(train_preds)):

    if(train_preds[i]>0.55):

        train_preds[i] = 1

    else:

        train_preds[i] = 0



k = ras(train_preds,train_y)

print("Training score =",k)
test_predict = logreg.predict_proba(test_x)

test_predict
test_preds = test_predict[:,1]

test_preds
for i in range(0,len(test_preds)):

    if(test_preds[i]>0.55):

        test_preds[i] = 1

    else:

        test_preds[i] = 0



sp = ras(test_preds,test_y)

print("Test score = ",sp)
my_submission = pd.DataFrame({'Id': test.Id, 'SurvivalPrediction': sp})

pd.to_csv('submission.csv', index=False)