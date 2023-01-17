import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")
train.info()

train[0:10]
train = train[['County','Province_State','Country_Region','Date','Population','Weight','Target','TargetValue']]

train.head()
#Country_Region top 50

train.Country_Region.value_counts()[0:50].plot(kind='bar')

plt.show()
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train.info()
train['Date'] = train['Date'].astype('int64')

test['Date'] = test['Date'].astype('int64')
test.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def FunLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df
train = FunLabelEncoder(train)

train.info()

train.iloc[235:300,:]
test = FunLabelEncoder(test)

test.info()

test.iloc[235:300,:]
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["TargetValue"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
#Select feature column names and target variable we are going to use for training

features=['County','Province_State','Country_Region','Date','Population','Weight','Target']

target = 'TargetValue'
#This is input which our classifier will use as an input.

train[features].head(10)
#Display first 10 target variables

train[target].head(10).values
from sklearn.tree import DecisionTreeClassifier



# We define the model

dtcla = DecisionTreeClassifier(random_state=None)





# We train model

dtcla.fit(train[features],train[target])
#Make predictions using the features from the test data set

predictions = dtcla.predict(test[features])



predictions
pred_list = [int(x) for x in predictions]



result = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})

print(result)
a=result.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=result.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=result.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05'].clip(0,10000)

a['q0.5']=a['q0.5'].clip(0,10000)

a['q0.95']=a['q0.95'].clip(0,10000)

a
a['Id'] =a['Id']+ 1

a
submission=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

submission['variable']=submission['variable'].str.replace("q","", regex=False)

submission['ForecastId_Quantile']=submission['Id'].astype(str)+'_'+submission['variable']

submission['TargetValue']=submission['value']

submission=submission[['ForecastId_Quantile','TargetValue']]

submission.reset_index(drop=True,inplace=True)

submission.to_csv("submission.csv",index=False)

submission.head(50)