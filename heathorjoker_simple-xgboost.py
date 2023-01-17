import numpy as np 

import pandas as pd

import xgboost as xgb
df = pd.read_csv('../input/Admission_Predict.csv',index_col=0)
df.head(5)
def CoaToLabels(pointValue):

    percentage = pointValue * 100 #converting 0.6234 to 62.34

    percentage = int(percentage//10) #converting 62.34 to 6

    return percentage
df['admitLabels'] = df['Chance of Admit '].map(CoaToLabels)
df['admitLabels'].unique()
df['admitLabels'] = df['admitLabels'] - 3
df.drop('Chance of Admit ',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
y = df['admitLabels']

df.drop('admitLabels',axis=1,inplace=True)
train_X,test_X,train_y,test_y = train_test_split(df,y)
xgtrain = xgb.DMatrix(train_X,label=train_y)

xgtest = xgb.DMatrix(test_X) 
#setting the parameters

param = {

    'objective':'multi:softmax',

    'num_class': 7

}
clf = xgb.train(param,xgtrain)
predictionsXg = clf.predict(xgtest)

predictionsXg
predictionsXg = predictionsXg.astype(int)
from sklearn.metrics import accuracy

scoreXg = accuracy_score(test_y,predictionsXg)
print("We have the accuracy score "+str(scoreXg))