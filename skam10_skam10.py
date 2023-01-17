#Importing required packages



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0'] 
df_train.head()
X = df_train.loc[:,'V1':'V16']

X1=df_train.loc[:, ['V1','V2','V3','V4','V6','V10','V12','V13','V11','V7','V8']]

y = df_train.loc[:,'Class']
rf = RandomForestClassifier(n_estimators=76, random_state=123,max_features=3,min_samples_leaf=2,min_samples_split=2,max_depth=8)
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 

scaler.fit(df_train) 
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X1,y,random_state=101,test_size=0.3)

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(11).plot(kind='barh')

plt.show()
error_rate=[]

for i in range(1,10,1):

    ne=RandomForestClassifier(max_features=i)

    ne.fit(train_X,train_y)

    pred_i=ne.predict(test_X)

    

    error_rate.append(np.mean(pred_i!=test_y))
plt.figure(figsize=(10,10))

plt.style.use('ggplot')



plt.plot(range(1,10),error_rate,marker='o',markerfacecolor='red')

plt.xlabel('N ESTIMATORS')

plt.ylabel('ERROR RATE')
df_train.plot.scatter(x='V12',y='V6')
df_test.plot.scatter(x='V12',y='V6')
df_train.plot.scatter(x='V1',y='V6')
df_train.plot.scatter(x='V13',y='V1')

df_train.drop(df_train[df_train['V12'] >3000].index, inplace = True) 

df_train.drop(df_train[df_train['V6'] >50000].index, inplace = True) 

df_train.drop(df_train[df_train['V1'] >63].index, inplace = True) 

df_train.drop(df_train[df_train['V13'] >50].index, inplace = True) 



rf.fit(train_X, train_y)

df_test = df_test.loc[:, ['V1','V2','V3','V4','V6','V10','V12','V13','V11','V7','V8']]

pred = rf.predict_proba(df_test)

rf.score(test_X.loc[:, ['V1','V2','V3','V4','V6','V10','V12','V13','V11','V7','V8']],test_y)

result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)