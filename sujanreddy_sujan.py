import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] #copying test index for later
df_train.head()




train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']





df_test = df_test.loc[:, 'V1':'V16']



train_X=pd.get_dummies(train_X,columns=['V2','V4','V9','V16','V3','V11'])

df_test=pd.get_dummies(df_test,columns=['V2','V4','V9','V16','V3','V11'])





df_test=df_test.drop(labels=['V11_7','V11_0','V11_11'],axis=1)

#train_X.shape

df_test.shape
import numpy as np



from sklearn.model_selection import train_test_split as tt

x_train,x_test,y_train,y_test=tt(train_X,train_y,test_size=0.2,random_state=0)

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier as x

co=0.1

"""for _ in range (25):

  model=x(n_estimators=500,learning_rate=co)

  model.fit(x_train,y_train)

  y_pred = model.predict(x_test)

  predictions = [round(value) for value in y_pred]

# evaluate predictions

  accuracy = accuracy_score(y_test, predictions)

  print("Accuracy: %.2f%%" % (accuracy * 100.0))

  print(co)

  co=0.01+co"""





"""from xgboost import XGBClassifier as x

co=1

for _ in range(10):

  classifier=x(n_estimators=500,learning_rate=0.16,max_depth=co,min_child_weight=1)

  classifier.fit(x_train,y_train)

  y_pred = classifier.predict(x_test)

  predictions = [round(value) for value in y_pred]

# evaluate predictions

  accuracy = accuracy_score(y_test, predictions)

  print("Accuracy: %.2f%%" % (accuracy * 100.0))

  print(co)

  co=co+1"""

model=x(n_estimators=500,learning_rate=0.16,max_depth=1,min_child_weight=1)

model.fit(train_X,train_y)











print(df_test.head())

pred = model.predict_proba(df_test)

result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)




