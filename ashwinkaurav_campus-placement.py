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
data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.info()
data.head()
salaries=data['salary']
status=data['status']
data.drop(['sl_no','status','salary'],axis=1,inplace=True)
categorical_cols=[cat for cat in data.columns if data[cat].dtype=='object']
categorical_cols
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for i in categorical_cols:
    data[i]=encoder.fit_transform(data[i])
data.head()
salaries.fillna(0)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,status,random_state=0,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier()
model1.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error,mean_squared_error
predictions=model1.predict(X_test)
predictions
actual_results=list(y_test)
wrong_preds=0
for i in range(len(predictions)):
    if predictions[i]!=actual_results[i]:
        wrong_preds+=1
print('%error = ',100*wrong_preds/len(predictions))        
from xgboost import XGBClassifier
model2=XGBClassifier(n_estimators=400,learning_rate=0.001)
model2.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_test, y_test)],verbose=False)
#encoded_y_train=encoder.fit_transform(y_train)
#print(y_train[0],encoded_y_train[0])
predictions2=model2.predict(X_test)
wrong_preds=0
for i in range(len(predictions2)):
    if predictions2[i]!=actual_results[i]:
        wrong_preds+=1
print(wrong_preds*100/len(predictions2),'%  error')    

from keras.models import Sequential
from keras.layers import Dense
dummy_y_test=pd.get_dummies(y_test)
dummy_y_train=pd.get_dummies(y_train)

dummy_y_train.drop('Not Placed',axis=1,inplace=True)
dummy_y_test.drop('Not Placed',axis=1,inplace=True)
ann_model=Sequential()
ann_model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))
ann_model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
ann_model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann_model.fit(X_train,dummy_y_train,batch_size=10,epochs=100)

ann_predictions=(ann_model.predict(X_test)).tolist()
for i in range(len(ann_predictions)):
    ann_predictions[i]=ann_predictions[i][0]
for i in range(len(ann_predictions)):
    if ann_predictions[i]>0.5:
        ann_predictions[i]=1
    else:
        ann_predictions[i]=0
wrong_preds=0
actual_y_test=list(dummy_y_test["Placed"])
for i in range(len(ann_predictions)):
    if ann_predictions[i]!=actual_y_test[i]:
        wrong_preds+=1
print(wrong_preds*100/len(predictions2),'%  error')       
list(dummy_y_test["Placed"])

