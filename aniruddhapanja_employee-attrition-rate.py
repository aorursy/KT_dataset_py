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
train_data=pd.read_csv("/kaggle/input/summeranalytics2020/train.csv")

train_data.head()
train_data.drop("Id",axis=1,inplace=True)

train_data.head()
x=train_data.drop(['Attrition','EmployeeNumber'],axis=1)

y=train_data['Attrition']
cater_col=[ col for col in x.columns if x[col].dtype=='object']
cater_col
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

ohe_x = pd.DataFrame(ohe.fit_transform(x[cater_col]))

ohe_x.head()
ohe_x.index = x.index

x_num = x.drop(cater_col, axis=1)

x_num.head()
x_train=pd.concat([x_num,ohe_x],axis=1)

x_train.head()
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model=ExtraTreesClassifier()

model.fit(x_train,y)
important_features=pd.Series(model.feature_importances_,index=x_train.columns)
top30=list(important_features.nlargest(30).index)

top30
x_top30=x_train[top30]

x_top30.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_top30,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=10,n_estimators=200, criterion='entropy', random_state=420)

model.fit(x_train, y_train)

predict_train = model.predict_proba(x_train)[:,1]

predict_test  = model.predict_proba(x_test)[:,1]



model.score(x_test, y_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,model.predict(x_test))
confusion_matrix(y_train,model.predict(x_train))
test_=pd.read_csv("/kaggle/input/summeranalytics2020/test.csv")

test_.head()
test = test_.drop('Id', axis=1)

test.head()
ohe_test = pd.DataFrame(ohe.fit_transform(test[cater_col]))

ohe_test.index = test.index

test_num = test.drop(cater_col, axis=1)

test_data = pd.concat([test_num, ohe_test], axis=1)



test_data_top30 = test_data[top30]
prediction = model.predict_proba(test_data_top30)[:,1]
output = pd.Series(prediction)

output_final = pd.concat([test_['Id'], output], axis=1)

output_final.columns=['Id', 'Attrition']

output_final.set_index('Id',inplace=True)
output_final.head()
output_final.to_csv("output_final.csv",index=False)