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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression 

from statistics import mean

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBClassifier
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.info()
data.status.unique()
data[data.status == 'Not Placed']['salary']
data.salary.fillna(0, inplace = True)  # filling missing values with 0 because 'nan' is for 'not placed candicates'
data.describe()
ax = sns.countplot('status',data=data)



total = float(len(data))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,height+.5,'{:1.0%}'.format((height/total)),ha="center")
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

data['gender'] = number.fit_transform(data['gender'])

data['ssc_b'] = number.fit_transform(data['ssc_b'])

data['hsc_b'] = number.fit_transform(data['hsc_b'])

data['hsc_s'] = number.fit_transform(data['hsc_s'])

data['degree_t'] = number.fit_transform(data['degree_t'])

data['specialisation'] = number.fit_transform(data['specialisation'])

data['workex'] = number.fit_transform(data['workex'])

data['status'] = number.fit_transform(data['status'])
data1=data.drop(columns=['salary','sl_no'])
plt.figure(figsize=(14,12))

sns.heatmap(data1.corr(), linewidth=0.2, annot=True)
fig,axes = plt.subplots(3,2, figsize=(20,12))

sns.lineplot(y = "ssc_p", x= 'sl_no', data = data, hue = "status",ax=axes[0][0])

sns.lineplot(y = "hsc_p", x= 'sl_no', data = data, hue = "status",ax=axes[0][1])

sns.lineplot(y = "degree_p", x= 'sl_no', data = data, hue = "status",ax=axes[1][0])

sns.lineplot(y = "mba_p", x= 'sl_no', data = data, hue = "status",ax=axes[1][1])

sns.lineplot(y = "etest_p", x= 'sl_no', data = data, hue = "status",ax=axes[2][0])

fig.delaxes(ax = axes[2][1]) 
ax = sns.countplot('specialisation',data=data)



total = float(len(data))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,height+.5,'{:1.0%}'.format((height/total)),ha="center")
X= data.drop(['status'],axis=1)

Y= data['status']
# Spliting the data using kFold split(n_splits= 3)

train_accuracy_log = []

test_accuracy_log = []

train_accuracy_rf = []

test_accuracy_rf = []

train_accuracy_xgb = []

test_accuracy_xgb = []



parameters = { 'max_features': [2, 4, 6, 8]}

param_grid = {'max_depth':np.arange(1,6),'learning_rate':[0.1,0.01,0.001]}



kf = KFold(n_splits=3)

for train_index, test_index in kf.split(X,Y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    

    model = LogisticRegression()

    model.fit(X_train,y_train)

    

    y_train_predict_log = model.predict(X_train)

    train_accuracy_log.append(accuracy_score(y_train,y_train_predict_log))

    

    y_test_predict_log = model.predict(X_test)

    test_accuracy_log.append(accuracy_score(y_test,y_test_predict_log))

    

    

    # RandomForestClassifier

    tune_model_rf = GridSearchCV(RandomForestClassifier(),parameters,cv=5,scoring='accuracy')

    tune_model_rf.fit(X_train,y_train)

        

    model_rf = RandomForestClassifier(n_estimators=150,max_depth = tune_model_rf.best_params_['max_features'],verbose=1,random_state=82)

    model_rf.fit(X_train,y_train)



    y_train_predict_rf = model_rf.predict(X_train)

    train_accuracy_rf.append(accuracy_score(y_train,y_train_predict_rf))

    

    y_test_predict_rf = model_rf.predict(X_test)

    test_accuracy_rf.append(accuracy_score(y_test,y_test_predict_rf))

    

    # XGBCClassifier

    tune_model = GridSearchCV(XGBClassifier(objective = 'binary:logistic'),param_grid,cv=5)

    tune_model.fit(X_train,y_train)

    

    model_xgb = XGBClassifier(objective = 'binary:logistic',random_state=82,learning_rate= tune_model.best_params_['learning_rate'], max_depth=tune_model.best_params_['max_depth'] )

    model_xgb.fit(X_train,y_train)

    

    y_train_predict_xgb = model_xgb.predict(X_train)

    train_accuracy_xgb.append(accuracy_score(y_train,y_train_predict_xgb))

    

    y_test_predict_xgb = model_xgb.predict(X_test)

    test_accuracy_xgb.append(accuracy_score(y_test,y_test_predict_xgb))

print('Logistic regression Train accuracy : ' , mean(train_accuracy_log)) 

print('Logistic regression Test accuracy : ' , mean(test_accuracy_log)) 



print('RandomForest classifier Train accuracy: ', mean(train_accuracy_rf)) 

print('RandomForest classifier Test accuracy: ', mean(test_accuracy_rf) )



print('XGboost classifier Train accuracy : ', mean(train_accuracy_xgb))   

print('XGboost classifier Test accuracy : ', mean(test_accuracy_xgb))