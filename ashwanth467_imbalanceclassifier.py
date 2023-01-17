# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data.columns
data.head()
data.count()
data.isnull().sum()
data.describe()
data['age'].corr(data['length_of_service'])
data[data['age']>39]['age'].corr(data[data['age']>39]['length_of_service'])
data.corr()

edct_missed = data[data['education'].isnull()]

pyrt_missed = data[data['previous_year_rating'].isnull()]

print(edct_missed.count())

print(pyrt_missed.count())
train_edu = data[-data['education'].isnull()]

train_pyr = data[-data['previous_year_rating'].isnull()]
train_edu = train_edu[train_edu.columns.difference(['previous_year_rating','employee_id'])]

edct_missed = edct_missed[train_edu.columns.difference(['previous_year_rating','employee_id','education'])]



column_names_for_onehot = ['department','region','gender','recruitment_channel']



train_x = pd.get_dummies(train_edu, columns=column_names_for_onehot, drop_first=True)

pred_x = pd.get_dummies(edct_missed, columns=column_names_for_onehot, drop_first=True)



train_y = train_x['education']

train_x = train_x[train_x.columns.difference(['education'])]

missing_columns = [x for x in train_x.columns if x not in pred_x.columns]

for x in missing_columns:

    pred_x[x]=0
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(train_x,train_y)



pred_y = clf.predict(pred_x)

train_edu = data[-data['education'].isnull()]

edu_missed = data[data['education'].isnull()]

edu_missed['education'] = pred_y

data = train_edu.append(edu_missed)
data.isnull().sum()
#Lets fill the previous_year_rating using a Multiple Regression

train_pyr = data[-data['previous_year_rating'].isnull()]

train_pyr = train_pyr[train_edu.columns.difference(['employee_id'])]

pyr_missed = data[data['previous_year_rating'].isnull()]

pyr_missed = pyr_missed[pyr_missed.columns.difference(['employee_id'])]



column_names_for_onehot = ['department','region','gender','recruitment_channel','education']



train_x = pd.get_dummies(train_pyr, columns=column_names_for_onehot, drop_first=True)

pred_x = pd.get_dummies(pyr_missed, columns=column_names_for_onehot, drop_first=True)

train_y = train_x['previous_year_rating']

train_x = train_x[train_x.columns.difference(['previous_year_rating'])]

pred_x = pred_x[pred_x.columns.difference(['previous_year_rating'])]
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()

regr.fit(train_x,train_y)

pred_y = regr.predict(pred_x)

train_pyr = data[-data['previous_year_rating'].isnull()]

pyr_missed = data[data['previous_year_rating'].isnull()]

pyr_missed['previous_year_rating'] = pred_y

data = train_pyr.append(pyr_missed)
data.drop('employee_id',inplace=True, axis=1)

print(data.columns)
column_names_for_onehot = ['department','region','gender','recruitment_channel','education']

data = pd.get_dummies(data,columns=column_names_for_onehot,drop_first=True)



train_x = data[data.columns.difference(['is_promoted'])]

train_y = data['is_promoted']



print(train_y.value_counts(normalize=True))
#split the data set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25, random_state=42)



#Build the model

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)

clf.fit(X_train, y_train)



#Predict using the above model

y_pred = clf.predict(X_test)



#Performace Metrics for the predictions 

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

print('Confusion Matrix:')

print(confusion_matrix(y_test,y_pred))

print('Accuracy:',accuracy_score(y_test,y_pred))

print('Recall:',recall_score(y_test,y_pred))
from imblearn.over_sampling import SVMSMOTE

cnn = SVMSMOTE(sampling_strategy='minority',random_state=42,n_jobs=8,m_neighbors=50)

X_res, y_res = cnn.fit_resample(train_x, train_y)



t_df=pd.DataFrame(data=y_res[0:],

                index=[i for i in range(y_res.shape[0])],

                columns=['is_promoted'])



df=pd.DataFrame(data=X_res[0:,0:],

                index=[i for i in range(X_res.shape[0])],

                columns=train_x.columns.tolist())



t_df['is_promoted'].value_counts()
X_train, X_test, y_train, y_test = train_test_split(df, y_res, test_size=0.25, random_state=42)



#Build Random Forest Classifier

clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Confusion Matrix:')

print(confusion_matrix(y_test,y_pred))

print('Accuracy:',accuracy_score(y_test,y_pred))

print('Recall:',recall_score(y_test,y_pred))



import xgboost as xgb



#label is used to define our outcome variable

dtrain=xgb.DMatrix(X_train,label=y_train)

dtest=xgb.DMatrix(X_test)

parameters={'max_depth':7, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}



#train XGBOOST

num_round=50

from datetime import datetime 

start = datetime.now() 

xg=xgb.train(parameters,dtrain,num_round) 

stop = datetime.now()

print("Execution Time:",stop-start )



#predict

y_pred=xg.predict(dtest) 





#Converting probabilities into 1 or 0  

for i in range(0,len(y_pred)): 

    if y_pred[i]>=.5:       # setting threshold to .5 

        y_pred[i]=1 

    else: 

        y_pred[i]=0  



print(y_pred)





#Calculate the performace of the model

print('Confusion Matrix:')

print(confusion_matrix(y_test,y_pred))

print('Accuracy:',accuracy_score(y_test,y_pred))

print('Recall:',recall_score(y_test,y_pred))