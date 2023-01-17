import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head()
train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
train_data
train_data.isnull().sum()
train_data['Embarked'] = train_data['Embarked'].fillna(method ='pad')
train_data.info()
from sklearn.preprocessing import LabelEncoder
def Change_obj_type(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data
train_data = Change_obj_type(train_data)
train_data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

imput_train = imputer.fit(train_data)
train_data = imput_train.transform(train_data)
train_data
X = train_data[:,1:]
Y = train_data[:,0]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, tol=0.0000001)
lr.fit(X,Y)
lr.classes_
lr.coef_
lr.intercept_
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0, n_estimators=300, max_depth=3)
RFC.fit(X,Y)
params = {}
params['learning_rate'] = 0.04
params['max_depth'] = 7
params['n_estimators'] = 1000
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
#params['class_weight']: {0: 0.44, 1: 0.4}
import lightgbm
from lightgbm import LGBMClassifier
model = LGBMClassifier(**params)
model.fit(X,Y)
t_data = pd.read_csv("../input/titanic/test.csv")
t_data
test_data = t_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_data['Embarked'] = test_data['Embarked'].fillna(method ='pad')
test_data
test_data = Change_obj_type(test_data)
test_data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

imput_train = imputer.fit(test_data)
test_data = imput_train.transform(test_data)
test_data
#Using RGBMClassifier regression
y_pred = model.predict(test_data)
y_pred = y_pred.astype(int)
y_pred
# using rnadomforestclassifier
pred = RFC.predict(test_data)
pred = pred.astype(int)
pred
new_data = pd.DataFrame({ 'PassengerId' : t_data['PassengerId'], 'Survived': y_pred})
new_data.to_csv('Titanic.csv',mode='w',index=False)
new_data
sample = pd.read_csv("../input/titanic/gender_submission.csv")
sample