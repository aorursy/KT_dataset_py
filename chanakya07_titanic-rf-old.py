import pandas as pd

import sklearn.metrics.cluster as skmetric

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler



train_data = pd.read_csv('../input/train.csv', sep=',')

train_data = train_data.drop(['Name','Ticket','Cabin','Fare','Embarked','PassengerId'],axis=1)



test_data = pd.read_csv('../input/test.csv', sep=',')

test_data_cp = test_data.copy()

test_data = test_data.drop(['Name','Ticket','Cabin','Fare','Embarked','PassengerId'],axis=1)

# data.dtypes

# data[data.isnull().any(axis=1)]

#Data Cleaning



#Drop rows with null values

train_data = train_data.dropna(axis=0)

# print(test_data)

# test_data.iloc[:,0]
def data_cleaning(data):

  #Data Cleaning



  #Drop rows with null values

  #data = data.dropna(axis=0)

  mode_df = test_data.mode()

  for i in np.arange(0,5):

    mode_value = mode_df.iloc[0,i]

    data.iloc[:,i].fillna(mode_value,inplace=True)

  return data

# test_data = data_cleaning(test_data)

# test_data[test_data.isnull().any(axis=1)]
def get_scaler(x_data):

  scaler = StandardScaler(copy=False)

  scaler.fit(x_data)

  return scaler
def data_pp(data):

  #Data Cleaning



  #Drop rows with null values

  #data = data.dropna(axis=0)



  #Data Preprocessing



  #One hot encoding

  # data = pd.get_dummies(data)

  y_data = None

  try:

    y_data = data['Survived']

  except:

    pass

  x_data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']] #5 Attributes used

  x_data = x_data.replace('male',0)

  x_data = x_data.replace('female',1)

  

  #Feature Scaling

  #scaler = StandardScaler(copy=False)

  #scaler.fit_transform(x_data)

  # print(x_data)

  return (x_data,y_data)
def fit_model(x_data,y_data):

  #Model fitting

  

  #Using Random Forest

  clf = RandomForestClassifier(n_estimators = 5, max_features = 3, max_depth=None, min_samples_split=2)

  clf.fit(x_data, y_data)

  return clf
def predict_y(clf,x_data):

  #Prediction

  y_values = clf.predict(x_data)

  return y_values
# test_data = test_data.dropna(axis=0)

# test_data[test_data.isnull().any(axis=1)



data_cleaning(test_data)



(x_train_data,y_train_data) = data_pp(train_data)

(x_test_data,y_test_data) = data_pp(test_data)



scaler = get_scaler(x_train_data)

scaler.transform(x_train_data)

scaler.transform(x_test_data)



clf = fit_model(x_train_data,y_train_data)



y_pred_val = predict_y(clf,x_test_data)

sol_df = pd.DataFrame({'PassengerId':test_data_cp['PassengerId'],'Survived':y_pred_val})

sol_df.to_csv('./sub-sol-3.csv',sep=',',index=False)