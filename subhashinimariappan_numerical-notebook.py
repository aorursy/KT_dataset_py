#importing libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
#importing data from my activity url

file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter04/Dataset/phpB0xrNj.csv'
#reading data

df = pd.read_csv(file_url)
df.head()
#data preprocessing

y = df.pop('class')
#splitting train and test data.

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=888)
#training model

def train_rf(X_train, y_train, random_state=888, n_estimators=10, max_depth=None, min_samples_leaf=1, max_features='sqrt'):

  rf_model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features)

  rf_model.fit(X_train, y_train)

  return rf_model
rf_1 = train_rf(X_train, y_train)

rf_1.get_params()
#getting model prediction

def get_preds(rf_model, X_train, X_test):

  train_preds = rf_model.predict(X_train)

  test_preds = rf_model.predict(X_test)

  return train_preds, test_preds
trn_preds, tst_preds = get_preds(rf_1, X_train, X_test)
#printing accuracy

def print_accuracy(y_train, y_test, train_preds, test_preds):

  train_acc = accuracy_score(y_train, train_preds)

  test_acc = accuracy_score(y_test, test_preds)

  print(train_acc)

  print(test_acc)

  return train_acc, test_acc
trn_acc, tst_preds = print_accuracy(y_train, y_test, trn_preds, tst_preds)
def fit_predict_rf(X_train, X_test, y_train, y_test, random_state=888, n_estimators=10, max_depth=None, min_samples_leaf=1, max_features='sqrt'):

  rf_model = train_rf(X_train, y_train, random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features)

  train_preds, test_preds = get_preds(rf_model, X_train, X_test)

  train_acc, test_acc = print_accuracy(y_train, y_test, train_preds, test_preds)

  return rf_model, train_preds, test_preds, train_acc, test_acc
#training and testing model for accuracy.

rf_model_1, trn_preds_1, tst_preds_1, trn_acc_1, tst_acc_1 = fit_predict_rf(X_train, X_test, y_train, y_test, random_state=888, n_estimators=20, max_depth=None, min_samples_leaf=1, max_features='sqrt')
#training and testing model again for higher accuracy

rf_model_2, trn_preds_2, tst_preds_2, trn_acc_2, tst_acc_2 = fit_predict_rf(X_train, X_test, y_train, y_test, random_state=888, n_estimators=50, max_depth=None, min_samples_leaf=1, max_features='sqrt')