# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.impute import SimpleImputer

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH = "/kaggle/input/titanic/"
train = pd.read_csv(f'{PATH}train.csv')
print(train.shape)

train.describe()
train.columns
train.dtypes
train.isna().sum()
train.head()
test = pd.read_csv(f'{PATH}test.csv')
drop_col = ["PassengerId","Ticket","Name"]

X = train.drop(drop_col + ["Survived"],axis = 1)

y = train["Survived"]

X_test = test.drop(drop_col,axis = 1)

print(f"X {X.shape}")

print(f"y {y.shape}")

print(f"X_test {X_test.shape}")
ohe_X = pd.get_dummies(X)

ohe_X_test = pd.get_dummies(X_test)

ohe_X.head().T
ohe_X_test.head().T
final_ohe_X, final_ohe_X_test = ohe_X.align(ohe_X_test,join="left",axis =1 )

                                                                  
final_ohe_X_test.head()
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val =  train_test_split(final_ohe_X,y,test_size=0.3, random_state=123)

print(f"X_train {X_train.shape}")

print(f"y_train {y_train.shape}")

print(f"X_val {X_val.shape}")

print(f"y_val {y_val.shape}")
def impute_data(X_train,X_valid,X_test):

    cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]

    print(f"cols_with_missing {cols_with_missing}")

    X_train_plus = X_train.copy()

    X_valid_plus = X_valid.copy()

    X_test_plus = X_test.copy()



    # Make new columns indicating what will be imputed

    for col in cols_with_missing:

        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

        X_test_plus[col + '_was_missing'] = X_test_plus[col].isnull()



    # Imputation

    my_imputer = SimpleImputer()

    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    imputed_X_test_plus = pd.DataFrame(my_imputer.transform(X_test_plus))

    



    # Imputation removed column names; put them back

    imputed_X_train_plus.columns = X_train_plus.columns

    imputed_X_valid_plus.columns = X_valid_plus.columns

    imputed_X_test_plus.columns = X_test_plus.columns

    

    

    return imputed_X_train_plus,imputed_X_valid_plus,imputed_X_test_plus

imputed_X_train,imputed_X_val,imputed_X_test = impute_data(X_train,X_val,final_ohe_X_test)
imputed_X_train.T
imputed_X_test.isna().sum()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')

# Fit on training data

model.fit(imputed_X_train, y_train)
from sklearn.metrics import accuracy_score

val_predict = model.predict(imputed_X_val)

print(f"accuracy {accuracy_score(y_val, val_predict)}")
from sklearn import metrics

import matplotlib.pyplot as plt

y_pred_proba = model.predict_proba(imputed_X_val)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_val, y_pred_proba)

auc = metrics.roc_auc_score(y_val, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val =  train_test_split(final_ohe_X,y,test_size=0.3, random_state=123)

print(f"X_train {X_train.shape}")

print(f"y_train {y_train.shape}")

print(f"X_val {X_val.shape}")

print(f"y_val {y_val.shape}")
type(imputed_X_test)
from sklearn.model_selection import cross_val_score

train_X = pd.concat([imputed_X_train,imputed_X_val])

train_y = pd.concat([y_train,y_val])

model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')

# Fit on training data

model.fit(train_X, train_y)

scores = cross_val_score(model, train_X, train_y, cv=5, scoring = "accuracy")

print(scores)

print(scores.mean())
submit_file =  pd.DataFrame()

submit_file['PassengerId'] = test.PassengerId

submit_file['Survived'] = model.predict(imputed_X_test)
submit_file
submit_file.to_csv("submit.csv",index=False)
feat_importances = pd.Series(model.feature_importances_, index=imputed_X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')