# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.metrics import classification_report



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_breast_cancer







data   = load_breast_cancer()





print(data)
X = pd.DataFrame(data.data, columns = data.feature_names)



X.head()
y = pd.DataFrame(data.target , columns = ['target'])

y.head()
X.info()
y['target'].value_counts()
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 0) 

print(X_train.shape)

print(X_test.shape)
from sklearn.tree import DecisionTreeClassifier  

classifier1 = DecisionTreeClassifier()  

classifier1.fit(X_train, y_train) 





y_pred_1 = classifier1.predict(X_test)  

print(confusion_matrix(y_test, y_pred_1))

target_names = ['Benign', 'Malignant']

print(classification_report(y_test, y_pred_1, target_names=target_names))



from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc_1 = accuracy_score(y_test,y_pred_1)

print("Accuracy for Gini model {} %".format(acc_1*100))





from sklearn.ensemble import RandomForestClassifier

rclf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42)



rclf.fit(X_train, y_train)



ry_pred = rclf.predict(X_test) 



print(confusion_matrix(y_test, ry_pred))

target_names = ['Benign', 'Malignant']

print(classification_report(y_test, ry_pred, target_names=target_names))



from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,ry_pred)

print("Accuracy for this model {} %".format(acc*100))
print(classifier1.feature_importances_)
imp = list(zip(data.feature_names  , classifier1.feature_importances_))

print(imp)
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()



gb_clf.fit(X_train, y_train)



gb_pred = gb_clf.predict(X_test) 



print(confusion_matrix(y_test, gb_pred))



target_names = ['Benign', 'Malignant']

print(classification_report(y_test, gb_pred, target_names=target_names))





from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,gb_pred)

print("Accuracy for this model {} %".format(acc*100))
from sklearn.svm import LinearSVC

l_svm_clf = LinearSVC()



l_svm_clf.fit(X_train, y_train)



l_svm_pred = l_svm_clf.predict(X_test) 



print(confusion_matrix(y_test, l_svm_pred))



target_names = ['Benign', 'Malignant']

print(classification_report(y_test, l_svm_pred, target_names=target_names))





from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,l_svm_pred)

print("Accuracy for this model {} %".format(acc*100))
from sklearn import svm

svm_clf = svm.SVC(gamma='scale')



svm_clf.fit(X_train, y_train)



svm_pred = svm_clf.predict(X_test) 



print(confusion_matrix(y_test, svm_pred))



target_names = ['Benign', 'Malignant']

print(classification_report(y_test, svm_pred, target_names=target_names))





from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,svm_pred)

print("Accuracy for this model {} %".format(acc*100))
import xgboost as xgb





xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=40)

xgb_model.fit(X_train, y_train)



y_pred = xgb_model.predict(X_test)



print(confusion_matrix(y_test, y_pred))



target_names = ['Benign', 'Malignant']

print(classification_report(y_test, y_pred, target_names=target_names))



acc = accuracy_score(y_test,y_pred)

print("Accuracy for this model {} %".format(acc*100))
import lightgbm



categorical_features = [c for c, col in enumerate(X_train.columns) if 'cat' in col]

train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=categorical_features)

test_data = lightgbm.Dataset(X_test, label=y_test)



parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

}



model = lightgbm.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=100)



y_pred = model.predict(X_test)

new_pred =[]

for val in y_pred :

    if val > 0.5 :

        new_pred.append(1)

    else :

        new_pred.append(0)



#print(new_pred)



print(confusion_matrix(y_test, new_pred))



target_names = ['Benign', 'Malignant']

print(classification_report(y_test, new_pred, target_names=target_names))



acc = accuracy_score(y_test,new_pred)

print("Accuracy for this model {} %".format(acc*100))
