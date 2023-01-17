# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.preprocessing

from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier as gb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier



# Prescaler:

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



X["Age"] = np.nan_to_num(X["Age"])

# X["Age"] = img.fit_transform(X[["Age"]])
img = SimpleImputer(missing_values= 0.0 , strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)



train_data["Age"] = np.nan_to_num(train_data["Age"])

test_data["Age"] = np.nan_to_num(test_data["Age"])



Xinfo = pd.get_dummies(train_data[features])

Xinfo["Age"] = img.fit_transform(Xinfo[["Age"]])



Test_info = pd.get_dummies(test_data[features])

Test_info["Age"] = img.fit_transform(Test_info[["Age"]])





xx_train, xx_test, yy_train, yy_test = train_test_split(

    Xinfo, train_data["Survived"], random_state = 0)



convert_xtrain = pd.get_dummies(xx_train)

convert_xtest = pd.get_dummies(xx_test)
# print(np.nan_to_num(X["Age"]))

# print(X.info())



model = RandomForestClassifier(n_estimators=400, max_depth=5, random_state=1)

model.fit(X, y)



print("{:.3f}".format(model.score(X, y)))
model_new = gb(random_state = 0, max_depth = 2, learning_rate = 0.35)

model_new.fit(convert_xtrain, yy_train)





# grid search for the best rate

# for j in range(1, 6):

#     max_score = 0

#     max_learning_rate = 0

#     for i in range(1, 100):

#         model_new = gb(random_state = 0, max_depth = j, learning_rate = (i / 100), n_estimators = 100)

#         model_new.fit(convert_xtrain, yy_train)

#         score = model_new.score(convert_xtest, yy_test)

#         if (score > max_score):

#             max_score = score

#             max_learning_rate = i / 100

#     print("{:.3f}, {rate}, {depth}".format(max_score, rate = max_learning_rate, depth = j))

    

# print("---------------------------------")

print("Last best score: {:.3f}".format(model_new.score(convert_xtest, yy_test)))



# predictions = model_new.predict(Test_info)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# print(output)

# output.to_csv('my_submission.csv', index=False)

print("---------------------------------success-----------------------------------")

svc = SVC()

# print(convert_xtrain.info())

# print(convert_xtrain)

svc.fit(convert_xtrain, yy_train)

print("train:{:.3f}".format(svc.score(convert_xtrain, yy_train)))

print("test:{:.3f}".format(svc.score(convert_xtest, yy_test)))

print("---------------normal------------------------")



# robustscaler

scaler1 = RobustScaler()

x_train_scaler = scaler1.fit_transform(convert_xtrain)

x_test_scaler = scaler1.fit_transform(convert_xtest)

svc.fit(x_train_scaler, yy_train)

print("train:{:.3f}".format(svc.score(x_train_scaler, yy_train)))

print("test:{:.3f}".format(svc.score(x_test_scaler, yy_test)))

print("--------------robustscaler-------------------")



# minmaxscaler:

scaler2 = MinMaxScaler()

x_train_scaler = scaler2.fit_transform(convert_xtrain)

x_test_scaler = scaler2.fit_transform(convert_xtest)

svc.fit(x_train_scaler, yy_train)

print("train:{:.3f}".format(svc.score(x_train_scaler, yy_train)))

print("test:{:.3f}".format(svc.score(x_test_scaler, yy_test)))

print("--------------minmaxscaler-------------------")



# Normalizer:

scaler3 = Normalizer()

x_train_scaler = scaler3.fit_transform(convert_xtrain)

x_test_scaler = scaler3.fit_transform(convert_xtest)

svc.fit(x_train_scaler, yy_train)

print("train:{:.3f}".format(svc.score(x_train_scaler, yy_train)))

print("test:{:.3f}".format(svc.score(x_test_scaler, yy_test)))

print("--------------Normalizer---------------------")





# StandardScaler:

scaler3 = StandardScaler()

x_train_scaler = scaler3.fit_transform(convert_xtrain)

x_test_scaler = scaler3.fit_transform(convert_xtest)

test_test_scaler = scaler3.fit_transform(Test_info)



svc.fit(x_train_scaler, yy_train)

print("train:{:.3f}".format(svc.score(x_train_scaler, yy_train)))

print("test:{:.3f}".format(svc.score(x_test_scaler, yy_test)))

print("--------------StandardScaler-----------------")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],

             'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}



grid_search = GridSearchCV(SVC(), param_grid, cv = 5)

grid_search.fit(x_train_scaler, yy_train)



print("train:{:.3f}".format(grid_search.score(x_train_scaler, yy_train)))

print("test:{:.3f}".format(grid_search.score(x_test_scaler, yy_test)))

print("test param:{}".format(grid_search.best_params_))



results = pd.DataFrame(grid_search.cv_results_)

display(results.head())
svc_best = SVC(C = 10, gamma = 0.1)

model_sg = XGBClassifier()

model_sg.fit(x_train_scaler, yy_train)

svc_best.fit(x_train_scaler, yy_train)

print("train:{:.3f}".format(model_sg.score(x_train_scaler, yy_train)))

print("test:{:.3f}".format(model_sg.score(x_test_scaler, yy_test)))

# print("train:{:.3f}".format(svc_best.score(x_train_scaler, yy_train)))

# print("test:{:.3f}".format(svc_best.score(x_test_scaler, yy_test)))





# print(test_test_scaler)



predictions = model_sg.predict(test_test_scaler)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

# display(output['Survived'].value_counts())

print("-----------success----------")