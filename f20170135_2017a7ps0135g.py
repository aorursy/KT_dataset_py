import numpy as np

import pandas as pd

import seaborn as sns

import os
DATA_DIR = '/kaggle/input/eval-lab-1-f464-v2/'



train_file = 'train.csv'

test_file = 'test.csv'

train = pd.read_csv(os.path.join(DATA_DIR, train_file))
train.head()
train.info()
print(train.isnull().sum(axis = 0))
numerical_features = train.columns[1:10].append(train.columns[11:13])

categorical_features = train.columns[10]

rating = train.columns[13]

print(numerical_features)

print(categorical_features)

print(rating)
print(train[numerical_features].mean())

print(train[categorical_features].mode())
train[numerical_features] = train[numerical_features].fillna(train[numerical_features].mean())

train[categorical_features] = train[categorical_features].fillna(train[categorical_features].mode())

train = pd.get_dummies(data = train, columns = ['type'])
train.head()
train.isnull().any().any()
print(train.isnull().sum(axis = 0))
sns.relplot(x = 'feature5', y = 'rating', data = train)
train.head()
features = train.columns[1:12].append(train.columns[13:])

x = train[features]

y = train[rating]
x['type_new'].unique()
print(x.isnull().any().any())

x.head()
y.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.005)
x_train.head()
x_train.shape
scale = RobustScaler()



scale_features = features[:-2]

x_train[scale_features] = scale.fit_transform(x_train[scale_features])

x_test[scale_features] = scale.transform(x_test[scale_features])
scale_features
print(x_train.head())

print(x_test.head())
from sklearn.linear_model import LogisticRegression



model = LogisticRegression(penalty = 'none', solver = 'newton-cg', multi_class = 'multinomial', max_iter = 5000)

model.fit(x_train.values, y_train.values)
print(model.predict(x_test[:10]))

print(y_test.values[:10])

model.score(x_train.values, y_train.values)
model.score(x_test.values, y_test.values)
from sklearn.metrics import mean_squared_error



pred = model.predict(x_test)



print(mean_squared_error(y_test, pred) ** 0.5)
test = pd.read_csv(os.path.join(DATA_DIR, test_file))
test[numerical_features] = test[numerical_features].fillna(test[numerical_features].mean())

test[categorical_features] = test[categorical_features].fillna(test[categorical_features].mode())

test = pd.get_dummies(data = test, columns = ['type'])
print(test.isnull().any().any())

test.head()
x_sub = test[features]

x_sub.head()
x_sub[scale_features] = scale.transform(x_sub[scale_features])
x_sub.head()
pred_sub = model.predict(x_sub.values)

print(pred_sub.shape)
test['id'].values
sub = pd.DataFrame(data = pred_sub, columns = ['rating'], index = test['id'])
sub.head()
sub.isnull().any().any()
sub.to_csv('sub2.csv')
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors = 10)
knn_model.fit(x_train, y_train)
knn_model.score(x_test, y_test)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Initialize and train

dt_model = DecisionTreeClassifier()

rf_model = RandomForestClassifier()
dt_model.fit(x_train, y_train)
print(dt_model.predict(x_test[:10]))

print(y_test.values[:10])
pred_dt = dt_model.predict(x_test)
from sklearn.metrics import accuracy_score



print(accuracy_score(y_test.values, pred_dt))
rf_model.fit(x_train, y_train)
pred_rf = rf_model.predict(x_test)

print(accuracy_score(y_test.values, pred_rf))

print(pred_rf[:10])
print(pd.get_dummies(y_train).sum(0))
print(y_train.shape)

sorted(y_train.unique())
from sklearn.utils import class_weight



cw = class_weight.compute_class_weight('balanced', sorted(y_train.unique()), y_train)

print(cw)
cw_dict = {i : cw[i] for i in range(7)}

print(cw_dict)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.ensemble import RandomForestClassifier



#TODO

rf_model_2 = RandomForestClassifier(class_weight = cw_dict)        #Initialize the classifier object



parameters = {'n_estimators':[100, 1000]}    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_model_2, parameters, scoring = scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(x_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf_model = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (rf_model_2.fit(x_train, y_train)).predict(x_test)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_rf_model.predict(x_test)        #Same, but use the best estimator



acc_unop = accuracy_score(y_test, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_test, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
print(x_test.head())
print(y_test.head())
x_sub.head()
pred_rf_sub = best_rf_model.predict(x_sub.values)

pred_last = rf_model_2.predict(x_sub.values)

print(pred_rf_sub[:10])
print(pred_rf_sub[:100])
sub_rf = pd.DataFrame(data = pred_rf_sub, columns = ['rating'], index = test['id'])

sub_rf_l = pd.DataFrame(data = pred_last, columns = ['rating'], index = test['id'])
sub_rf.to_csv('sub_rf_last.csv')
