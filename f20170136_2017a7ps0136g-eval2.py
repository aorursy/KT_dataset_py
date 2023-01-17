import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
train.head()
train.info()
train.describe()
print (train.isnull().any(axis = 0))
a = list(train.columns)

a.remove('class')

a.remove('id')

# print(a)

features = a

X = train[features]

y=train[['class']].values.reshape(120)
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000)  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from RandomForestClassifier is : {accuracy}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier(n_estimators=1000)  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from ExtraTreesClassifier is : {accuracy}")

# y_pred
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from naiveBayes is : {accuracy}")
from xgboost import XGBClassifier

classifier = XGBClassifier(max_depth = 10,  n_estimators=1000)  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from XGBClassifier is : {accuracy}")
from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier()  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from AdaBoostClassifier is : {accuracy}")
from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(n_estimators = 1000)  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from GradientBoostingClassifier is : {accuracy}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import ExtraTreesClassifier

from matplotlib.legend_handler import HandlerLine2D



# learning_rates = np.arange(0.01,0.5,0.01)

n_estimators_s = range(100, 3000, 100)

# max_depths = range(1,50)

train_results=[]

test_results=[]



val_learning_rates = []

for ne in n_estimators_s:

    classifier = ExtraTreesClassifier(n_estimators = ne, max_depth = 8)  

    classifier.fit(X_train, y_train)

    

    train_pred = classifier.predict(X_train)

    train_pred = train_pred.round()

    accuracy = accuracy_score(y_train, train_pred)

    train_results.append(accuracy)

    

    y_pred = classifier.predict(X_test)

    y_pred = y_pred.round()

    accuracy = accuracy_score(y_test, y_pred)

    test_results.append(accuracy)

    val_learning_rates.append(eta)



plt.figure(figsize=(10,10))

plt.title("n_estimators accuracy's")

line1,= plt.plot(n_estimators_s, train_results, label="Train accuracy")

line2, = plt.plot(n_estimators_s, test_results, label="Test accuracy")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel("accuracy values")

plt.xlabel("n_estimators")

plt.show
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from xgboost import XGBClassifier

from matplotlib.legend_handler import HandlerLine2D



# learning_rates = np.arange(0.01,0.5,0.01)

gammas = [i/100 for i in range(0,100)]

train_results=[]

test_results=[]



val_learning_rates = []

for gm in gammas:

    classifier = XGBClassifier(min_child_weight = 7, max_depth = 5, gamma = gm)  

    classifier.fit(X_train, y_train)

    

    train_pred = classifier.predict(X_train)

    train_pred = train_pred.round()

    accuracy = accuracy_score(y_train, train_pred)

    train_results.append(accuracy)

    

    y_pred = classifier.predict(X_test)

    y_pred = y_pred.round()

    accuracy = accuracy_score(y_test, y_pred)

    test_results.append(accuracy)

#     val_learning_rates.append(eta)



plt.figure(figsize=(10,10))

plt.title("gammas accuracy's")

line1,= plt.plot(gammas, train_results, label="Train accuracy")

line2, = plt.plot(gammas, test_results, label="Test accuracy")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel("accuracy values")

plt.xlabel("gammas")

plt.show
from sklearn.model_selection import GridSearchCV



param_test1 = {

    'max_depth':range(1,10,1),

    'min_child_weight':range(1,6,1)

#     'gamma':range(0,1)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, gamma = 0), 

 param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X,y)

gsearch1.best_params_
param_test2= {

#     'max_depth':range(1,10,1),

#     'min_child_weight':range(1,6,1)

    'gamma':[i/100.0 for i in range(0,100, 1)]

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, gamma = 0, max_depth= 5, min_child_weight= 3), 

 param_grid = param_test2, scoring='accuracy',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X,y)

gsearch1.best_params_
param_test3= {

#     'subsample':[i/100.0 for i in range(75,90,5)],

#     'colsample_bytree':[i/100.0 for i in range(75,90,5)],

#     'subsample':[i/100.0 for i in range(0,100,5)],

#     'colsample_bytree':[i/100.0 for i in range(0,100,5)],

    'subsample':[i/100.0 for i in range(75,100, 5)],

    'colsample_bytree':[i/100.0 for i in range(30,90, 5)]

#     'subsample':[i/10.0 for i in range(6,10)],

#     'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, gamma = 0.21, max_depth= 5, min_child_weight= 3), 

 param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X,y)

gsearch1.best_params_
param_test4= {

    'reg_alpha':[0,1e-5, 1e-2, 0.1, 1, 100]

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, gamma = 0.21, 

                                                  max_depth= 5, min_child_weight= 3,

                                                colsample_bytree = 0.45, subsample= 0.95), 

 param_grid = param_test4, scoring='accuracy',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X,y)

gsearch1.best_params_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from xgboost import XGBClassifier

classifier = XGBClassifier(n_estimators=1000, max_depth=5, min_child_weight= 3, gamma = 0.21, colsample_bytree= 0.45, subsample= 0.95)  

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

y_pred = y_pred.round()

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy from XGBClassifier is : {accuracy}")
# Submission 1, parameter tuning didn't result in any significant change in the output, so chose the one that gave me the best result for public.



from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X, y)

test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

test_features = list(test.columns)

test_features.remove('id')

print(test_features)

test_points = test[test_features]

# test_points

y_pred_file = model.predict(test_points)

y_pred_file

final_df = pd.DataFrame({ 'id' : test['id'], 'class' : y_pred_file})

final_df.to_string(index = False)

final_df

# final_df.to_csv("predictions_extraTreesClassifierv3.csv")
#submission 2

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X, y)

test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

test_features = list(test.columns)

test_features.remove('id')

print(test_features)

test_points = test[test_features]

# test_points

y_pred_file = model.predict(test_points)

y_pred_file

final_df = pd.DataFrame({ 'id' : test['id'], 'class' : y_pred_file})

final_df.to_string(index = False)

final_df