## importing libraries required

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go
# Change directory

path = "../input"

os.chdir(path)

print( path )
#read Train and Test Data

X_train_robo=pd.read_csv('X_train.csv')

X_test_robo=pd.read_csv('X_test.csv')

y_train_robo=pd.read_csv('y_train.csv')

sample = pd.read_csv('sample_submission.csv')
X_train_robo.head()
y_train_robo.head()
X_test_robo.head()
print("X_Train : " , X_train_robo.shape)

print("X_Test : " , X_test_robo.shape)

print("y_Train : " , y_train_robo.shape)
print("Train series count : %d" % len(X_train_robo.series_id.value_counts()))

print("Test series count : %d" % len(X_test_robo.series_id.value_counts()))
print("Train measurement_number count : %d" % len(X_train_robo.measurement_number.value_counts()))

print("Test measurement_number count : %d" % len(X_test_robo.measurement_number.value_counts()))
print("X_Train : \n" , X_train_robo.isna().sum())

print("---------------------------")

print("X_Test : \n" , X_test_robo.isna().sum())

print("---------------------------")

print("y_Train : \n" , y_train_robo.isna().sum())
X_train_robo.describe()
X_test_robo.describe()
X_train_grp = X_train_robo.groupby(['series_id'], as_index=False).mean()

print(X_train_grp.shape)

X_train_grp.head()
## test data

test = X_test_robo.groupby(['series_id'], as_index=False).mean()

print(test.shape)

test.head()
train = pd.merge(X_train_grp,y_train_robo, on= ['series_id'])

train.shape
train.drop(["measurement_number"], axis=1, inplace=True)

test.drop(["measurement_number"], axis=1, inplace=True)
print("train:",train.shape)

print("test:",test.shape)
train.dtypes
test.dtypes
# plt.figure(figsize=(10,10)

train.surface.value_counts().plot(kind='bar',

                                 figsize=(10,6),

                                  color="red",

                                  alpha = 0.7,

                                  fontsize=13)

plt.title('Surface (in numbers)')

plt.xlabel('Surface')

plt.ylabel('Count')

plt.show()
plt.figure();

train.hist(bins=50, figsize=(20, 15))
y=train["surface"]

X=train.drop('surface', axis=1)

X=X.drop('group_id', axis=1)



#from sklearn.model_selection import train_test_split  

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20)  

print("X_train", X_train.shape)

print("X_validation", X_validation.shape)

print("y_train", y_train.shape)

print("y_validation", y_validation.shape)

y.unique()
## Random Forest model



seed = 7

num_folds = 10



# Params for Random Forest

num_trees = 100

max_features = 3

models = []

models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))



# evalutate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# random_forest = RandomForestClassifier(n_estimators=50,max_features=5)

# random_forest.fit(X_train, y_train)

# predictions_rf = random_forest.predict(X_validation)

# print("Accuracy: %s%%" % (100*accuracy_score(y_validation, predictions_rf)))

# print('----------------')

# print(confusion_matrix(y_validation, predictions_rf))

# print('----------------')

# print(classification_report(y_validation, predictions_rf))
random_forest1 = RandomForestClassifier(n_estimators=50,max_features=5)

random_forest1.fit(X_train, y_train)

predictions_rf = random_forest1.predict(X_validation)

print("Accuracy: %s%%" % (100*accuracy_score(y_validation, predictions_rf)))

print('----------------')

print(confusion_matrix(y_validation, predictions_rf))

print('----------------')

print(classification_report(y_validation, predictions_rf))
# predication on  test data 

predictions = random_forest1.predict(test)

print(len(predictions))

sample['surface']=predictions

#sample.to_csv('samplesubmission.csv', index=False)

sample.head()