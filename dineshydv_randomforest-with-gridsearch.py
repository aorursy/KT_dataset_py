# Basic libraries

import numpy as np

import pandas as pd



# model libraries

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# model accuracy check

from sklearn import metrics



# plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns

# read the data

train_file = '../input/digit-recognizer/train.csv'

test_file = '../input/digit-recognizer/test.csv'



df_train = pd.read_csv(train_file)

df_test = pd.read_csv(test_file)

df_train.head()
df_train.shape
# columns pixel0....pixel785 are independent variable of a digit

# column label contains the digit (dependent variable)



df_train.columns
df_test.shape
df_test.columns
# no null values in train dataset



df_train.isnull().values.any()
df_train.info()
# print the frequency of each label



print(df_train['label'].value_counts())

sns.countplot(df_train['label'])
plt.figure(figsize=(12,4))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(df_train.drop(['label'],axis=1).values[i].reshape(28,28) )

    plt.axis('off')

plt.show()



# print corresponding labels:

print(list(df_train['label'].loc[0:9]))

print(list(df_train['label'].loc[10:19]))

print(list(df_train['label'].loc[20:29]))
plt.figure(figsize=(12,4))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(df_test.values[i].reshape(28,28) )

    plt.axis('off')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['label'],axis=1),

                                                   df_train['label'],

                                                   test_size = 0.2,

                                                   random_state=13)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train.head()
y_train.head()
# build model RandomForest with best parameters



# Set the parameters by cross-validation

parameter_grid = {'n_estimators': (np.arange(10,20)),

                  'criterion': ['gini', 'entropy'],

                  'max_depth': (np.arange(10,30)) }



# use KFold

cross_validation = KFold(n_splits=10)



# define model search and fit

#-- gs = GridSearchCV(RandomForestClassifier(), param_grid=parameter_grid, cv=cross_validation)

#-- gs.fit(X_train, y_train)

#-- print('Best score: {0}'.format(gs.best_score_))

#-- print('Best parameters: {0}'.format(gs.best_params_))



# store best parameter model

#-- m1 = gs.best_estimator_

#-- print(m1)

m1 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

            max_depth=18, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=19, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)



m1.fit(X_train, y_train)

# run preduction on X_test

# then check accuracy on y_test



y_pred = m1.predict(X_test)

y_pred
print('Accuracy score for y_test: ', metrics.accuracy_score(y_test,y_pred))
pd.DataFrame(metrics.confusion_matrix(y_test,y_pred))
X_test['label'] = y_test

X_test['pred'] = y_pred
X_test_err = X_test[X_test['label'] != X_test['pred']]

X_test_err.shape
for i in range(11):  

    print('Label {0}, Prediction {1}'.format(X_test_err['label'].values[i],X_test_err['pred'].values[i]))

    plt.figure(figsize=(1,1))

    plt.imshow(X_test_err.drop(['label','pred'],axis=1).values[i].reshape(28,28) )

    plt.axis('off')

    plt.show()

pred = m1.predict(df_test)

pred
# display digit images

for i in range(11):  

    print('Prediction {0}'.format(pred[i]))

    plt.figure(figsize=(1,1))

    plt.imshow(df_test.values[i].reshape(28,28) )

    plt.axis('off')

    plt.show()

pred = pd.Series(pred,name="Label")
submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)



submit.to_csv("cnn_mnist_randomforest_gridsearch.csv",index=False)