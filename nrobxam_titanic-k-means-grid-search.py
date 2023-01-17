import numpy as np

import pandas as pd



from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.preprocessing import LabelEncoder
# making a dataframe for the train sample



df_train = pd.read_csv('../input/titanic/train.csv')

print(df_train.shape)

df_train.head()
# dealing with missing values



print(df_train.isnull().sum())
# filling missing Age values with the mean 

df_train.Age[df_train.Age.isnull()] = df_train.Age.mean()



# filling missing Embarked values with the majority

df_train.Embarked[df_train.Embarked.isnull()] = df_train.groupby('Embarked').count()['PassengerId'][df_train.groupby('Embarked').count()['PassengerId'] == df_train.groupby('Embarked').count()['PassengerId'].max()].index[0]
# dropping currently useless features



df_train = df_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_train.columns
# initiating label encoder for sklearn

label = LabelEncoder()

dicts = {}



# initiating the labels for sex column

label.fit(df_train.Sex.drop_duplicates())

dicts['Sex'] = list(label.classes_)

# coding the sex column values

df_train.Sex = label.transform(df_train.Sex)



# initiating the labels for Embarked column

label.fit(df_train.Embarked.drop_duplicates())

dicts['Embarked'] = list(label.classes_)

# coding the sex column values

df_train.Embarked = label.transform(df_train.Embarked)
# initializing the target variable and dropping it from features dataframe

y = df_train.Survived

X = df_train.drop(['Survived'], axis=1)
print(X.columns)

print(X.shape, y.shape)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
# initializing the dictionaries for the parameters that will be passed to GridSearchCV method



## for the number of k neighbors

k = list(range(1, 60, 2))



## for the weights

weights_options = ['uniform', 'distance']



## for the algorithms applied 

algos = ['ball_tree', 'kd_tree', 'brute']



## leaf size (since i've initiated BallTree and KDTree algorithms)

leaves = list(np.arange(10, 110, 10))



## for the metrics

metric_options = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']



## for the parameters of the metrics

#metric_params=metric_param_options



# initializing the grid



params_grid = dict(n_neighbors=k, weights=weights_options, algorithm=algos, leaf_size=leaves, metric=metric_options, )



# initializing the grid search with 10 cross_validation splits



model_titanic = KNeighborsClassifier() 



grid = GridSearchCV(model_titanic, params_grid, cv=10, scoring='accuracy')



# training the model

grid.fit(X_train, y_train)
print(f'best parameters: {grid.best_params_},\nbest accuracy score: {grid.best_score_},\nbest estimator: {grid.best_estimator_}')
df_test = pd.read_csv('../input/titanic/test.csv')

df_test.columns
# checking the null values

df_test.info()
# proccesing the test features



X_test = df_test



X_test.Age[X_test.Age.isnull()] = X_test.Age.mean()

X_test.Embarked[X_test.Embarked.isnull()] = X_test.groupby('Embarked').count()['PassengerId'][X_test.groupby('Embarked').count()['PassengerId'] == X_test.groupby('Embarked').count()['PassengerId'].max()].index[0]



# got a single missing value in column Fare

X_test.Fare[X_test.Fare.isnull()] = X_test.Fare.median()



result = pd.DataFrame(df_test.PassengerId)

X_test = df_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)



label = LabelEncoder()

dicts = {}



label.fit(X_test.Sex.drop_duplicates())

dicts['Sex'] = list(label.classes_)



label.fit(X_test.Sex.drop_duplicates())

dicts['Sex'] = list(label.classes_)

X_test.Sex = label.transform(X_test.Sex)





label.fit(X_test.Embarked.drop_duplicates())

dicts['Embarked'] = list(label.classes_)

X_test.Embarked = label.transform(X_test.Embarked)
# duplicating the df_test to save PassengerId columns



print(df_test.columns, df_test.shape)

print(X_test.columns, X_test.shape)
# launching prediction based on best grid parameters



predictions = grid.predict(X_test)

predictions
# makeing the submission dataframe



submit = pd.DataFrame(list(zip(df_test.PassengerId, predictions)), columns = ['PassengerId', 'Survived'])

submit.head()
# saving the submission dataframe to csv

submit.to_csv('submission.csv', sep=',', index=False)