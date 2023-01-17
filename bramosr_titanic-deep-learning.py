# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
cabin_Nan = df_train['Cabin'].isna().sum()
cabin_Values = df_train['Cabin'].count()
print('% of nulls in Cabin Colums is: ' + str((cabin_Nan / (cabin_Values + cabin_Nan))*100) + '%')
df_train = df_train.drop(['Cabin'], axis = 1)
df_test = df_test.drop(['Cabin'], axis = 1)
cabin_Nan = df_train['Embarked'].isna().sum()
cabin_Values = df_train['Embarked'].count()
print('% of nulls in Embarked Colums is: ' + str((cabin_Nan / (cabin_Values + cabin_Nan))*100) + '%')
df_train = df_train.dropna(subset=['Embarked'])
df_test = df_test.dropna(subset=['Embarked'])
#df_train = df_train.drop(['Embarked'], axis = 1)

df_train = df_train.drop(columns = ['Name', 'Ticket'], axis = 1)
X = df_train.iloc[:, 2: ].values
y = df_train.iloc[:, 1].values

df_test = df_test.drop(columns = ['Name', 'Ticket'], axis = 1)

X_final = df_test.iloc[:, 1:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
imputer.fit(X_final[:, 2:3])
X_final[:, 2:3] = imputer.transform(X_final[:, 2:3])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_final = np.array(ct.fit_transform(X_final))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
X_final = np.array(ct.fit_transform(X_final))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_final = sc.fit_transform(X_final)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 10))
#In case of overfit:
#classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 15, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = [int(elem) for elem in y_pred]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
def calculate_accuracy():
    right = cm[0][0] + cm[1,1]
    total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] 
    print(str((right/total) * 100) + '%')
calculate_accuracy()
y_pred = classifier.predict(X_final)
y_pred = (y_pred > 0.5)
y_pred = [int(elem) for elem in y_pred]
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission_ANN_2.csv', index=False)
print("Your submission was successfully saved!")
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
print('The mean accuracy is:' + str(mean) + ' and the variance is: ' + str(variance))
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer, output_dim):
    classifier = Sequential()
    classifier.add(Dense(output_dim = output_dim, init = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(output_dim = output_dim, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [15,27],
              'nb_epoch' : [100,300],
              'optimizer' : ['adam', 'rmsprop'],
              'output_dim' : [5,6]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy
best_parameters