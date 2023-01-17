import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Reading the data

train = pd.read_csv('../input/train.csv')

y_train = train.iloc[:,1].values



train.sample(5)
#Getting rid of the columns we dont need

train = train.drop(['Survived','PassengerId','Name','Ticket','Cabin', 'Embarked'], axis=1)



# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

train.iloc[:,1] = labelencoder_X.fit_transform(train.iloc[:,1])
#Eliminate Siblings, parents or marriage, to create a new feature isAlone

length = train.iloc[:,3].size

for j in range(3,5):

    for i in range(0, length):

        if train.iloc[i,j] > 0:

            train.iloc[i,j] = 1

            

#Create a new feature isAlone

train['isAlone'] = pd.Series(np.zeros(length), index=train.index)

for i in range(0, length):

    if train.iloc[i,3] == 0 and train.iloc[i,4] == 0:

        train.iloc[i,6] = 1



#drop columns

train = train.drop(['SibSp', 'Parch'], axis=1)



#Taking care of the missing data

def imputeData(data):

    from sklearn.preprocessing import Imputer

    imputer = Imputer(missing_values = 'NaN', axis = 0)

    imputer = imputer.fit(data)

    data = imputer.transform(data)

    return data



train.iloc[:,:] = imputeData(train.iloc[:,:])







#Splitting the data

from sklearn.model_selection import train_test_split

train, test, y_train, y_test = train_test_split(train, y_train, test_size = 0.25)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train.iloc[:,0] = sc.fit_transform(train.iloc[:,0].values.reshape(-1,1))

train.iloc[:,2:4] = sc.fit_transform(train.iloc[:,2:4].values.reshape(-1,2))

test = sc.fit_transform(test)
# Applying Grid Search to find the best model and the best parameters

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [10,14], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
#trying with SVC

from sklearn.svm import SVC

classifier = SVC()

parameters = { 'C': [0.1, 0.5, 1],

              'kernel': ['poly', 'rbf', 'sigmoid', 'linear'],

              'degree': [2, 3, 4],

              'decision_function_shape': ['ovo', 'ovr'],

              'class_weight': ['balanced', None]

}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



parameters = { 'C': [0.1,0.3, 0.5, 0.8, 1],

}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
#Applying KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

parameters = {'n_neighbors': [3,4,5,6,7],

              'weights': ['uniform', 'distance'],

              'algorithm': ['auto', 'ball_tree', 'kd_tree','brute'],

              'leaf_size': [25,30,35,40],

              'p': [1,2],

}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)