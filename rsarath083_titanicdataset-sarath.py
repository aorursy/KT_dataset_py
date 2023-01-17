#Data Analysis



import pandas as pd

import numpy as np



#Data Visualisation



import matplotlib.pyplot as plt

import seaborn as sns
#Acquiring data





train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

#Viewing the train data



train.head()
#Viewing the train data



test.head()

test.shape
train.info()
test.shape
test.info()
#Statistical details of numerical columns



train.describe()
#Plotting a heatmap to visualise missing values



plt.figure(figsize = (10,8))

sns.heatmap(train.isnull())
#Countplot of Survived column



sns.countplot(x='Survived',data = train)
#Countplot of Survived column WRT Sex column



sns.countplot(x='Survived',hue = 'Sex', data = train)
#Countplot of Survived column WRT Pclass column



sns.countplot(x='Survived', hue='Pclass', data=train)
sns.countplot(x='SibSp', data = train)
import pandas_profiling as pdp



pandas_profiling = pdp.ProfileReport(train)

pandas_profiling
pandas_profiling = pdp.ProfileReport(test)

pandas_profiling
#Plotting a boxplot to understand the mean age in each class



plt.figure(figsize = (10,7))

sns.boxplot(x='Pclass', y='Age', data= train)
#Function to fill misssing values in Age column WRT the findings from boxplot





def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else :

            return 24

    else:

        return Age
#Filling the missing values in Age Column



train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)

test['Age'] = test[['Age','Pclass']].apply(impute_age, axis = 1)
sns.heatmap(train.isnull())
#Dropping Cabin column because 687 out of 891, 77.1% are missing values.



train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)
#Dropped cabin column



sns.heatmap(train.isnull())
#Filling the Embarked column with the most frequent entry in the column



train["Embarked"].fillna('S', inplace = True) 

  

test["Embarked"].fillna('S', inplace = True)



#Filling the Fare column with the mean values



test["Fare"].fillna(35.627, inplace = True)
#heatmap for train dataset



sns.heatmap(train.isnull())
#heatmap for train dataset



sns.heatmap(test.isnull())
#Using pandas dummies method for encoding



sex = pd.get_dummies(train['Sex'], drop_first = True)
#Using pandas dummies method for encoding



embark = pd.get_dummies(train['Embarked'], drop_first = True)
#Merging encoded columns to dataset



train = pd.concat([train,sex,embark], axis =1)
#Applying the same for test dataset



sex = pd.get_dummies(test['Sex'], drop_first = True)

embark = pd.get_dummies(test['Embarked'], drop_first = True)

test = pd.concat([test,sex,embark], axis =1)
#Dropping Sex and Embarked column as we already have encoded columns

#Dropping Name, Ticket, PassengerID coulmn as they dont provide relevant information for the problem statement



train.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis =1, inplace= True)
#Dropping Sex and Embarked column as we already have encoded columns

#Dropping Name, Ticket coulmn as they dont provide relevant information for the problem statement



test.drop(['Sex','Embarked','Name','Ticket'], axis =1, inplace= True)
#Final train dataset looks like this



train.head()
#Final test dataset looks like this



test.head()
#Splitting dependant and independant variales



X_train = train.drop("Survived", axis=1)

y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()



X_train.shape, y_train.shape, X_test.shape
X_train.head()
X_test.head()
#importing machine learning algorithms



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
# Logistic Regression



lr = LogisticRegression()

lr.fit(X_train, y_train)

Y_pred = lr.predict(X_test)

acc_lr = round(lr.score(X_train, y_train) * 100, 2)

acc_lr
# # Support Vector Machines



# svc = SVC()

# svc.fit(X_train, y_train)

# y_pred = svc.predict(X_test)

# acc_svc = round(svc.score(X_train, y_train) * 100, 2)

# acc_svc
# param_grid = [

#   {'C': [0.1,1, 10, 100, 1000], 'kernel': ['linear']},

#   {'C': [0.11, 10, 100, 1000],'gamma': [1,0.1,0.001,0.0001] , 'kernel': ['rbf']}]



# from sklearn.model_selection import GridSearchCV



# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 10)
# grid.fit(X_train, y_train)
# grid.best_params_
# grid.best_estimator_
# grid_predictions = grid.predict(X_test)

# acc_svc = round(grid_predictions.score(X_train, y_train) * 100, 2)

# acc_svc
# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# acc_knn = round(knn.score(X_train, y_train) * 100, 2)

# acc_knn
# # Decision Tree



# dt = DecisionTreeClassifier()

# dt.fit(X_train, y_train)

# y_pred = dt.predict(X_test)

# acc_dt = round(dt.score(X_train, y_train) * 100, 2)

# acc_dt
# ## Random Forest



# rf = RandomForestClassifier(n_estimators=1)

# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)

# rf.score(X_train, y_train)

# acc_rf = round(rf.score(X_train, y_train) * 100, 2)

# acc_rf
# from pprint import pprint

# print('Parameters currently in use:\n')

# pprint(rf.get_params())




# #importing RandomizedSearchCV



# from sklearn.model_selection import RandomizedSearchCV



# # Number of trees in random forest



# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]



# # Number of features to consider at every split



# max_features = ['auto', 'sqrt']



# # Maximum number of levels in tree



# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# max_depth.append(None)



# # Minimum number of samples required to split a node



# min_samples_split = [2, 5, 10]



# # Minimum number of samples required at each leaf node



# min_samples_leaf = [1, 2, 4]



# # Method of selecting samples for training each tree



# bootstrap = [True, False]



# # Create the random grid



# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}



# pprint(random_grid)

# {'bootstrap': [True, False],

#  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

#  'max_features': ['auto', 'sqrt'],

#  'min_samples_leaf': [1, 2, 4, 6, 8],

#  'min_samples_split': [2, 5, 10, 15],

#  'n_estimators': [100,200, 400, 600,700, 800, 900, 1000,1100, 1200,1300, 1400, 1600, 1800, 2000]}
# # Use the random grid to search for best hyperparameters



# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



# # Fit the random search model

# rf_random.fit(X_train, y_train)
# #Best parameters



# rf_random.best_params_
# #predictions



# grid_predictions = rf_random.predict(X_test)
# models = pd.DataFrame({

#     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

#               'Random Forest','Decision Tree'],

#     'Confidence Score': [acc_svc, acc_knn, acc_lr, 

#               acc_rf,acc_dt]})

# models.sort_values(by='Confidence Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })





submission.to_csv('submission.csv', index = False) 



submission