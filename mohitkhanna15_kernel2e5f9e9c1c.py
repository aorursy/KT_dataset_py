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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
df.shape
df.info()
# Percentage missing values

df.isnull().sum()*100/df.shape[0]
df.describe()
# % of male  boarded the ship

len(df[df.Sex == 'male'])*100/df.shape[0]
# % of male  boarded the ship

len(df[df.Sex == 'female'])*100/df.shape[0]
# Out of the total passengers, what is the percentage of females those survived the incident?

pd.crosstab(df.Survived, df.Sex, normalize=True)
# Question: Which age group survived the most?

# "Age" <= 16: 0

# > 16  & <= 32 :1

# > 32 & <= 48 :2

# > 48 & <= 64 :3

# "Age" > 64 :4

df['Age_group'] = pd.cut(df.Age, bins = [0,16,32,48,64,200], labels = [0,1,2,3,4])
sns.countplot("Age_group", hue = "Survived", data = df)
df.drop('Age_group', axis =1, inplace = True)
## Missing Value Treatment

# Percentage of missing

100*(df.isnull().sum()/df.shape[0])
# Dropping Cabin

df.drop('Cabin', axis =1, inplace= True)
# Check the Embarked variable, which of the following category has the highest count?

df.Embarked.value_counts()
## Impute S for missing at Embarked columns

df.Embarked = df.Embarked.fillna('S')
# Percentage of missing

100*(df.isnull().sum()/df.shape[0])
# Replace Sex columns

df.Sex.replace(['female', 'male'], [0,1], inplace = True)
## Label Encoder on Embarked Varibale

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df.Embarked = le.fit_transform(df.Embarked)

df.head()
# Drop PassengerID, Name and Ticket as they dont contribute to modelign

df.drop(['PassengerId', 'Name', 'Ticket'], inplace = True, axis = 1)

df.head()
# Preserve the column names

df_col = df.columns
from fancyimpute import IterativeImputer
df_clean = pd.DataFrame(IterativeImputer().fit_transform(df))

df_clean.columns = df_col

df_clean.head()
# Percentage of missing

100*(df_clean.isnull().sum()/df_clean.shape[0])
df_clean.info()
#  Check the outliers in the “Age” variable, Is there any outliers?

sns.boxplot(df_clean.Age)
# Let's remove the -ve age

df_clean.drop(df_clean.index[df_clean.Age<0], inplace = True)
sns.boxplot(df_clean.Age)
#  Check Fare variable, is there any outliers in the fare variable?

sns.boxplot(df_clean.Fare)
# Drop the row that have fare greater than 300



df_clean.drop(df_clean.index[df_clean.Fare>300], inplace = True)
df.Fare.quantile(0.90)
df_clean.info()
# This required my cat data in String format(object)

# Let's convert all the cat data into string or object

for i in ['Pclass', 'SibSp', 'Parch','Embarked']:

  df_clean[i] = df_clean[i].astype(str)
df_clean.info()
cat_data = df_clean[['Pclass', 'SibSp', 'Parch','Embarked']]
df_dummies = pd.get_dummies(cat_data, drop_first=True)

df_dummies.head()
df_clean.drop(list(cat_data.columns), axis = 1, inplace = True)

df_clean = pd.concat([df_clean, df_dummies], axis = 1)

df_clean.head()
# Split Data into Train - Test

X = df_clean.drop('Survived', axis = 1)

y = df_clean.Survived
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)
## Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
## Modellling Part

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print("Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pred)))

print("Sensitivity/Recall : {0}".format(metrics.recall_score(y_test, y_pred)))
logreg = LogisticRegression(penalty='l2') # Ridge

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print("Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pred)))

print("Sensitivity/Recall : {0}".format(metrics.recall_score(y_test, y_pred)))
logreg = LogisticRegression(solver='saga',penalty='l1') # Lasso

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print("Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pred)))

print("Sensitivity/Recall : {0}".format(metrics.recall_score(y_test, y_pred)))
## Class_weight = =Class_imbalace

100*(df_clean.Survived.value_counts()/ df_clean.shape[0])
logreg = LogisticRegression(solver='saga',penalty='l1', C = 0.1 , class_weight = 'balanced') # Lasso

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print("Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pred)))

print("Sensitivity/Recall : {0}".format(metrics.recall_score(y_test, y_pred)))
## We can try range of C: GridSearch

from sklearn.model_selection import GridSearchCV

param = { 'C': [0.0001, 0.001,0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1] ,'penalty': ['l1', 'l2']}

logr = LogisticRegression(class_weight = 'balanced')

model = GridSearchCV(estimator = logr, cv = 5, param_grid = param, scoring = 'recall')

model.fit(X_train, y_train)
model.best_score_
model.best_params_
# Logistic regression with best parameters

logreg = LogisticRegression(penalty='l1', C = 0.3, class_weight = 'balanced')

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print("Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pred)))

print("Sensitivity/Recall : {0}".format(metrics.recall_score(y_test, y_pred)))
## ElastiNet

## Combines -- Ridge and Lasso

logreg = LogisticRegression(penalty='elasticnet', C = 0.1, l1_ratio = 0.2, class_weight = 'balanced', solver = 'saga')

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print("Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pred)))

print("Sensitivity/Recall : {0}".format(metrics.recall_score(y_test, y_pred)))
# Importing random forest classifier from sklearn library

from sklearn.ensemble import RandomForestClassifier



# Running the random forest with default parameters.

rfc = RandomForestClassifier()
# fit

rfc.fit(X_train,y_train)
# Making predictions

predictions = rfc.predict(X_test)
# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(accuracy_score(y_test,predictions))
# Let's check the report of our default model

print(classification_report(y_test,predictions))
# Printing confusion matrix

print(confusion_matrix(y_test,predictions))
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 20, 1)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head(5)
# plotting accuracies with max_depth

plt.figure()



plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(100, 1500, 400)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=8)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal max_features

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_features': [4, 8, 12 , 16]}



# instantiate the model

rf = RandomForestClassifier(max_depth=8)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_features

plt.figure()

plt.plot(scores["param_max_features"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_features")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(100, 400, 50)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_features

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(200, 500, 50)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_features

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 200),

    'n_estimators': [100,200, 300], 

    'max_features': [5, 10 ,12]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data

grid_search.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)