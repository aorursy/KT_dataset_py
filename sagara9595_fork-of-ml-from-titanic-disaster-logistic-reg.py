# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics

from sklearn.metrics import precision_score, recall_score
pd.set_option('display.max_columns', 500)
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_train.head()
titanic_train.shape
titanic_train.info()
titanic_train.isnull().sum()
titanic_train.Cabin.value_counts(dropna = False)
titanic_train.Cabin = titanic_train.Cabin.fillna("Unknown_Cabin")
titanic_train.Cabin = titanic_train.Cabin.str[0]
titanic_train.Cabin.value_counts()
titanic_train.isna().sum()/titanic_train.shape[0] *100
titanic_train.Embarked.value_counts(dropna = False)
titanic_train.Embarked = titanic_train.Embarked.fillna('S')
titanic_train['Title'] = titanic_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

titanic_train = titanic_train.drop('Name',axis=1)
titanic_train.head()
#let's replace a few titles -> "other" and fix a few titles

titanic_train['Title'] = np.where((titanic_train.Title=='Capt') | (titanic_train.Title=='Countess') | \

                                  (titanic_train.Title=='Don') | (titanic_train.Title=='Dona')| (titanic_train.Title=='Jonkheer') \

                                  | (titanic_train.Title=='Lady') | (titanic_train.Title=='Sir') | (titanic_train.Title=='Major') | \

                                  (titanic_train.Title=='Rev') | (titanic_train.Title=='Col'),'Other',titanic_train.Title)



titanic_train['Title'] = titanic_train['Title'].replace('Ms','Miss')

titanic_train['Title'] = titanic_train['Title'].replace('Mlle','Miss')

titanic_train['Title'] = titanic_train['Title'].replace('Mme','Mrs')
titanic_train.Title.value_counts()
titanic_train.groupby('Title').Age.mean()
titanic_train["Age"] = np.where((titanic_train.Age.isnull()) & (titanic_train.Title == 'Master'), 5,\

                               np.where((titanic_train.Age.isnull()) & (titanic_train.Title=='Miss'),22,\

                                        np.where((titanic_train.Age.isnull()) & (titanic_train.Title=='Mr'),32,\

                                                 np.where((titanic_train.Age.isnull()) & (titanic_train.Title=='Mrs'),36,\

                                                          np.where((titanic_train.Age.isnull()) & (titanic_train.Title=='Other'),47,\

                                                                   np.where((titanic_train.Age.isnull()) & (titanic_train.Title=='Dr'),42,titanic_train.Age))))))
titanic_train.isnull().sum()/titanic_train.shape[0] *100
titanic_train.shape
col_list = list(titanic_train.columns)

for col in col_list:

    print(titanic_train[col].value_counts())

    print("-----------------------------")
titanic_train = titanic_train.drop("Ticket", axis=1)

titanic_train.head()
dummy_1 = pd.get_dummies(titanic_train["Embarked"], prefix= "Embarked", drop_first=True)

dummy_2 = pd.get_dummies(titanic_train["Sex"], drop_first=False)

Pclass = pd.get_dummies(titanic_train["Pclass"], prefix= "Pclass")

siblings = pd.get_dummies(titanic_train["SibSp"], prefix= "SibSp")

Parch = pd.get_dummies(titanic_train["Parch"], prefix= "Parch")

cabin = pd.get_dummies(titanic_train["Cabin"], prefix= "Cabin")

Title = pd.get_dummies(titanic_train["Title"], prefix= "Title")
titanic_train = pd.concat([titanic_train, dummy_1, dummy_2, Pclass, siblings, Parch], axis=1)

titanic_train = titanic_train.drop(["Embarked","PassengerId", "Sex", "Pclass", "SibSp", "Parch", "Cabin", "Title"], axis=1)

titanic_train.head()
scalar = MinMaxScaler()

scale_var = ["Age", "Fare"]

titanic_train[scale_var] = scalar.fit_transform(titanic_train[scale_var])

titanic_train.head()
y_train = titanic_train.pop("Survived")

y_train.head()
X_train = titanic_train

X_train.head()
titanic_train.shape
plt.figure(figsize=(20, 15))

sns.heatmap(X_train.corr(), annot=True)

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



rfc = RandomForestClassifier()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 200),

    'n_estimators': [100,200, 300], 

    'max_features': [5, 10]

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
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=10,

                             min_samples_leaf=100, 

                             min_samples_split=200,

                             max_features=10,

                             n_estimators=100)
# fit

rfc.fit(X_train,y_train)
titanic_test = pd.read_csv(r"/kaggle/input/titanic/test.csv")

titanic_test.head()
servived_test = pd.read_csv(r"/kaggle/input/titanic/gender_submission.csv")

servived_test.head()
titanic_test = pd.merge(titanic_test, servived_test, how='inner', on='PassengerId')

titanic_test.head()



titanic_test.Cabin = titanic_test.Cabin.fillna("Unknown_Cabin")

titanic_test.Cabin = titanic_test.Cabin.str[0]

titanic_test.Cabin.value_counts()



round(titanic_test.isnull().sum()/titanic_test.shape[0]*100, 2)

titanic_test['Title'] = titanic_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

titanic_test = titanic_test.drop('Name',axis=1)



titanic_test.head()

#let's replace a few titles -> "other" and fix a few titles

titanic_test['Title'] = np.where((titanic_test.Title=='Capt') | (titanic_test.Title=='Countess') | \

                                  (titanic_test.Title=='Don') | (titanic_test.Title=='Dona')| (titanic_test.Title=='Jonkheer') \

                                  | (titanic_test.Title=='Lady') | (titanic_test.Title=='Sir') | (titanic_test.Title=='Major') | \

                                  (titanic_test.Title=='Rev') | (titanic_test.Title=='Col'),'Other',titanic_test.Title)



titanic_test['Title'] = titanic_test['Title'].replace('Ms','Miss')

titanic_test['Title'] = titanic_test['Title'].replace('Mlle','Miss')

titanic_test['Title'] = titanic_test['Title'].replace('Mme','Mrs')



titanic_test.groupby('Title').Age.mean()



titanic_test["Age"] = np.where((titanic_test.Age.isnull()) & (titanic_test.Title == 'Master'), 7,\

                               np.where((titanic_test.Age.isnull()) & (titanic_test.Title=='Miss'),22,\

                                        np.where((titanic_test.Age.isnull()) & (titanic_test.Title=='Mr'),32,\

                                                 np.where((titanic_test.Age.isnull()) & (titanic_test.Title=='Mrs'),39,\

                                                          np.where((titanic_test.Age.isnull()) & (titanic_test.Title=='Other'),42,\

                                                                   np.where((titanic_test.Age.isnull()) & (titanic_test.Title=='Dr'),53,titanic_test.Age))))))



round(titanic_test.isnull().sum()/titanic_test.shape[0]*100, 2)
titanic_test.Fare = titanic_test.Fare.fillna(0)



round(titanic_test.isnull().sum()/titanic_test.shape[0]*100, 2)
PassengerId = titanic_test.PassengerId



dummy_1 = pd.get_dummies(titanic_test["Embarked"], prefix= "Embarked", drop_first=True)

dummy_2 = pd.get_dummies(titanic_test["Sex"], drop_first=False)

Pclass = pd.get_dummies(titanic_test["Pclass"], prefix= "Pclass")

siblings = pd.get_dummies(titanic_test["SibSp"], prefix= "SibSp")

Parch = pd.get_dummies(titanic_test["Parch"], prefix= "Parch")

cabin = pd.get_dummies(titanic_test["Cabin"], prefix= "Cabin")

Title = pd.get_dummies(titanic_test["Title"], prefix= "Title")



titanic_test = pd.concat([titanic_test, dummy_1, dummy_2, Pclass, siblings, Parch], axis=1)

titanic_test = titanic_test.drop(["Embarked","PassengerId", "Sex", "Pclass", "SibSp", "Parch", "Cabin", "Title"], axis=1)

titanic_test.head()



titanic_test = titanic_test.drop(["Ticket"], axis=1)



test_scale_var = ["Age", "Fare"]

titanic_test[test_scale_var] = scalar.transform(titanic_test[test_scale_var])

titanic_test = titanic_test.drop(['Survived', 'Parch_9'], axis=1)
titanic_test.shape
# predict

predictions = rfc.predict(titanic_test)
predictions
precision_score(servived_test.Survived, predictions)
recall_score(servived_test.Survived, predictions)
y_test_pred_final = pd.DataFrame({"PassengerId":PassengerId ,"Survived":predictions})

y_test_pred_final.head()