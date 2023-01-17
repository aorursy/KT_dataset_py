# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)



from sklearn.model_selection import train_test_split, GridSearchCV
# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

import seaborn as sns
import pandas as pd
import numpy as np
## train = pd.read_csv("C:/Users/DELL/Desktop/payton klasor/lab/train.csv")
# Read train and test data with pd.read_csv():

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# copy data in order to avoid any change in the original:

train = train_data.copy()

test = test_data.copy()
train.head()
test.head()
def targetSummaryWithCat(Parch, train, number_of_classes = 10):

   

    import pandas as pd

    target_name = train[Pclass].name



    for var in df:

        if var != Parch:

            if len(list(df[var].unique())) <= number_of_classes:

                    print(pd.DataFrame({"Pclass": df.groupby(var)[Pclass].mean()}), end = "\n\n\n")
train.groupby("Parch").aggregate([min, np.median, max])
train.isnull().sum()
test.isnull().sum()
g = sns.factorplot(y="Age",x="Sex",data=train,kind="box")

g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=train,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=train,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=train,kind="box")
train["Age"].fillna(train["Age"].median(), inplace = True)
test["Age"].fillna(test["Age"].median(), inplace = True)
full_data = [train, test]
train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Create CabinBool variable which states if someone has a Cabin data or not:



train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
train.head().T
train.dtypes
train.isnull().sum()
test.isnull().sum()
train["Age"].fillna(train["Age"].median(), inplace = True)
train["Age"].fillna(train["Age"].median(), inplace = True)
train.isnull().sum()
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Embarked'].value_counts()
sns.catplot(x = "Sex", y = "Age", hue= "Survived",data = train);
sns.barplot(x = 'Pclass', y = 'Survived', data = train);
sns.barplot(x = 'SibSp', y = 'Survived', data = train);
sns.barplot(x = 'Parch', y = 'Survived', data = train);
sns.barplot(x = 'Sex', y = 'Survived', data = train);
sns.catplot(y = "Age", kind = "violin", data = train);
sns.catplot(x= "Survived", y = "Sex", kind = "violin", data = train);
sns.catplot(x= "Survived", y = "Age", kind = "violin", data = train);
sns.lmplot(x = "Survived", y = "Age", data = train);
sns.lmplot(x = "Survived", y = "Age", hue = "Sex", data = train);
sns.lmplot(x = "Survived", y = "Age", hue = "Sex", col = "Pclass", data = train);
sns.pairplot(train);
train.head().T
import seaborn as sns

sns.jointplot(x="Survived", y = "Fare", data = train, kind = "reg");
sns.distplot(train.Fare);
train["Survived"].value_counts()
train["Age"].value_counts()
train["Fare"].describe()
train.groupby("Survived").aggregate([min, np.median, max])
Q1 = train['Fare'].quantile(0.25)

Q3 = train['Fare'].quantile(0.75)

IQR = Q3 - Q1



lower_limit = Q1- 1.5*IQR

lower_limit



upper_limit = Q3 + 1.5*IQR

upper_limit
# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 

train['Fare'] = train['Fare'].replace(512.3292, 200)

test['Fare'] = test['Fare'].replace(512.3292, 200)
test["Fare"].fillna(test["Fare"].median(), inplace = True)

train["Fare"].fillna(train["Fare"].median(), inplace = True)
# Map Fare values into groups of numerical values:

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
# Drop Fare values:

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
train.sort_values("FareBand", ascending=False).head()
train.isnull().sum()
train.groupby("Survived").aggregate([min, np.median, max])
train[train.notnull().all(axis=1)].T
sns.distplot(train.Age, kde = False);
sns.distplot(train.Pclass);
test.head().T
test.isnull().sum()
train.isnull().sum()
test.isnull().sum()
# Map each Embarked value to a numerical value:



embarked_mapping = {"S": 1, "C": 2, "Q": 3}



train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)
train.head().T
test.head().T
train.isnull().sum()
test.isnull().sum()
test.head().T
test.isnull().sum()
train.isnull().sum()
# We can drop the Ticket feature since it is unlikely to have useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)

train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# Convert Sex values into 1-0:



from sklearn import preprocessing



lbe = preprocessing.LabelEncoder()

train["Sex"] = lbe.fit_transform(train["Sex"])

test["Sex"] = lbe.fit_transform(test["Sex"])
train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.head().T
train[["Title","Survived"]].groupby("Title").count()
# Map each of the title groups to a numerical value



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}



train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
# Map each Age value to a numerical value:

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
#dropping the Age feature for now, might change:

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
train.head()
test.head().T
def targetSummaryWithCat(target_name, df, number_of_classes = 10):

   

    import pandas as pd

    target_name = train[Pclass].name



    for var in df:

        if var != target_name:

            if len(list(df[var].unique())) <= number_of_classes:

                    print(pd.DataFrame({"TARGET_MEAN": df.groupby(var)[target_name].mean()}), end = "\n\n\n")
import seaborn as sns

sns.jointplot(x="Survived", y = "AgeGroup", data = train, kind = "reg");
# Convert Title and Embarked into dummy variables:



train = pd.get_dummies(train, columns = ["Title"])

train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
train.head()
test = pd.get_dummies(test, columns = ["Title"])

test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
# Create categorical values for Pclass:

train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
test["Pclass"] = test["Pclass"].astype("category")

test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
train.isnull().sum()
test.isnull().sum()
train.head()
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
x_train.shape
x_test.shape
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
xgb_params = {

        'n_estimators': [200, 500],

        'subsample': [0.6, 1.0],

        'max_depth': [2,5,8],

        'learning_rate': [0.1,0.01,0.02],

        "min_samples_split": [2,5,10]}
xgb = GradientBoostingClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(x_train, y_train)
xgb_cv_model.best_params_
xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 

                    max_depth = xgb_cv_model.best_params_["max_depth"],

                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],

                    n_estimators = xgb_cv_model.best_params_["n_estimators"],

                    subsample = xgb_cv_model.best_params_["subsample"])
xgb_tuned =  xgb.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
test.head()
test
train.isnull().sum()
test.isnull().sum()
ids = test['PassengerId']
predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
output.head()