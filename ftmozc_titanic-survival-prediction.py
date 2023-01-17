# 1.Introduction

# 2.Data Understanding

#   2.1 İmporting Libraries and Loading Data

#   2.2 Feature Analysis

#   2.3 Visualization

# 3.Data Preparation

#   3.1 Deleting Unnecessary Variables

#   3.2 Outlier Treatment

#   3.3 Missing Value Treatment

#      3.3.1 Age

#      3.3.2 Embarked

#      3.3.3 Fare

#      3.3.4 Cabin

#   3.4 Variables Transformation

#      3.4.1 Embarked

#      3.4.2 Sex

#      3.4.3 Name-Title

#      3.4.4 AgeGroup

#      3.4.5 Fare

#   3.5 Feature Engineering

#      3.5.1 Family Size

#      3.5.2 Embarked-Title

#      3.5.3 Pclass

# 4.Modelling

#   4.1 Spliting The Train Data

#   4.2 Model Tuning

#   4.3 Deployment

 
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

# Read train and test data with pd.read_csv():

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train = train_data.copy()

test = test_data.copy()
# I want to combine test and train to make operations easier



df=pd.concat([train,test],ignore_index=True)
df
df.info()
df.describe().T
df["Pclass"].value_counts()
df["Sex"].value_counts()
df["SibSp"].value_counts()
df["Parch"].value_counts()
df["Ticket"].value_counts()
df["Cabin"].value_counts()
df["Embarked"].value_counts()
age = pd.cut(train["Age"], [0, 18,35,50,90])

age.head(10)
train.pivot_table("Survived", ["Sex", age], "Pclass")
sns.barplot(x="Pclass",y="Survived",data=df);
df.groupby("Pclass")[["Survived"]].mean()
sns.barplot(x="Sex",y="Survived",data=df);
df.groupby("Sex")[["Survived"]].mean()
sns.factorplot('Pclass','Survived',hue='Sex',data=df)

plt.show()
sns.barplot(x="SibSp",y="Survived",data=df);
df.groupby("SibSp")[["Survived"]].mean()
sns.barplot(x="Parch",y="Survived",data=df);
df.groupby("SibSp")[["Survived"]].mean()
sns.barplot(x=age ,y="Survived", data=train);
train.groupby(age)[["Survived"]].count()
sns.kdeplot(df.Age, shade = True);
# We can drop the Ticket feature since it is unlikely to have useful information



df = df.drop(['Ticket'], axis = 1)



df.head()
df.describe().T
# It looks like there is a problem in Fare max data. Visualize with boxplot.

sns.boxplot(x = df['Fare']);
Q1 = df['Fare'].quantile(0.25)

Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1



lower_limit = Q1- 1.5*IQR

lower_limit



upper_limit = Q3 + 1.5*IQR

upper_limit
# observations with Fare data higher than the upper limit:



df['Fare'] > (upper_limit)
df.sort_values("Fare", ascending=False).head()
# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 



df['Fare'] = df['Fare'].replace(512.3292, 300)

df.sort_values("Fare", ascending=False).head()

df.isnull().sum()
# Missing value of "Survived" is in test data.So it is not important.
# We can fill the age variable with median.



df["Age"] = df["Age"].fillna(df["Age"].median())
df.isnull().sum()
df["Embarked"].value_counts()
# Fill NA with the most frequent value:



df["Embarked"] = df["Embarked"].fillna("S")
df.isnull().sum()
df[df["Fare"].isnull()]
df[["Pclass","Fare"]].groupby("Pclass").mean()
# We fill the missing value in fare with the mean of class where Pclass is 3.



df["Fare"] = df["Fare"].fillna(13)
df["Fare"].isnull().sum()
# Create CabinBool variable which states if someone has a Cabin data or not:



df["CabinBool"] = (df["Cabin"].notnull().astype('int'))



df = df.drop(['Cabin'], axis = 1)



df.head()
df.isnull().sum()
# Map each Embarked value to a numerical value:



embarked_mapping = {"S": 1, "C": 2, "Q": 3}



df["Embarked"] = df["Embarked"].map(embarked_mapping)

df.head()
# Convert Sex values into 1-0:



df["Sex"]=df["Sex"].map(lambda x:1 if x== "male" else 0)
df.head()
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()
df = df.drop(['Name'], axis = 1)
df['Title'] = df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')
df.head()
df[["Title","PassengerId"]].groupby("Title").count()
df[['Title', 'Survived']].groupby(['Title'], as_index=False).agg({"count","mean"})
# Map each of the title groups to a numerical value



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}



df['Title'] = df['Title'].map(title_mapping)

df.head()
# Map Age values into groups of numerical values:

bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = [1, 2, 3, 4, 5, 6, 7]

df['AgeGroup'] = pd.cut(df["Age"], bins, labels = mylabels)

df["AgeGroup"] = df["AgeGroup"].astype("int")
df.head()
#dropping the Age feature for now, might change:

df = df.drop(['Age'], axis = 1)
df.head()
# Map Fare values into groups of numerical values:

df["FareBand"]=pd.qcut(df["Fare"],4,labels=[1,2,3,4])

df["FareBand"] = df["FareBand"].astype("int")
df=df.drop(["Fare"],axis=1)
df.head()
df["FamilySize"]=df["SibSp"]+df["Parch"]+1

df.head()
# Create new feature of family size:



df['Single'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

df['SmallFam'] = df['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

df['MedFam'] = df['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df['LargeFam'] = df['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
df.head()
# Convert Title and Embarked into dummy variables:



df = pd.get_dummies(df, columns = ["Title"])

df = pd.get_dummies(df, columns = ["Embarked"], prefix="Em")
df.head()
# Create categorical values for Pclass:

df["Pclass"] = df["Pclass"].astype("category")

df= pd.get_dummies(df, columns = ["Pclass"],prefix="Pc")
df.head()
train=df[(df.PassengerId <892 )].astype(int)

test=df[(df.PassengerId >891 )]

train
test
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
def base_models(df):

    

    

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score

    

    Y = df["Survived"]

    X = df.drop(["Survived","PassengerId"], axis=1)

    

    X_train, X_test, y_train, y_test = train_test_split(X, Y, 

                                                    test_size = 0.20, 

                                                    random_state = 42)

    

    #results = []

    

    names = ["LogisticRegression","GaussianNB","KNN","LinearSVC","SVC",

             "CART","RF","GBM","XGBoost","LightGBM","CatBoost"]

    

    

    classifiers = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),LinearSVC(),SVC(),

                  DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier(),

                  XGBClassifier(), LGBMClassifier(), CatBoostClassifier(verbose = False)]

    

    

    for name, clf in zip(names, classifiers):



        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        msg = "%s: %f" % (name, acc)

        print(msg)
base_models(train)
#As showed, xgboost gives the best results.therefore ı want to choise xgboost as a model.
Y = train["Survived"]

X = train.drop(["Survived","PassengerId"], axis=1)

    

X_train, X_test, y_train, y_test = train_test_split(X, Y, 

                                                    test_size = 0.20, 

                                                    random_state = 42)

    

xgb_params = {

        'n_estimators': [100, 500, 1000],

        'subsample': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5,6],

        'learning_rate': [0.1,0.01,0.02,0.05],

        "min_samples_split": [2,5,10]}

xgb = XGBClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)

xgb_cv_model.fit(X_train, y_train)
xgb_cv_model.best_params_
from sklearn.metrics import accuracy_score

xgb = XGBClassifier(learning_rate = 0.05, 

                    max_depth = 3,

                    min_samples_split = 2,

                    n_estimators = 100,

                    subsample = 0.6)

xgb_tuned =  xgb.fit(X_train,y_train)

y_pred = xgb_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
feature_imp = pd.Series(xgb_tuned.feature_importances_,

                        index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Variable İmportance Scores')

plt.ylabel('Variables')

plt.title("Variable Significance Levels")

plt.show()
test=test.drop("Survived", axis=1)

test=test.astype(int)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
output.head()