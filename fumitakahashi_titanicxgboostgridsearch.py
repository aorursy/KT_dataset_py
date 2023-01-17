import pandas as pd

import numpy as np

import missingno as msno



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_row = pd.read_csv('../input/titanic/train.csv')

test_row = pd.read_csv('../input/titanic/test.csv')
train_row.head(10)
train_row.describe()
train_row.isnull().sum()
test_row.head(10)
test_row.describe()
test_row.isnull().sum()
df_row = pd.concat([train_row,test_row]).sort_values(by = 'PassengerId')

df = df_row.copy()
msno.matrix(df)
msno.bar(df)
df.drop(columns = ["Cabin"], inplace = True)



df["Fare"].fillna(df["Fare"].median(), inplace = True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace = True)



df.isnull().sum()
surv_dict = {0:'Survived', 1:'Killed'}

df_age = df.copy()

df_age['Survived'] = df_age['Survived'].map(surv_dict)



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,5))



sns.set(palette='deep')

sns.boxplot(x='Survived', y='Age', data=df_age, ax = ax1)

sns.boxplot(x='Sex', y='Age', data=df_age, ax = ax2)

sns.boxplot(x='Pclass', y='Age', data=df_age, ax = ax3)
df.Embarked.fillna('S', inplace = True)

df_dummies = pd.get_dummies(df['Embarked'], prefix = 'Embarked')

df = pd.concat([df, df_dummies], axis = 1)



df["is_male"]=(df["Sex"]=="male").astype(int)

df.drop(columns=["Sex"],inplace=True)



emb_dict={"S":0,"C":1,"Q":2}

df["Embarked"]=df["Embarked"].map(lambda x: emb_dict[x])

df.drop(columns=["Ticket"],inplace=True)
df.head(10)
df["NameTitle"]=df["Name"].str.extract('([A-Za-z]+)\.', expand=True)

df["NameLength"]=df["Name"].map(lambda x: len(x))
df["NameTitle"].value_counts()
Title_dict = {

    "Mr": "Mr",

    "Ms": "Mrs",

    "Mrs":"Mrs",

    "Mme":"Mrs",

    "Miss": "Miss",

    "Mlle":"Miss",

    "Master": "Master",

    "Lady":"Royalty",

    "Sir":"Royalty",

    "Don": "Royalty",

    "the Countess": "Royalty",

    "Jonkheer":"Royalty",

    "Capt": "Officer",

    "Col":"Officer",

    "Major":"Officer",

    "Dr":"Officer",

    "Rev":"Officer"

}

df["NameTitle"] = df["Name"].map(lambda name: name.split(',')[1].split('.')[0].strip())

df["NameTitle"] = df.NameTitle.map(Title_dict)
df["NameTitle"].value_counts()
df.head(10)
grouped = df.groupby(['is_male','Pclass','NameTitle'])

grouped_median = grouped.median()

grouped_median = grouped_median.reset_index()[['is_male','Pclass','NameTitle', 'Age']]
grouped_median
def fill_age(row):

    condition = (

        (grouped_median["is_male"]==row["is_male"])&

        (grouped_median["NameTitle"]==row["NameTitle"])&

        (grouped_median["Pclass"]==row["Pclass"])

    )

    if np.isnan(grouped_median[condition]["Age"].values[0]):

        print('true')

        condition = (

            (grouped_median["is_male"]==row["is_male"])&

            (grouped_median["Pclass"]==row["Pclass"])

        )

    return grouped_median[condition]["Age"].values[0]





def process_age():

    global df

    df["Age"] = df.apply(lambda row: fill_age(row) if np.isnan(row["Age"]) else row["Age"], axis = 1)

    return df



df = process_age()


df.drop("Name", axis = 1, inplace = True)



titles_dummies = pd.get_dummies(df["NameTitle"], prefix = "NameTitle")

df = pd.concat([df, titles_dummies], axis = 1)



df.drop("NameTitle", axis = 1, inplace = True)
df.dtypes
import matplotlib.gridspec as gridspec



plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df.drop(columns=["Survived"])):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df["Survived"]==0], bins=50, color="steelblue", kde = False)

    sns.distplot(df[cn][df["Survived"]==1], bins=50, color="orangered", kde = False)

    ax.legend(['Survived', 'Killed'])

    ax.set_xlabel('')

    ax.set_title('Feature: ' + str(cn))

plt.show()


colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Correlation of features', y=1.05, size=15)

sns.heatmap(df[df["PassengerId"]<892].astype(float).corr(), linewidths=0.1,cmap=colormap, vmax=1.0, square=True, annot=True)
cori = []

for col in df.columns:

    print(col,'     ', df[col].dtypes)

    cori.append([col, df[col].corr(df['Survived'])])

df_cor = pd.DataFrame(sorted(cori, key = lambda x: abs(x[1]), reverse = True), columns = ['feature', 'corr'])
plt.figure(figsize=(10, 7))

plt.barh(df_cor['feature'], df_cor['corr'])

plt.title("Correlation with Target")

plt.show()


# 16才以下なら1

age_labels  = [0,1,2]

df['Age_bin'] = pd.cut(df['Age'],[0, 16, 58, 80] , labels=age_labels)





#運賃が40以下なら1

df["is_cheap"]=0

df.loc[df["Fare"]<40,"is_cheap"]=1



df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

df['single'] = 0

df.loc[df['FamilySize'] == 1, 'single'] = 1
df.head(10)
train = df[df["PassengerId"] < 892]

test = df[df["PassengerId"] >= 892].drop(columns="Survived")
train.shape
test.shape
import numpy as np



train_X = np.array(train[train["PassengerId"]<=600].drop(columns=["Survived"]))

train_y = np.array(train[train["PassengerId"]<=600]["Survived"])



test_X = np.array(train[train["PassengerId"]>600].drop(columns=["Survived"]))

test_y = np.array(train[train["PassengerId"]>600]["Survived"])

from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier()

dc.fit(train_X, train_y)

dc_pred_y = dc.predict(test_X)
import sklearn.tree as tree

import graphviz

dt_feature_names = list(df.drop('Survived', inplace = False, axis = 1).columns)

dt_target_names = [str(s) for s in df['Survived'].unique()]

dot_data = tree.export_graphviz(dc, feature_names=dt_feature_names, class_names=dt_target_names, filled=True, out_file=None)  

graph = graphviz.Source(dot_data)  

graph
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=100,   max_depth=10, n_jobs = -1, random_state=100)
rfc.fit(train_X,train_y)

pred_y = rfc.predict(test_X)
from sklearn.metrics import accuracy_score



accuracy_score(pred_y,test_y)
from xgboost import XGBClassifier



xgc = XGBClassifier(n_estimators=100, max_depth=10, random_state=100, eta=0.5)



xgc.fit(train_X,train_y)

pred_y = xgc.predict(test_X)

accuracy_score(pred_y,test_y)
from sklearn.model_selection import cross_val_score

def cross_val(rfc):

    scores = cross_val_score(rfc, train_X, train_y, cv=10, scoring = "accuracy")

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard Deviation:", scores.std())
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

from sklearn.model_selection import GridSearchCV



# grid_param = {

#     'num_boost_round': [100, 250, 500],

#     'eta': [0.05, 0.1, 0.3],

#     'max_depth': [6, 9, 12],

#     'subsample': [0.9, 1.0],

#     'colsample_bytree': [0.9, 1.0],

    

#     }



grid_param = {

     'num_boost_round': [100],

     'eta': [0.05],

     'max_depth': [10],

     'subsample': [1.0],

     'colsample_bytree': [1.0],

     

    

    }





Xgb = XGBClassifier()

cv = GridSearchCV(Xgb, grid_param, cv = 5, scoring='f1',n_jobs =-1,verbose=True)

cv.fit(train_X,train_y)

print(cv.best_params_, cv.best_score_)
Xgb2 = XGBClassifier(**cv.best_params_)

Xgb2.fit(train_X,train_y)

pred_y = Xgb2.predict(test_X)



accuracy_score(pred_y,test_y)
train_X = np.array(train[train["PassengerId"]<=600].drop(columns=["Survived"]))

train_y = np.array(train[train["PassengerId"]<=600]["Survived"])



test_X = np.array(train[train["PassengerId"]>600].drop(columns=["Survived"]))

test_y = np.array(train[train["PassengerId"]>600]["Survived"])





train_X = np.array(train.drop(columns=["Survived"]))

train_y = np.array(train["Survived"])



test_X = np.array(test)
Xgb2.fit(train_X,train_y)

pred_y = Xgb2.predict(test_X)



ans = test.loc[:,["PassengerId"]]

ans["Survived"] = pd.Series(pred_y).astype(int)

ans.to_csv("submission_final.csv",index=False)