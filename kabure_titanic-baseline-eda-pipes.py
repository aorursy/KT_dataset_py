#This librarys is to work with matrices

import pandas as pd 

# This librarys is to work with vectors

import numpy as np

# This library is to create some graphics algorithmn

import seaborn as sns

# to render the graphs

import matplotlib.pyplot as plt

# import module to set some ploting parameters

from matplotlib import rcParams

# Library to work with Regular Expressions

import re

import gc



from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



from scipy import stats



# This function makes the plot directly on browser

%matplotlib inline



# Seting a universal figure size 

rcParams['figure.figsize'] = 12,5
# Importing train dataset

df_train = pd.read_csv("../input/titanic/train.csv")



# Importing test dataset

df_test = pd.read_csv("../input/titanic/test.csv")



submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
resumetable(df_train)
resumetable(df_test)
df_train['Survived'].replace({0:'No', 1:'Yes'}, inplace=True)
total = len(df_train)

plt.figure(figsize=(12,7))

#plt.subplot(121)

g = sns.countplot(x='Survived', data=df_train, color='green')

g.set_title(f"Passengers alive or died Distribution \nTotal Passengers: {total}", 

            fontsize=22)

g.set_xlabel("Passenger Survived?", fontsize=18)

g.set_ylabel('Count', fontsize=18)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

g.set_ylim(0, total *.70)



plt.show()
#First I will look my distribuition without NaN's

#I will create a df to look distribuition 

age_high_zero_died = df_train[(df_train["Age"] > 0) & 

                              (df_train["Survived"] == 'No')]

age_high_zero_surv = df_train[(df_train["Age"] > 0) & 

                              (df_train["Survived"] == 'Yes')]



#figure size

plt.figure(figsize=(16,5))



plt.subplot(121)

plt.suptitle('Age Distributions', fontsize=22)

sns.distplot(df_train[(df_train["Age"] > 0)]["Age"], bins=24)

plt.title("Distribuition of Age",fontsize=20)

plt.xlabel("Age Range",fontsize=15)

plt.ylabel("Probability",fontsize=15)



plt.subplot(122)



sns.distplot(age_high_zero_surv["Age"], bins=24, color='r', label='Survived')

sns.distplot(age_high_zero_died["Age"], bins=24, color='blue', label='Not Survived')

plt.title("Distribution of Age by Target",fontsize=20)

plt.xlabel("Age",fontsize=15)

plt.ylabel("Probability",fontsize=15)

plt.legend()





plt.show()
def plot_categoricals(df, col=None, cont='Age', binary=None, dodge=True):

    tmp = pd.crosstab(df[col], df[binary], normalize='index') * 100

    tmp = tmp.reset_index()



    plt.figure(figsize=(16,12))



    plt.subplot(221)

    g= sns.countplot(x=col, data=df, order=list(tmp[col].values) , color='green')

    g.set_title(f'{col} Distribuition', 

                fontsize=20)

    g.set_xlabel(f'{col} Values',fontsize=17)

    g.set_ylabel('Count Distribution', fontsize=17)

    sizes = []

    for p in g.patches:

        height = p.get_height()

        sizes.append(height)

        g.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.0f}'.format(height),

                ha="center", fontsize=15) 

    g.set_ylim(0,max(sizes)*1.15)



    plt.subplot(222)

    g1= sns.countplot(x=col, data=df, order=list(tmp[col].values),

                     hue=binary,palette="hls")

    g1.set_title(f'{col} Distribuition by {binary} ratio %', 

                fontsize=20)

    gt = g1.twinx()

    gt = sns.pointplot(x=col, y='Yes', data=tmp, order=list(tmp[col].values),

                       color='black', legend=False)

    gt.set_ylim(0,tmp['Yes'].max()*1.1)

    gt.set_ylabel("Survived %Ratio", fontsize=16)

    g1.set_ylabel('Count Distribuition',fontsize=17)

    g1.set_xlabel(f'{col} Values', fontsize=17)

    

    sizes = []

    

    for p in g1.patches:

        height = p.get_height()

        sizes.append(height)

        g1.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=10) 

    g1.set_ylim(0,max(sizes)*1.15)



    plt.subplot(212)

    g2= sns.swarmplot(x=col, y=cont, data=df, dodge=dodge, order=list(tmp[col].values),

                     hue="Survived",palette="hls")

    g2.set_title(f'{cont} Distribution by {col} and {binary}', 

                fontsize=20)

    g2.set_ylabel(f'{cont} Distribuition',fontsize=17)

    g2.set_xlabel(f'{col} Values', fontsize=17)





    plt.suptitle(f'{col} Distributions', fontsize=22)

    plt.subplots_adjust(hspace = 0.4, top = 0.90)

    

    plt.show()
plot_categoricals(df_train, col='Sex', cont='Age', binary='Survived')
plot_categoricals(df_train, col='Pclass', cont='Age', binary='Survived')
(round(pd.crosstab(df_train['Survived'], [df_train['Pclass'], df_train['Sex']], 

             normalize='columns' ) * 100,2))
plot_categoricals(df_train, col='Embarked', cont='Age', binary='Survived')

#lets input the NA's with the highest frequency

df_train["Embarked"] = df_train["Embarked"].fillna('S')
(round(pd.crosstab(df_train['Survived'], [df_train['Embarked'], df_train['Pclass']], 

             normalize='columns' ) * 100,2))
(round(pd.crosstab(df_train['Survived'], [df_train['Embarked'], df_train['Sex']], 

             normalize='columns' ) * 100,2))
df_train['Fare'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index()
df_train['Fare_log'] = np.log(df_train['Fare'] + 1)

df_test['Fare_log'] = np.log(df_test['Fare'] + 1)
# Seting the figure size

plt.figure(figsize=(16,10))



# Understanding the Fare Distribuition 

plt.subplot(221)

sns.distplot(df_train["Fare"], bins=50 )

plt.title("Fare Distribuition", fontsize=20)

plt.xlabel("Fare", fontsize=15)

plt.ylabel("Density",fontsize=15)



plt.subplot(222)

sns.distplot(df_train["Fare_log"], bins=50 )

plt.title("Fare LOG Distribuition", fontsize=20)

plt.xlabel("Fare (Log)", fontsize=15)

plt.ylabel("Density",fontsize=15)



plt.subplot(212)

g1 = plt.scatter(range(df_train[df_train.Survived == 'No'].shape[0]),

                 np.sort(df_train[df_train.Survived == 'No']['Fare'].values), 

                 label='No Survived', alpha=.5)

g1 = plt.scatter(range(df_train[df_train.Survived == 'Yes'].shape[0]),

                 np.sort(df_train[df_train.Survived == 'Yes']['Fare'].values), 

                 label='Survived', alpha=.5)

g1= plt.title("Fare ECDF Distribution", fontsize=18)

g1 = plt.xlabel("Index")

g1 = plt.ylabel("Fare Amount", fontsize=15)

g1 = plt.legend()



plt.suptitle('Fare Distributions', fontsize=22)

plt.subplots_adjust(hspace = 0.4, top = 0.90)



plt.show()
def ploting_cat_group(df, col):

    plt.figure(figsize=(14,6))

    tmp = pd.crosstab(df['Survived'], df[col], 

                      values=df['Fare'], aggfunc='mean').unstack(col).reset_index().rename(columns={0:'FareMean'})

    g = sns.barplot(x=col, y='FareMean', hue='Survived', data=tmp)

    g.set_xlabel(f'{col} values', fontsize=18)

    g.set_ylabel('Fare Mean', fontsize=18)

    g.set_title(f"Fare Distribution by {col} ", fontsize=20)

    

    plt.show()
ploting_cat_group(df_train, 'Pclass')
ploting_cat_group(df_train, 'Embarked')
ploting_cat_group(df_train, 'Sex')
plt.figure(figsize=(14,6))

g = sns.scatterplot(x='Age', y='Fare_log', data=df_train, hue='Survived')

g.set_title('Fare Distribution by Age', fontsize= 22)

g.set_xlabel('Age Distribution', fontsize=18)

g.set_ylabel("Fare Log Distribution", fontsize=18)



plt.show()
df_train.groupby(['Survived', 'Pclass'])['Age'].mean().unstack('Survived').reset_index()
df_train['Name'].unique()[:10]
# Extracting the prefix of all Passengers

df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

df_test['Title'] = df_test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))



(df_train['Title'].value_counts(normalize=True) * 100).head(5)

#Now, I will identify the social status of each title



Title_Dictionary = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Mrs",

        "Ms":         "Mrs",

        "Mrs" :       "Mrs",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

}

    

# we map each title to correct category

df_train['Title'] = df_train.Title.map(Title_Dictionary)

df_test['Title'] = df_test.Title.map(Title_Dictionary)
plot_categoricals(df_train, col='Title', cont='Age', binary='Survived')
#Let's group the median age by sex, pclass and title, to have any idea and maybe input in Age NAN's

age_group = df_train.groupby(["Sex","Pclass","Title"])["Age"]



#printing the variabe that we created by median

age_group.median().unstack('Pclass').reset_index()
#inputing the values on Age Na's 

# using the groupby to transform this variables

df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')

df_test.loc[df_train.Age.isnull(), 'Age'] = df_test.groupby(['Sex','Pclass','Title']).Age.transform('median')



# printing the total of nulls in Age Feature

print(df_train["Age"].isnull().sum())

#df_train.Age = df_train.Age.fillna(-0.5)



#creating the intervals that we need to cut each range of ages

interval = (0, 5, 12, 18, 25, 35, 60, 120) 



#Seting the names that we want use to the categorys

cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']



# Applying the pd.cut and using the parameters that we created 

df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=cats)

df_test["Age_cat"] = pd.cut(df_test.Age, interval, labels=cats)



# Printing the new Category

df_train["Age_cat"].unique()
plot_categoricals(df_train, col='Age_cat', cont='Fare', binary='Survived')
plot_categoricals(df_train, col='SibSp', cont='Age', binary='Survived')
plot_categoricals(df_train, col='Parch', cont='Age', binary='Survived', dodge=False)
#Create a new column and sum the Parch + SibSp + 1 that refers the people self

df_train["FSize"] = df_train["Parch"] + df_train["SibSp"] + 1

df_test["FSize"] = df_test["Parch"] + df_test["SibSp"] + 1



family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 

              5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',

              11: 'Large'}



df_train['FSize'] = df_train['FSize'].map(family_map)

df_test['FSize'] = df_test['FSize'].map(family_map)
plot_categoricals(df_train, col='FSize', cont='Fare', binary='Survived', dodge=True)
## I saw this code in another kernel and it is very useful

## Link: https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic

import string



def extract_surname(data):    

    

    families = []

    

    for i in range(len(data)):        

        name = data.iloc[i]



        if '(' in name:

            name_no_bracket = name.split('(')[0] 

        else:

            name_no_bracket = name

            

        family = name_no_bracket.split(',')[0]

        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        

        for c in string.punctuation:

            family = family.replace(c, '').strip()

            

        families.append(family)

            

    return families
df_train['Family'] = extract_surname(df_train['Name'])

df_test['Family'] = extract_surname(df_test['Name'])
df_train['Ticket'].value_counts()[:10]
df_train['Ticket_Frequency'] = df_train.groupby('Ticket')['Ticket'].transform('count')

df_test['Ticket_Frequency'] = df_test.groupby('Ticket')['Ticket'].transform('count')
# Creating a list of families and tickets that are occuring in both training and test set

non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]

non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]
df_train.groupby(['Survived', 'FSize'])['Fare'].mean().unstack('FSize').reset_index()
#Filling the NA's with -0.5

df_train.Fare = df_train.Fare.fillna(-1)

df_test.Fare = df_test.Fare.fillna(-1)

#intervals to categorize

quant = (-1, 0, 12, 30, 80, 100, 200, 600)



#Labels without input values

label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4', 'quart_5', 'quart_6']



#doing the cut in fare and puting in a new column

df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)

df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)
plot_categoricals(df_train, col='Fare_cat', cont='Age', binary='Survived', dodge=False)
# Excellent implementation from: 

# https://www.kaggle.com/franjmartin21/titanic-pipelines-k-fold-validation-hp-tuning



def cabin_extract(df):

    return df['Cabin'].apply(lambda x: str(x)[0] if(pd.notnull(x)) else str('M'))



df_train['Cabin'] = cabin_extract(df_train)

df_test['Cabin'] = cabin_extract(df_test)
plot_categoricals(df_train, col='Cabin', cont='Age', binary='Survived', dodge=True)
pd.crosstab(df_train['Cabin'], df_train['Pclass'])
df_train['Cabin'] = df_train['Cabin'].replace(['A', 'B', 'C'], 'ABC')

df_train['Cabin'] = df_train['Cabin'].replace(['D', 'E'], 'DE')

df_train['Cabin'] = df_train['Cabin'].replace(['F', 'G'], 'FG')

# Passenger in the T deck is changed to A

df_train.loc[df_train['Cabin'] == 'T', 'Cabin'] = 'A'



df_test['Cabin'] = df_test['Cabin'].replace(['A', 'B', 'C'], 'ABC')

df_test['Cabin'] = df_test['Cabin'].replace(['D', 'E'], 'DE')

df_test['Cabin'] = df_test['Cabin'].replace(['F', 'G'], 'FG')

df_test.loc[df_test['Cabin'] == 'T', 'Cabin'] = 'A'
from pandas.api.types import CategoricalDtype 

family_cats = CategoricalDtype(categories=['Alone', 'Small', 'Medium', 'Large'], ordered=True)
df_train.FSize = df_train.FSize.astype(family_cats)

df_test.FSize = df_test.FSize.astype(family_cats)
df_train.Age_cat = df_train.Age_cat.cat.codes

df_train.Fare_cat = df_train.Fare_cat.cat.codes

df_test.Age_cat = df_test.Age_cat.cat.codes

df_test.Fare_cat = df_test.Fare_cat.cat.codes

df_train.FSize = df_train.FSize.cat.codes

df_test.FSize = df_test.FSize.cat.codes
#Now lets drop the variable Fare, Age and ticket that is irrelevant now

df_train.drop([ 'Ticket', 'Name'], axis=1, inplace=True)

df_test.drop(['Ticket', 'Name', ], axis=1, inplace=True)

#df_train.drop(["Fare", 'Ticket', 'Age', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)

#df_test.drop(["Fare", 'Ticket', 'Age', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)
df_test['Survived'] = 'test'

df = pd.concat([df_train, df_test], axis=0, sort=False )
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Family'] = le.fit_transform(df['Family'].astype(str))
df = pd.get_dummies(df, columns=['Sex', 'Cabin', 'Embarked', 'Title'],\

                          prefix=['Sex', "Cabin", 'Emb', 'Title'], drop_first=True)



df_train, df_test = df[df['Survived'] != 'test'], df[df['Survived'] == 'test'].drop('Survived', axis=1)

del df
df_train['Survived'].replace({'Yes':1, 'No':0}, inplace=True)
print(f'Train shape: {df_train.shape}')

print(f'Train shape: {df_test.shape}')
df_train.drop(['Age', 'Fare','Fare_log','Family', 'SibSp', 'Parch'], axis=1, inplace=True)

df_test.drop(['Age', 'Fare','Fare_log','Family', 'SibSp', 'Parch'], axis=1, inplace=True)
X_train = df_train.drop(["Survived","PassengerId"],axis=1)

y_train = df_train["Survived"]



X_test = df_test.drop(["PassengerId"],axis=1)
resumetable(X_train)
#Importing the auxiliar and preprocessing librarys 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline



from sklearn.model_selection import train_test_split, KFold, cross_validate

from sklearn.metrics import accuracy_score



#Models

import warnings

warnings.filterwarnings("ignore")



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
clfs = []

seed = 3



clfs.append(("LogReg", 

             Pipeline([("Scaler", StandardScaler()),

                       ("LogReg", LogisticRegression())])))



clfs.append(("XGBClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("XGB", XGBClassifier())]))) 

clfs.append(("KNN", 

             Pipeline([("Scaler", StandardScaler()),

                       ("KNN", KNeighborsClassifier())]))) 



clfs.append(("DecisionTreeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("DecisionTrees", DecisionTreeClassifier())]))) 



clfs.append(("RandomForestClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RandomForest", RandomForestClassifier(n_estimators=100))]))) 



clfs.append(("GradientBoostingClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingClassifier(n_estimators=100))]))) 



clfs.append(("RidgeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RidgeClassifier", RidgeClassifier())])))



clfs.append(("BaggingRidgeClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("BaggingClassifier", BaggingClassifier())])))



clfs.append(("ExtraTreesClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreesClassifier())])))



#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'accuracy'

n_folds = 7



results, names  = [], [] 



for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, 

                                 cv= 5, scoring=scoring,

                                 n_jobs=-1)    

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())

    print(msg)

    

# boxplot algorithm comparison

fig = plt.figure(figsize=(15,6))

fig.suptitle('Classifier Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names, y=results)

ax.set_xticklabels(names)

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("Accuracy of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)



plt.show()
import time



def objective(params):

    time1 = time.time()

    params = {

        'max_depth': params['max_depth'],

        'max_features': params['max_features'],

        'n_estimators': params['n_estimators'],

        'min_samples_split': params['min_samples_split'],

        'criterion': params['criterion']

    }



    print("\n############## New Run ################")

    print(f"params = {params}")

    FOLDS = 10

    count=1



    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)



    kf = KFold(n_splits=FOLDS, shuffle=False, random_state=42)



    score_mean = 0

    for tr_idx, val_idx in kf.split(X_train, y_train):

        clf = RandomForestClassifier(

            random_state=4, 

            verbose=0,  n_jobs=-1, 

            **params

        )



        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]

        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

        clf.fit(X_tr, y_tr)

        #y_pred_train = clf.predict_proba(X_vl)[:,1]

        #print(y_pred_train)

        score = make_scorer(accuracy_score)(clf, X_vl, y_vl)

        # plt.show()

        score_mean += score

        print(f'{count} CV - score: {round(score, 4)}')

        count += 1

    time2 = time.time() - time1

    print(f"Total Time Run: {round(time2 / 60,2)}")

    gc.collect()

    print(f'Mean ROC_AUC: {score_mean / FOLDS}')

    del X_tr, X_vl, y_tr, y_vl, clf, score

    return -(score_mean / FOLDS)



rf_space = {

    'max_depth': hp.choice('max_depth', range(2,8)),

    'max_features': hp.choice('max_features', range(1,X_train.shape[1])),

    'n_estimators': hp.choice('n_estimators', range(100,500)),

    'min_samples_split': hp.choice('min_samples_split', range(5,35)),

    'criterion': hp.choice('criterion', ["gini", "entropy"])

}
best = fmin(fn=objective,

            space=rf_space,

            algo=tpe.suggest,

            max_evals=40, 

            # trials=trials

           )
best_params = space_eval(rf_space, best)

best_params
clf = RandomForestClassifier(

        **best_params, random_state=4,

        )



clf.fit(X_train, y_train)



y_preds= clf.predict(X_test)



submission['Survived'] = y_preds.astype(int)

submission.to_csv('Titanic_rf_model_pred.csv')
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                  np.unique(y_train),

                                                  y_train)
def objective_logreg(params):

    time1 = time.time()

    params = {

        'tol': params['tol'],

        'C': params['C'],

        'solver': params['solver'],

    }



    print("\n############## New Run ################")

    print(f"params = {params}")

    FOLDS = 10

    count=1



    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)



    kf = KFold(n_splits=FOLDS, shuffle=False, random_state=42)



    score_mean = 0

    for tr_idx, val_idx in kf.split(X_train, y_train):

        clf = LogisticRegression(

            random_state=4,  

            **params

        )



        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]

        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

        clf.fit(X_tr, y_tr)

        score = make_scorer(accuracy_score)(clf, X_vl, y_vl)

        score_mean += score

        print(f'{count} CV - score: {round(score, 4)}')

        count += 1

    time2 = time.time() - time1

    print(f"Total Time Run: {round(time2 / 60,2)}")

    gc.collect()

    print(f'Mean ROC_AUC: {score_mean / FOLDS}')

    del X_tr, X_vl, y_tr, y_vl, clf, score

    return -(score_mean / FOLDS)



space_logreg = {

    'tol' : hp.uniform('tol', 0.00001, 0.001),

    'C' : hp.uniform('C', 0.001, 2),

    'solver' : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),

}
best = fmin(fn=objective_logreg,

            space=space_logreg,

            algo=tpe.suggest,

            max_evals=45, 

            # trials=trials

           )
best_params = space_eval(space_logreg, best)

best_params
clf = LogisticRegression(

        **best_params, random_state=4,

        )



clf.fit(X_train, y_train)



y_preds= clf.predict(X_test)



submission['Survived'] = y_preds.astype(int)

submission.to_csv('Titanic_logreg_model_pred.csv')


