# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

from collections import Counter

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

SEED=2020

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.combine import SMOTETomek

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



## thanks to @Nadezda Demidova  https://www.kaggle.com/demidova/titanic-eda-tutorial-with-seaborn

train.loc[train['PassengerId'] == 631, 'Age'] = 48



# Passengers with wrong number of siblings and parch

train.loc[train['PassengerId'] == 69, ['SibSp', 'Parch']] = [0,0]

test.loc[test['PassengerId'] == 1106, ['SibSp', 'Parch']] = [0,0]

## to reduce the amount of code let's introduce a general frame

full_data = [train, test]

## train dataset overview

train.head()
def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.7 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



#detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train.loc[Outliers_to_drop] # Show the outliers rows

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
## train dataset overview



def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b



basic_details(train)
basic_details(test)
train.info()
full_data = [train, test]



for df in full_data:

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    df.drop(["PassengerId", 'Cabin'], axis=1, inplace=True)

## and we have 2 NaN in train dataser for Embarked feature, so let's impute it with most frequent value 'S'

train["Embarked"] = train["Embarked"].fillna("S")

## the same for test dataset

test.info()
#we have 1 NaN in test dataser for Fare feature, so let's impute it...

test['Fare'].fillna(test["Fare"].median(), inplace=True)
## Has_Cabin feature survival rate

train[["Has_Cabin", "Survived"]].groupby(['Has_Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
## checking for Embarked feature distribution before we'll make it numerical

train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
## Making from categorial Embarked -  numerical feature

for df in full_data:

    df["Embarked"][df["Embarked"] == "S"] = 1

    df["Embarked"][df["Embarked"] == "C"] = 2

    df["Embarked"][df["Embarked"] == "Q"] = 3

    df["Embarked"] = df["Embarked"].astype(int)
## new feature Name length, late we will check it's influence on target

for df in full_data:

    df['Name_length'] = df['Name'].apply(len)



# New Title feature

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for df in full_data:

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['Title'] = df['Title'].map(title_mapping)

    df['Title'] = df['Title'].fillna(0)





##dropping Name feature from both datasets

for df in full_data:

    df.drop(['Name'], axis=1,inplace=True)

# Convert 'Sex' variable to integer form

for df in full_data:

    df["Sex"][df["Sex"] == "male"] = 1

    df["Sex"][df["Sex"] == "female"] = 0

    df["Sex"] = df["Sex"].astype(int)
# New 'FamilySize' feature

for df in full_data:

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

## checking for Survived dependence of FamilySize feature

train[["FamilySize", "Survived"]].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# FamilySize distribution

g = sns.kdeplot(train['FamilySize'][(train["Survived"] == 0) & (train['FamilySize'].notnull())], color="Red", shade = True)

g = sns.kdeplot(train['FamilySize'][(train["Survived"] == 1) & (train['FamilySize'].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel('FamilySize')

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
## making new FamilySize_cat feature based on above mentioned conclusion and let's see its distribution

bins = [0,1,4,11]

labels=[0,1,2]

train['FamilySize_cat'] = pd.cut(train['FamilySize'], bins=bins, labels=labels)

test['FamilySize_cat'] = pd.cut(test['FamilySize'], bins=bins, labels=labels)

train[["FamilySize_cat", "Survived"]].groupby(['FamilySize_cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.xticks([0,12,29,50, 80])



g = sns.kdeplot(train['Name_length'][(train["Survived"] == 0) & (train['Name_length'].notnull())], color="Red", shade = True)

g = sns.kdeplot(train['Name_length'][(train["Survived"] == 1) & (train['Name_length'].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel('Name_length')

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
## let's make new cat feature based on this conclusion:



bins = [11,29,100]

labels=[0,1]

train['Name_length_cat'] = pd.cut(train['Name_length'], bins=bins, labels=labels)

test['Name_length_cat'] = pd.cut(test['Name_length'], bins=bins, labels=labels)

train[["Name_length_cat", "Survived"]].groupby(['Name_length_cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.xticks([0,16,32,60,100])



g = sns.kdeplot(train['Age'][(train["Survived"] == 0) & (train['Age'].notnull())], color="Red", shade = True)

g = sns.kdeplot(train['Age'][(train["Survived"] == 1) & (train['Age'].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel('Age')

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
## new Age_cat feature baset on this conclusions

bins = [0,16,32,60,100]

labels=[0,1,2,3]

train['Age_cat'] = pd.cut(train['Age'], bins=bins, labels=labels)

test['Age_cat'] = pd.cut(test['Age'], bins=bins, labels=labels)

train[["Age_cat", "Survived"]].groupby(['Age_cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for df in full_data:

    df['Ticket'] = df['Ticket'].apply(len)
g = sns.kdeplot(train['Ticket'][(train["Survived"] == 0) & (train['Ticket'].notnull())], color="Red", shade = True)

g = sns.kdeplot(train['Ticket'][(train["Survived"] == 1) & (train['Ticket'].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel('Ticket')

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
for df in full_data:

    df['Ticket_5'] = train['Ticket'].map(lambda x: 1 if x == 5 else 0)

    df['Ticket_6'] = train['Ticket'].map(lambda x: 1 if x == 6 else 0)



train[["Ticket_5", "Survived"]].groupby(['Ticket_5'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Ticket_6", "Survived"]].groupby(['Ticket_6'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.set(rc={'figure.figsize':(20,10)})

plt.xticks([0,8,18,28,75,100,150,200])







g = sns.kdeplot(train['Fare'][(train["Survived"] == 0) & (train['Fare'].notnull())], color="Red", shade = True)

g = sns.kdeplot(train['Fare'][(train["Survived"] == 1) & (train['Fare'].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel('Fare')

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
## making bins based on picture info for new feature Fare_cat

bins = [-1,8,18,28,520]

labels=[0,1,2,3]

train['Fare_cat'] = pd.cut(train['Fare'], bins=bins, labels=labels)

test['Fare_cat'] = pd.cut(test['Fare'], bins=bins, labels=labels)

train[["Fare_cat", "Survived"]].groupby(['Fare_cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)
features_to_drop = ['Age','Fare','SibSp', 'Parch', 'Ticket', 'Name_length', 'FamilySize']

for df in full_data:

    df["FamilySize_cat"] = df["FamilySize_cat"].astype(int)

    df["Age_cat"] = df["Age_cat"].astype(int)

    df["Fare_cat"] = df["Fare_cat"].astype(int)

    df["Name_length_cat"] = df["Name_length_cat"].astype(int)

    df['Age_scaled'] = ss.fit_transform(df['Age'].values.reshape(-1,1)) ## new feature based on Age

    df['Fare_scaled'] = ss.fit_transform(df['Fare'].values.reshape(-1,1)) ## new feature based on Fare

    df['Name_length_log'] = np.log1p(df['Name_length']) # new normalized feature on base of Name_lenght 

    df.drop(features_to_drop, axis=1, inplace=True) ## drop unneccessary features
## train dataset before modelling

train.head(10)
## corrmatrix

def spearman(frame, features):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['Survived'], 'spearman') for f in features]

    spr = spr.sort_values('spearman')

    plt.figure(figsize=(6, 0.25*len(features)))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    

features = train.drop(['Survived'], axis=1).columns

spearman(train, features)
# ## adding new featrures

# def des_stat_feat(df):

#     df = pd.DataFrame(df)

#     dcol= [c for c in df.columns if df[c].nunique()>=3]

#     d_median = df[dcol].median(axis=0)

#     d_mean = df[dcol].mean(axis=0)

#     q1 = df[dcol].apply(np.float32).quantile(0.25)

#     q3 = df[dcol].apply(np.float32).quantile(0.75)

    

#     #Add mean and median column to data set having more then 3 categories

#     for c in dcol:

#         df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)

#         df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)

#         df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)

#         df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)

#     return df



# train = des_stat_feat(train) 
x = train.drop('Survived', axis=1)

y = train.Survived
# sm = SMOTE(random_state=SEED)

# smk=SMOTETomek(random_state=SEED)

# rus = RandomUnderSampler(random_state=SEED)

ros = RandomOverSampler(random_state=SEED)

# adasyn = ADASYN(random_state=SEED)



x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=SEED)

x_train, y_train= ros.fit_resample(x_train, y_train)

# x, y= ros.fit_resample(x, y)
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

acc_log_train = round(logreg.score(x_train, y_train)*100,2) 

acc_log_test = round(logreg.score(x_valid,y_valid)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
scores = cross_val_score(logreg, x, y, cv=10, scoring = "f1")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())


rf = RandomForestClassifier(max_depth=9, random_state=SEED,min_samples_split=5)

rf.fit(x_train, y_train)

acc_rf_train = round(rf.score(x_train, y_train)*100,2) 

acc_rf_test = round(rf.score(x_valid,y_valid)*100,2)

print("Training Accuracy: % {}".format(acc_rf_train))

print("Testing Accuracy: % {}".format(acc_rf_test))
scores = cross_val_score(rf, x, y, cv=10, scoring = 'f1')

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
import xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=610,

                        min_child_weight=2,

                        max_depth=5,

                        objective="binary:hinge",

                        learning_rate=.1, 

                        subsample=.8, 

                        colsample_bytree=.8,

                        gamma=0,

                        seed=29,

                        reg_alpha=0.05,

#                         reg_lambda=1,

                        nthread=4)

clf.fit(x_train, y_train)

acc_xgb_train = round(clf.score(x_train, y_train)*100,2) 

acc_xgb_test = round(clf.score(x_valid,y_valid)*100,2)

print("Training Accuracy: % {}".format(acc_xgb_train))

print("Testing Accuracy: % {}".format(acc_xgb_test))
scores = cross_val_score(clf, x, y, cv=10, scoring = 'f1')

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
fig,ax = plt.subplots(figsize=(15,20))

xgb.plot_importance(clf,ax=ax,max_num_features=20,height=0.8,color='g')

# plt.tight_layout()

plt.show()
xgb.to_graphviz(clf)
