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
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.preprocessing as skpe

import sklearn.metrics as sklm

import sklearn.model_selection as ms

import sklearn.tree as tree

import sklearn.ensemble as ensemble

import sklearn.linear_model as lm

import scipy.stats as stats

import numpy.random as nr

import sklearn.neighbors as neighbors

import xgboost as xgb

import lightgbm as lgb
path1 = "../input/titanic/train.csv"

train = pd.read_csv(path1)

train.head()
path2 = "../input/titanic/test.csv"

test = pd.read_csv(path2)

test.head()
print(train.info())

print("\n")

print("--------------------------------------")

print(test.info())
train.describe()
test.describe()
# Let's look at the target variable first

train['Survived'].describe()
sns.distplot(train['Age'])
sns.boxplot(y=train['Age'], data=train, width=0.2, palette='autumn')
sns.distplot(train['Fare'])
# Let's look at the sex ratio.

(train['Sex'].value_counts()/len(train['Sex'])*100).plot.bar()
# Mean age of both sex groups

train.groupby('Sex')['Age'].mean().plot.bar()
# Performing ch-square test to know how different both sex groups are

stats.chi2_contingency(pd.crosstab(train['Sex'], train['Survived']))
# Now, let's look at how gender affected the survival rate of passengers

pd.crosstab(train['Sex'], train['Survived'])
# Relationship between survived and fare

sns.set_style(style='whitegrid')

sns.scatterplot(x='Fare', y='Survived', data=train)
train = train.drop(train[((train['Fare'] > 500) & (train['Survived'] > 0.8))].index)
# Relationship between fare and age

sns.scatterplot(x='Age', y='Fare', data=train, legend='brief')
train = train.drop(train[((train['Fare'] > 500) & (train['Age'] > 30))].index)
sns.boxplot(data=train, y='Age', x='Pclass')
sns.boxplot(data=train, y='Age', x='Survived')
sns.boxplot(data=train, y='Age', x='Parch')
sns.boxplot(data=train, y='Age', x='SibSp')
# Checking distribution of fare

sns.distplot(train["Fare"], color="m", label="Skewness : %.2f"%(train["Fare"].skew()))
# Checking correlation bw different variables

sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=.7, cbar=True, annot=True)
corr = train.corr()



# Sort in descending order

corr_top = corr['Survived'].sort_values(ascending=False)[:10]

top_features = corr_top.index[1:]

print(corr_top)
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False).plot(kind='bar')
train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False).plot(kind='bar')
train[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)
train[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False).plot(kind='bar')
train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False).plot(kind='bar')
train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False).plot(kind='bar')
""""Q1 = []

Q3 = []

Lower_Bound = []

Upper_Bound = []

Outliers = []



for i in top_features:

    

    # 25th and 75th percentiles

    q1, q3 = np.percentile(train[i], 25), np.percentile(train[i], 75)

    

    # Interquartile range

    iqr = q3 - q1

    

    # Outlier cutoff

    cut_off = 1.5*iqr

    

    # Lower and upper bounds

    lower_bound = q1 - cut_off

    upper_bound = q3 + cut_off

    

    # Save outlier indexes

    outlier = [x for x in train.index if train.loc[x,i] < lower_bound or train.loc[x,i] > upper_bound]

    

    # Append values for dataframe

    Q1.append(q1)

    Q3.append(q3)

    Lower_Bound.append(lower_bound)

    Upper_Bound.append(upper_bound)

    Outliers.append(len(outlier))

    

    try:

        train.drop(outlier, inplace=True, axis=0)

        

    except:

        continue

        

df_out = pd.DataFrame({'column':top_features,'Q1':Q1,'Q3':Q3,'Lower_Bound':Lower_Bound,'Upper_Bound':Upper_Bound,'No. of Outliers':Outliers})

df_out.sort_values(by='No. of Outliers', ascending=False)"""
# Now, look at the size of this dataset

train.shape
# Saving train rows

ntrain = train.shape[0]



# Save the target variable

target = train['Survived']



# Drop Id and SalePrice from train dataframe

train.drop(['PassengerId', 'Ticket', 'Survived'], inplace=True, axis=1)



# Store test Id

test_Id = test['PassengerId']



# Drop test Id

test.drop(['PassengerId', 'Ticket'], inplace=True, axis=1)



# Concatenate train and test dataframes

train = pd.concat([train, test])
train.isnull().sum().sort_values(ascending=False)
# Filling Cabin with most frequent occurences

train['Cabin'].fillna(train['Cabin'].mode()[0], inplace=True)



# Getting the first letter of the cabin as the cabin name

def take_section(code):

    return code[0]

train['Cabin'] = train['Cabin'].apply(take_section)



# Converting all cabin categories into numericals

train['Cabin'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'],[2,1,4,7,6,5,3,0], inplace=True)



# Filling Embarked with most frequent occurences

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)



# Converting all embarked categories into numericals

train['Embarked'].replace(['Q', 'C', 'S'],[1,2,0], inplace=True)



# Filling age

train['Age'].fillna(train['Age'].astype('float').median(axis=0), inplace=True)



# Filling Fare

train['Fare'].fillna(train['Fare'].astype('float').dropna().median(axis=0), inplace=True)
# Getting useful ticket no.

#ticket = []

#for i in list(train["Ticket"]):

    #if i.isdigit():

        #ticket.append("x")  # Displaying ticket as a 'x' wherever the ticket as a whole is an integer

    #else:

        #ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])  # Else getting the prefix as ticket no.

        

#train["Ticket"] = ticket

#train["Ticket"].head()
# Using expression pattern to extract the Title of the passenger

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



# Changing to common category

train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don', 'Dona'], 'Others')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')

train['Title'] = train['Title'].replace('Mlle', 'Miss')



# Converting all Title categories into numericals

title_mapping = {"Mr": 1, "Miss": 4, "Mrs": 5, "Master": 3, "Others": 2}

train['Title'] = train['Title'].map(title_mapping)

train['Title'] = train['Title'].fillna(0)



# After getting title from name, drop the Name variable

train.drop(['Name'],axis=1,inplace=True)



train.head()
# Forming ageband

train['AgeBand'] = pd.cut(train['Age'], 5)



# Overwriting values in age with the help of ageband

train.loc[train['Age'] <= 16, 'Age'] = 0

train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1

train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2

train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3

train.loc[train['Age'] > 64, 'Age'] = 4



train.head()
# Now remove this feature

train = train.drop(['AgeBand'],axis=1)

train.head()
# Similarly forming Fareband

train['FareBand'] = pd.cut(train['Fare'], 4)



# Overwriting values in age with the help of ageband

train.loc[train['Fare'] <= 7.91, 'Fare'] = 0

train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare'] = 2

train.loc[train['Fare'] > 31, 'Fare'] = 3



train.head()
# Now remove this feature

train = train.drop(['FareBand'],axis=1)

train.head()
# Getting family size from sibling/spouse and parent/children variable and adding 1 is for the person himself 

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



# Converting all FamilySize categories into numericals

train['FamilySize'].replace([1,2,3,4,5,6,7,8,11],[3,5,6,7,2,1,4,0,0], inplace=True)



train.head()
# Converting categorical feature into numericals

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()
# Getting new features from FamilySize

#train['Single'] = train['FamilySize'].map(lambda x: 1 if x == 1 else 0)

#train['SmallFam'] = train['FamilySize'].map(lambda x: 1 if 2 <= x <= 3 else 0)

#train['MedFam'] = train['FamilySize'].map(lambda x: 1 if 4 <= x <= 5 else 0)

#train['LargeFam'] = train['FamilySize'].map(lambda x: 1 if x >= 6 else 0)



# Dropping this feature

#train = train.drop(['FamilySize'], axis=1)

#train.head()
rand_state = 25

# Train dataset

df = train.iloc[:ntrain,:]



# Test dataset

test = train.iloc[ntrain:,:]



# Seperating independent and dependent variables

X = df

y = target



# train,test split to get training,validation and testing

X_train,X_test,y_train,y_test = ms.train_test_split(X, y, random_state=rand_state, test_size=0.2)
#Validation function

n_folds = 5



def scores_cv(model):

    kf = ms.StratifiedKFold(n_folds, shuffle=True, random_state=rand_state).get_n_splits(train.values)

    scores = ms.cross_val_score(model, X_train, y_train, scoring="accuracy", cv = kf)

    return(scores)
DTC = tree.DecisionTreeClassifier(random_state=rand_state)

DTC.fit(X_train, y_train)
ABC = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=rand_state),random_state=rand_state,learning_rate=0.1)

ABC.fit(X_train, y_train)
XGBC = xgb.XGBClassifier(learning_rate=0.05,random_state =rand_state)

XGBC.fit(X_train, y_train)
LGBMC = lgb.LGBMClassifier(learning_rate=0.05)

LGBMC.fit(X_train, y_train)
RFC = ensemble.RandomForestClassifier(random_state=rand_state)

RFC.fit(X_train, y_train)
KNNC = neighbors.KNeighborsClassifier(n_neighbors=7)

KNNC.fit(X_train, y_train)
LR = lm.LogisticRegression(random_state = rand_state)

LR.fit(X_train, y_train)
scores = scores_cv(DTC)

print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
scores = scores_cv(ABC)

print("\nAda Boost score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
scores = scores_cv(XGBC)

print("\nXG Boost score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
scores = scores_cv(LGBMC)

print("\nLightGBM score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
scores = scores_cv(RFC)

print("\nRandom Forest score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
scores = scores_cv(KNNC)

print("\nKNN score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
scores = scores_cv(LR)

print("\nLogistic Regression score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))
param_dist = {'num_leaves':stats.randint(1,20), 'max_depth':stats.randint(1,15), 'learning_rate':[0.05, 0.1, 0.3]

              , 'n_estimators':[100, 300, 500], 'min_child_weight':nr.random(5), 'min_child_samples':stats.randint(1,20)

              , 'subsample':nr.random(1), 'colsample_bytree':nr.random(1)}

LightGBM = lgb.LGBMClassifier(random_state=rand_state)

LightGBM_cv = ms.RandomizedSearchCV(LightGBM,param_distributions=param_dist,cv=5)

LightGBM_cv.fit(X_train, y_train)

print("Tuned LightGBM Parameters: {}".format(LightGBM_cv.best_params_)) 

print("Best score is {}".format(LightGBM_cv.best_score_))
param_dist = {'colsample_bytree':nr.random(1), "learning_rate":[0.05, 0.01, 0.1, 0.3]

              , "max_depth":stats.randint(1,20), "min_child_weight":nr.random(5)

              , "n_estimators":[100, 300, 500]

              , "subsample":nr.random(1)}

XGBC = xgb.XGBClassifier(random_state=rand_state)

XGBC_cv = ms.RandomizedSearchCV(XGBC,param_distributions=param_dist,cv=5)

XGBC_cv.fit(X_train, y_train)

print("Tuned XGBoost Parameters: {}".format(XGBC_cv.best_params_)) 

print("Best score is {}".format(XGBC_cv.best_score_)) 
param_dist = {'n_estimators':[100,200,300,400,500,600], 'criterion':['gini','entropy']

              , 'max_depth':stats.randint(1,15), 'max_features':stats.randint(1,9), 'min_samples_leaf':stats.randint(1,9)}

RFC = ensemble.RandomForestClassifier(random_state=rand_state)

RFC_cv = ms.RandomizedSearchCV(RFC,param_distributions=param_dist,cv=5)

RFC_cv.fit(X_train, y_train)

print("Tuned Random Forest Tree Parameters: {}".format(RFC_cv.best_params_)) 

print("Best score is {}".format(RFC_cv.best_score_)) 
param_dist = {'C':[.1,1,10,100,1000]}

LR = lm.LogisticRegression(random_state=rand_state)

LR_cv = ms.RandomizedSearchCV(LR,param_distributions=param_dist,cv=5)

LR_cv.fit(X_train, y_train)

print("Tuned Logistic Regression Parameters: {}".format(LR_cv.best_params_)) 

print("Best score is {}".format(LR_cv.best_score_))
RFC_best = ensemble.RandomForestClassifier(criterion='entropy', max_depth=4, max_features=4, min_samples_leaf=4, n_estimators=500)

XGB_best = xgb.XGBClassifier(colsample_bytree=0.11928413027995177, learning_rate=0.1, max_depth=7, min_child_weight=0.6181595928891843

                             , n_estimators=500, subsample=0.8332606985908653)

LGBM_best = lgb.LGBMClassifier(colsample_bytree=0.1050164438734883, learning_rate=0.05, max_depth=11, min_child_samples=13

                               , min_child_weight=0.7362051196763799, n_estimators=300, num_leaves=9, subsample=0.5648846078715707)

LR_best = lm.LogisticRegression(C=0.1)
#votingC = ensemble.VotingClassifier(estimators=[('RFC', RFC_best), ('LR', LR_best),

#('XGB', XGB_best), ('LGBM', LGBM_best)], voting='hard', n_jobs=4)



#votingC.fit(X_train, y_train)



# Filling the predictions into test_Survived

#test_survived = pd.Series(votingC.predict(test), name="Survived")
LGBM_best = lgb.LGBMClassifier(colsample_bytree=0.1050164438734883, learning_rate=0.05, max_depth=11, min_child_samples=13

                               , min_child_weight=0.7362051196763799, n_estimators=300, num_leaves=9, subsample=0.5648846078715707)

LGBM_best.fit(X_train, y_train)



test_survived = pd.Series(LGBM_best.predict(test), name="Survived")
RFC_best = ensemble.RandomForestClassifier(criterion='entropy', max_depth=4, max_features=4, min_samples_leaf=4, n_estimators=500)

RFC_best.fit(X_train, y_train)



test_survived = pd.Series(RFC_best.predict(test), name="Survived")
#XGB_best = xgb.XGBClassifier(colsample_bytree=0.11928413027995177, learning_rate=0.1, max_depth=7, min_child_weight=0.6181595928891843

                             #, n_estimators=500, subsample=0.8332606985908653)

#XGB_best.fit(X_train, y_train)

#test_survived = pd.Series(XGB_best.predict(test), name="Survived")
#LR_best = lm.LogisticRegression(C=0.1)

#LR_best.fit(X_train, y_train)

#test_survived = pd.Series(LR_best.predict(test), name="Survived")
subm_dict = {'PassengerId':test_Id, 'Survived':test_survived}

submit = pd.DataFrame(subm_dict)

submit.to_csv('titanic_submission_a.csv', index=False)