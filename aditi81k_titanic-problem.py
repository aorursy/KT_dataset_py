import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#Load the Train set

train_set = pd.read_csv("../input/titanic/train.csv")

test_set=pd.read_csv("../input/titanic/test.csv")

train_set.tail(10)
# inspect the structure etc.

print(train_set.info(), "\n")

print(train_set.shape)
train_set['Survived'].value_counts()
sns.countplot(train_set['Survived'])

plt.show()

print('Percent of fraud transaction: ',len(train_set[train_set['Survived']==1])/len(train_set['Survived'])*100,"%")

print('Percent of normal transaction: ',len(train_set[train_set['Survived']==0])/len(train_set['Survived'])*100,"%")
# missing values in Train set df

train_set.isnull().sum()
round(100*(test_set.isnull().sum().sort_values(ascending=False)/len(test_set.index)), 2)
train_set.Age.describe()
train_set['Title']=train_set['Name'].map(lambda x: x.split(',')[1].split('.')[0].lstrip())

test_set['Title']=test_set['Name'].map(lambda x: x.split(',')[1].split('.')[0].lstrip())

train_set.head()
train_set['Title'].value_counts()
print(train_set.info())
#Check the list of values in title column

train_set.Title.unique()
# lets sort the remaining other categories in title to various sub category of Mr, Miss, mrs, train_set

title_list=['Mrs', 'Mr', 'Master', 'Miss']

train_set.loc[~train_set['Title'].isin(title_list),['Age','Sex','Title']]
# function to bucket other titles into major 4

def fix_title(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme','Lady','Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
train_set['Title']=train_set.apply(fix_title, axis=1)

train_set['Title'].value_counts()
test_set['Title']=test_set.apply(fix_title, axis=1)

test_set['Title'].value_counts()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}

train_set['Title'] = train_set['Title'].map(title_mapping)

train_set['Title'] = train_set['Title'].fillna(0)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}

test_set['Title'] = test_set['Title'].map(title_mapping)

test_set['Title'] = test_set['Title'].fillna(0)
train_set.Age.isnull().sum()
#Check mean  on title Subclass w.r.t Age

train_set.groupby(['Title'])['Age'].describe()
train_set.groupby(['Title'])['Age'].median()
round(100*(train_set.isnull().sum().sort_values(ascending=False)/len(train_set.index)), 2)
# Total Nullvalues in Age Column

train_set.Age.isnull().sum()
title = train_set['Title'].value_counts()

a= dict(title)

for keys, values in a.items():

    print('Value of {} Class with Null Values {other}'.format(keys, other=train_set.loc[(train_set.Title==keys),['Age']].isnull().sum()))
#Impute Missing values in Age Column

for keys, values in a.items():

    missing_val=train_set.loc[(train_set.Title==keys) & ~(train_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

    train_set.loc[(train_set.Title==keys) & (train_set.Age.isnull()),'Age']=train_set.loc[(train_set.Title==keys) & (train_set.Age.isnull()),'Age'].replace(np.nan,missing_val.median())
title = train_set['Title'].value_counts()

b= dict(title)

for keys, values in a.items():

    print('Value of {} Class with Null Values {other}'.format(keys, other=test_set.loc[(test_set.Title==keys),['Age']].isnull().sum()))
#Impute Missing values in Age Column

for keys, values in b.items():

    missing_val=test_set.loc[(test_set.Title==keys) & ~(test_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

    test_set.loc[(test_set.Title==keys) & (test_set.Age.isnull()),'Age']=test_set.loc[(test_set.Title==keys) & (test_set.Age.isnull()),'Age'].replace(np.nan,missing_val.median())
# After Imputation on Age Colums verify the null values

train_set.Age.isnull().sum()
test_set.Age.isnull().sum()
test_set['Fare'].fillna(test_set['Fare'].median(), inplace=True)
# GEt the unique set of Value of Cabin

train_set.Cabin.unique()
# Lets see the cabin with passenger class

class_cabin=train_set.groupby(['Pclass'])['Cabin'].count()

class_cabin
# No Pclass

train_set.Pclass.value_counts()
cls = train_set['Pclass'].value_counts()

for key, value in (dict(cls)).items():

    print('Value of {} passenger Class with Null Values {other}'.format(key,other=train_set.loc[(train_set.Pclass==key),['Cabin']].isnull().sum()))
train_set.loc[(train_set.Pclass==1) & ~(train_set.Cabin.isnull()),['Cabin']]

# Lets have Deck # as separte Columns and null as GNR

train_set['Deck']=pd.Series(train_set.loc[~(train_set.Cabin.isnull()),['Cabin']].values.flatten()).astype('str').str[0]
deck = pd.Series(train_set['Cabin'].values.flatten().astype('str'))

deck1 = []

for i in deck:

    if i != 'nan':

        deck1.append(i[0])

    else: 

        deck1.append(i)
train_set['Deck']=deck1
train_set.loc[~(train_set.Cabin.isnull()),['Cabin']].values.flatten()
train_set['Deck']
train_set.head(50)
# Lets see the unique value and count of Deck Column

train_set['Deck'].value_counts()
train_set.Deck.unique()
train_set.Deck.isnull().sum()
# Replace Nan in Decek to GNR

train_set['Deck']=train_set['Deck'].replace('nan','GNR')
train_set['Deck'].value_counts()
# Remove Cabin Column

train_set.drop('Cabin',axis=1,inplace=True)

test_set.drop('Cabin',axis=1,inplace=True)
train_set.head()
# Now lets check the column with null values

train_set.isnull().sum()
# Value of Embarked on various categories

train_set.Embarked.value_counts()
train_set.Embarked.isnull().sum()
#Lets impute 2 null records of Embarked with value 'S' as it have max occurance

train_set.loc[(train_set.Embarked.isnull()),'Embarked']=train_set.loc[ (train_set.Embarked.isnull()),'Embarked'].replace(np.nan,'S')

train_set.Embarked.isnull().sum()
# Check if any null columns are present

train_set.isnull().sum()
sns.distplot(train_set['Age'])
g = sns.FacetGrid(train_set, col='Survived',size=5)

g.map(plt.hist, 'Age', bins=30)
# pairplot

sns.pairplot(train_set)

plt.show()
sns.countplot(x="Pclass", data=train_set)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_set, col='Survived', row='Pclass', size=4, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=30)

grid.add_legend();
sns.distplot(train_set['Fare'])
sns.boxplot(y=train_set['Fare'])
# Checking for Outlier

train_set.Fare.describe(percentiles=[.25, .5, .75, .90, .95, .99])
grid = sns.FacetGrid(train_set, row='Embarked', col='Survived', size=4, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
import seaborn as sns

sns.countplot(x="Survived", data=train_set)
### Checking the Survival Rate Rate

survival = (sum(train_set['Survived'])/len(train_set['Survived'].index))*100

survival
# Remove name and Passenger Id Column

train_set.drop(['Name'],axis=1,inplace=True)

test_set.drop(['Name'],axis=1,inplace=True)
#Checking the Correlation Matrix

plt.figure(figsize = (15,10))

sns.heatmap(train_set.corr(),annot = True)

plt.show()
#Check the Survival rate by Paasaenger Class

# print(train_set [['Pclass','Survived']].groupby('Pclass').mean())

a = train_set.groupby(['Pclass','Survived']).agg({'Pclass': 'sum'})

a.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
sns.countplot(x="Pclass", hue="Survived", data=train_set)
sep="---------------------------------------------------------------"

a = train_set.groupby(['Pclass','Sex','Survived']).agg({'Pclass': 'sum'})

a.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

# print(round(a,2),'\n')
#Check the Survival rate by Paasaenger Class

sep="---------------------------------------------------------------"

print( round(train_set [['Sex','Survived']].groupby(['Sex']).mean()*100,1),'\n',sep)

print(train_set [['Pclass','Sex','Survived']].groupby(['Pclass','Sex']).agg(['count','mean']))
#tracking the Survival on the basis of Sex and PClass

g = sns.catplot(x="Pclass", hue="Sex", col="Survived",

                data=train_set, kind="count",

                height=4, aspect=.7, size = 7);
# check the impact of Embarked Colum on Survival

print( round(train_set [['Embarked','Survived']].groupby(['Embarked']).mean()*100,1))
pd.crosstab(train_set['Survived'],train_set['Pclass']).apply(lambda r: (r/r.sum())*100, axis=1)
pd.crosstab(train_set['Survived'],[train_set['Pclass'],train_set['Sex']]).apply(lambda r: (r/r.sum())*100, axis=1)
grid = sns.FacetGrid(train_set, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
pd.crosstab(train_set['Survived'],[train_set['Embarked'],train_set['Sex']]).apply(lambda r: (r/r.sum())*100, axis=1)
train_set.loc[(train_set['Parch']==0)&(train_set['SibSp']==0)]
sns.countplot(x="Parch", hue="Survived", data=train_set)
sns.countplot(x="SibSp", hue="Survived", data=train_set)
train_set['Family']=train_set['SibSp']+train_set['Parch']+1

test_set['Family']=test_set['SibSp']+test_set['Parch']+1

train_set.head()
train_set[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set['IsAlone'] = 0

test_set['IsAlone'] = 0

train_set.loc[train_set['Family'] == 1, 'IsAlone'] = 1

test_set.loc[test_set['Family'] == 1, 'IsAlone'] = 1

train_set[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
df = train_set.groupby(['Ticket']).size().reset_index(name='count')

print(df)
# New column for Ticket Head Count on teh complete data

# master=pd.concat([train_set, test_set])

# master.head()

# train_set['TicketHeadCount']=train_set['Ticket'].map(master['Ticket'].value_counts())

train_set['TicketHeadCount']=train_set['Ticket'].map(train_set['Ticket'].value_counts())

test_set['TicketHeadCount']=test_set['Ticket'].map(test_set['Ticket'].value_counts())

train_set.head()
#Let take fair per Person as per Ticket head Count

train_set['FairPerPerson']=train_set['Fare']/train_set['TicketHeadCount']

test_set['FairPerPerson']=test_set['Fare']/test_set['TicketHeadCount']

train_set[['FairPerPerson']].describe(percentiles=[.25, .5, .75, .90, .95, .99])
train_set['FairPerPerson'].value_counts()<1
# Lets check the distribution

sns.distplot(train_set['FairPerPerson'])
#Check the impact of Fair on chances of Survival

plt.figure(figsize = (15,10))

sns.violinplot(x="Family", y="FairPerPerson", hue="Survived",

                    data=train_set, palette="muted")

plt.show()
sns.violinplot(x="Pclass", y="FairPerPerson", hue="Sex",

                    data=train_set, palette="muted")
sns.violinplot(x="Survived", y="Age", hue="Sex",

                    data=train_set, palette="muted")
plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>0.0) & (train_set.Age<15.0)],jitter=True,palette='Set1')

plt.subplot(2,3,2)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>15.0) & (train_set.Age<40.0)],jitter=True,palette='Set1')

plt.subplot(2,3,3)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>40.0) & (train_set.Age<60.0)],jitter=True,palette='Set1')

plt.subplot(2,3,4)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>60.0) & (train_set.Age<80.0)],jitter=True,palette='Set1')



plt.show()
# Lets check on graph the survival of male aginst Female within age 15-40 years

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>15.0) & (train_set.Age<40.0)],jitter=True,hue='Sex',palette='Set1')
#tracking th Survival on the basis of Family Size and Sex

g = sns.catplot(x="Family", hue="Sex", col="Survived",

                data=train_set, kind="count",

                height=7, aspect=.7);
sns.boxplot(x = 'Pclass', y = 'FairPerPerson',hue='Survived', data = train_set)
# Group the Deck by Class

print(train_set.groupby([ 'Pclass','Deck'])['Survived'].agg(['count','mean']))
# Lets Check the pattern of Deck on Age

sns.swarmplot(x="Deck",y="Age",hue='Sex',data=train_set,palette="Set1", split=True)
sns.swarmplot(x="Deck",y="FairPerPerson",hue='Pclass',data=train_set,palette="Set1", split=True)
plt.figure(figsize=(25, 14))

sns.catplot(x="Deck", col="Survived",data=train_set, kind="count",height=4, aspect=.7, hue='Pclass')

plt.show()
print(train_set.loc[(train_set['FairPerPerson']==0),['Embarked','Ticket','SibSp','Parch','Age','Sex','Family','TicketHeadCount']])
train_set['FairPerPerson'].describe(percentiles=[.25, .5, .75, .90, .95, .99])
train_set.info()
quantile_1, quantile_3 = np.percentile(train_set.FairPerPerson, [25, 75])
print(quantile_1, quantile_3)
iqr_value = quantile_3 - quantile_1

iqr_value
lower_bound_val = quantile_1 - (1.5 * iqr_value)

upper_bound_val = quantile_3 + (1.5 * iqr_value)

print(lower_bound_val, upper_bound_val)
plt.figure(figsize = (10, 5))

sns.kdeplot(train_set.FairPerPerson)

plt.axvline(x=lower_bound_val, color = 'red')

plt.axvline(x=upper_bound_val, color = 'red')
train_set[(train_set.FairPerPerson >= lower_bound_val) & (train_set.FairPerPerson <= upper_bound_val)].info()
round(100*(train_set[(train_set.FairPerPerson >= lower_bound_val) & (train_set.FairPerPerson <= upper_bound_val)].count()/len(train_set.index)), 2)
round(100*(train_set[(train_set.FairPerPerson >= 0) & (train_set.FairPerPerson <= 100)].count()/len(train_set.index)), 2)
train_set_copy=train_set.loc[(train_set.FairPerPerson>0) & (train_set.FairPerPerson<=100)]

train_set_copy.shape
train_set_copy.info()
sns.boxplot(x = 'Pclass', y = 'FairPerPerson',hue='Survived', data = train_set_copy)
train_set.head()
test_set.head()
train_set_copy.drop(['Parch','Ticket','Fare','Deck','SibSp','TicketHeadCount'],axis=1,inplace=True)

test_set.drop(['Parch','Ticket','Fare','SibSp','TicketHeadCount'],axis=1,inplace=True)

train_set_copy.head()
train_set_copy = pd.concat([train_set_copy, pd.get_dummies(train_set_copy['Sex'], drop_first=True)], axis=1)

test_set = pd.concat([test_set, pd.get_dummies(test_set['Sex'], drop_first=True)], axis=1)
train_set_copy['Embarked'] = train_set_copy['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_set['Embarked'] = test_set['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#Drop Original Columns

train_set_copy.drop(['Sex'],axis=1,inplace=True)

test_set.drop(['Sex'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
X_train = train_set_copy.drop("Survived", axis=1)

y_train = train_set_copy["Survived"]

X_test  = test_set.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape
scaler = StandardScaler()



X_train[['Age','FairPerPerson']] = scaler.fit_transform(X_train[['Age','FairPerPerson']])

X_test[['Age','FairPerPerson']] = scaler.fit_transform(X_test[['Age','FairPerPerson']])



X_train.head()
### Checking the Survival Rate

Survival = (sum(train_set_copy['Survived'])/len(train_set_copy['Survived'].index))*100

Survival
plt.figure(figsize = (20,10))

sns.heatmap(X_train.corr(),annot = True)

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,accuracy_score
col = [ 'Pclass', 'Age', 'Embarked', 'Title', 'Family',

       'IsAlone', 'FairPerPerson', 'male']
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train[col], y_train)

Y_pred = logreg.predict(X_test[col])

acc_log = round(logreg.score(X_train[col], y_train) * 100, 2)

acc_log
# Support Vector Machines



svc = SVC()

svc.fit(X_train[col], y_train)

Y_pred = svc.predict(X_test[col])

acc_svc = round(svc.score(X_train[col], y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train[col], y_train)

Y_pred = knn.predict(X_test[col])

acc_knn = round(knn.score(X_train[col], y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train[col], y_train)

Y_pred = gaussian.predict(X_test[col])

acc_gaussian = round(gaussian.score(X_train[col], y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train[col], y_train)

Y_pred = perceptron.predict(X_test[col])

acc_perceptron = round(perceptron.score(X_train[col], y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train[col], y_train)

Y_pred = linear_svc.predict(X_test[col])

acc_linear_svc = round(linear_svc.score(X_train[col], y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train[col], y_train)

Y_pred = sgd.predict(X_test[col])

acc_sgd = round(sgd.score(X_train[col], y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train[col], y_train)

Y_pred = decision_tree.predict(X_test[col])

acc_decision_tree = round(decision_tree.score(X_train[col], y_train) * 100, 2)

acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train[col], y_train)

Y_pred = random_forest.predict(X_test[col])

random_forest.score(X_train[col], y_train)

acc_random_forest = round(random_forest.score(X_train[col], y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train[col], y_train)

Y_pred = random_forest.predict(X_test[col])

random_forest.score(X_train[col], y_train)

acc_random_forest = round(random_forest.score(X_train[col], y_train) * 100, 2)

acc_random_forest
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(X_train[col],y_train)



ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train[col],y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train[col],y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train[col],y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train[col], y_train)

Y_pred = votingC.predict(X_test[col])
from xgboost.sklearn import XGBClassifier

model = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

model.fit(X_train[col], y_train)

Y_pred = model.predict(X_test[col])
submission = pd.DataFrame({

        "PassengerId": test_set["PassengerId"],

        "Survived": Y_pred

     })

submission.to_csv('/kaggle/working/submission.csv', index=False)