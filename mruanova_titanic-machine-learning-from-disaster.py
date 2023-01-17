# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np # linear algebra
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from pandas import Series,DataFrame
from matplotlib import style
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head(5)
train_data.shape # 891
test_data = pd.read_csv("../input/titanic/test.csv")
test_data.head(5)
test_data.shape # 418
# train_data.info()
# train_data.describe()
# total = train_data.isnull().sum().sort_values(ascending=False)
# percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100
# percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
# missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
# missing_data

print(pd.isnull(train_data).sum())
print(pd.isnull(test_data).sum())
# drop unnecessary columns, these columns won't be useful in analysis and prediction
train_data = train_data.drop(['PassengerId','Name','Ticket'],axis=1)
train_data.head()
test_data = test_data.drop(['Name','Ticket'],axis=1)
test_data.tail()
# visualization

#data = [train_data, test_data]
#for dataset in data:
#    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
#    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
#    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
#    dataset['not_alone'] = dataset['not_alone'].astype(int)
#train_data['not_alone'].value_counts()
# visualization
FacetGrid = sns.FacetGrid(train_data, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
# only in train data, fill the two missing values with the most occurred value, which is "S".
train_data["Embarked"] = train_data["Embarked"].fillna("S")
# plot
sns.factorplot('Embarked','Survived', data=train_data,size=4,aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked', data=train_data, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train_data, order=[1,0], ax=axis2)
# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = train_data[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.
# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.
embark_dummies_titanic  = pd.get_dummies(train_data['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
embark_dummies_test  = pd.get_dummies(test_data['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)
train_data = train_data.join(embark_dummies_titanic)
test_data = test_data.join(embark_dummies_test)
train_data.drop(['Embarked'], axis=1,inplace=True)
test_data.drop(['Embarked'], axis=1,inplace=True)
# only for test data, since there is a missing "Fare" values
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
# convert from float to int
train_data['Fare'] = train_data['Fare'].astype(int)
test_data['Fare'] = test_data['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = train_data["Fare"][train_data["Survived"] == 0]
fare_survived = train_data["Fare"][train_data["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
train_data['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# get average, std, and number of NaN values in train data
avg_age_train = train_data["Age"].mean()
std_age_train = train_data["Age"].std()
count_nan_age_train = train_data["Age"].isnull().sum()

# get average, std, and number of NaN values in test data
avg_age_test = test_data["Age"].mean()
std_age_test = test_data["Age"].std()
count_nan_age_test = test_data["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(avg_age_train - std_age_train, avg_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(avg_age_test - std_age_test, avg_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
train_data['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# test_data['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
train_data["Age"][np.isnan(train_data["Age"])] = rand_1
test_data["Age"][np.isnan(test_data["Age"])] = rand_2

# convert from float to int
train_data['Age'] = train_data['Age'].astype(int)
test_data['Age'] = test_data['Age'].astype(int)

# plot new Age Values
train_data['Age'].hist(bins=70, ax=axis2)
# test_data['Age'].hist(bins=70, ax=axis4)
# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
train_data.drop(['Cabin'],axis=1,inplace=True)
train_data.head()
test_data.drop(['Cabin'],axis=1,inplace=True)
test_data.head()
# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train_data['Family'] =  train_data["Parch"] + train_data["SibSp"]
train_data['Family'].loc[train_data['Family'] > 0] = 1
train_data['Family'].loc[train_data['Family'] == 0] = 0

test_data['Family'] = test_data["Parch"] + test_data["SibSp"]
test_data['Family'].loc[test_data['Family'] > 0] = 1
test_data['Family'].loc[test_data['Family'] == 0] = 0

# drop Parch & SibSp
train_data = train_data.drop(['SibSp','Parch'], axis=1)
test_data = test_data.drop(['SibSp','Parch'], axis=1)

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
sns.countplot(x='Family', data=train_data, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = train_data[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# visualization
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_data[train_data['Sex']=='female']
men = train_data[train_data['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde = False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde = False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend() 
ax.set_title('Male')
# % of women who survived: train
women = train_data.loc[train_data.Sex == 'female']["Survived"]
if len(women) > 0 :
    rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

# % of men who survived: train
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = 0
if len(men) > 0 :
    rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)
sns.barplot(x='Sex', y='Survived', data=train_data)
# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
train_data['Person'] = train_data[['Age','Sex']].apply(get_person,axis=1)
test_data['Person'] = test_data[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
train_data.drop(['Sex'],axis=1,inplace=True)
test_data.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic = pd.get_dummies(train_data['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test = pd.get_dummies(test_data['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train_data = train_data.join(person_dummies_titanic)
test_data    = test_data.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Person', data=train_data, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = train_data[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

train_data.drop(['Person'],axis=1,inplace=True)
test_data.drop(['Person'],axis=1,inplace=True)
# axes = sns.factorplot(x='relatives',y='Survived',hue='Sex', data=train_data, aspect = 2.5 )
# axes = sns.catplot(x='relatives',y='Survived',hue='Sex',data=train_data, aspect = 2.5 )
# train_data['Sex'].replace(['female','male'],[0,1],inplace=True)
# test_data['Sex'].replace(['female','male'],[0,1],inplace=True)

# train_data['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
# test_data['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_data)
# Pclass

# sns.factorplot('Pclass',data=train_data,kind='count',order=[1,2,3])
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_data,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(train_data['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_data['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train_data.drop(['Pclass'],axis=1,inplace=True)
test_data.drop(['Pclass'],axis=1,inplace=True)

train_data = train_data.join(pclass_dummies_titanic)
test_data    = test_data.join(pclass_dummies_test)
print(pd.isnull(train_data).sum())
print(pd.isnull(test_data).sum())
# bins = [0,8,15,18,25,40,60,100]
# names = ['1','2','3','4','5','6','7']
# train_data['Age'] = pd.cut(train_data['Age'],bins,labels=names)
# test_data['Age'] = pd.cut(test_data['Age'],bins,labels=names)
print(train_data.shape)
print(test_data.shape)
print(train_data.head())
print(test_data.head())
train_data.head()
test_data.head()
# train_test_split
X = np.array(train_data.drop(['Survived'],1))
y = np.array(train_data['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
ids = test_data.PassengerId
# Logistic Regression # 0.81
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Y_pred = model.predict(test_data.drop('PassengerId',axis=1)) # X_test
score_logreg = model.score(X_train, y_train)
print('Logistic Regression score:', score_logreg)
score_logreg = model.score(X_test, y_test)
print('Logistic Regression score:', score_logreg)

out_logreg = pd.DataFrame({'PassengerId':ids,'Survived':Y_pred})
print(out_logreg.head())

# % of people who survived: train
people = out_logreg.loc[out_logreg.Survived == 1]["Survived"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(out_logreg) # 39 %
print("Logistic Regression % of people who survived:", rate_people)
model = SVC()
model.fit(X_train, y_train)
Y_pred = model.predict(test_data.drop('PassengerId',axis=1)) # X_test
score_svc = model.score(X_train, y_train)
print('Support Vector Clustering score: ', score_svc)
score_svc = model.score(X_test, y_test)
print('Support Vector Clustering score: ', score_svc)

out_svc = pd.DataFrame({'PassengerId':ids,'Survived':Y_pred})
print(out_svc.head())

# % of people who survived: train
people = out_svc.loc[out_svc.Survived == 1]["Survived"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(out_svc) # 18 %
print("Support Vector Clustering % of people who survived:", rate_people)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
Y_pred = model.predict(test_data.drop('PassengerId',axis=1)) # X_test
score_knc = model.score(X_train, y_train)
print('K Neighbors Classifier score:', score_knc)
score_knc = model.score(X_test, y_test)
print('K Neighbors Classifier score:', score_knc)

out_knc = pd.DataFrame({'PassengerId':ids,'Survived':Y_pred})
print(out_knc.head())

# % of people who survived: train
people = out_knc.loc[out_knc.Survived == 1]["Survived"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(out_knc) # 38 %
print("K Neighbors Classifier % of people who survived:", rate_people)
model = RandomForestClassifier(n_estimators=100) # , max_depth=5, random_state=1
model.fit(X_train, y_train)
Y_pred = model.predict(test_data.drop('PassengerId',axis=1)) # X_test
score_rfc = model.score(X_train, y_train)
print('Random Forest Classifier score: ', score_rfc)
score_rfc = model.score(X_test, y_test)
print('Random Forest Classifier score: ', score_rfc)

# X_test = pd.get_dummies(test_data[features])
out_rfc = pd.DataFrame({'PassengerId':ids, 'Survived': Y_pred})
print(out_rfc.head())

# % of people who survived: train
people = out_rfc.loc[out_rfc.Survived == 1]["Survived"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(out_rfc) # 148 / 418 = 35%
print("Random Forest Classifier % of people who survived:", rate_people)
decision_tree = DecisionTreeClassifier(max_depth=5) 
decision_tree.fit(X_train, y_train)  
Y_pred = model.predict(test_data.drop('PassengerId',axis=1)) # X_test
score_dtc = decision_tree.score(X_train, y_train)
print('Decision Tree Classifier score =', score_dtc)
score_dtc = decision_tree.score(X_test, y_test)
print('Decision Tree Classifier score =', score_dtc)

out_dtc = pd.DataFrame({'PassengerId':ids, 'Survived': Y_pred})
print(out_dtc.head())

# % of people who survived: train
people = out_dtc.loc[out_dtc.Survived == 1]["Survived"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(out_dtc) # 148 / 418 = 35%
print("Decision Tree Classifier % of people who survived:", rate_people)
# Fixing random state for reproducibility
plt.rcdefaults()
fig, ax = plt.subplots()
people = ('Logistic Regression', 'Support Vector Clustering', 'K Neighbors Classifier', 'Random Forest Classifier', 'Decision Tree Classifier')
y_pos = np.arange(len(people))
scores = (score_logreg,score_svc,score_knc,score_rfc,score_dtc)
ax.barh(y_pos, scores, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis() # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('Which one is the best algorithm?')
plt.show()
score = score_logreg
output = out_logreg
print("Logistic Regression!", score)
if score_svc > score:
    score = score_svc
    output = out_svc
    print("Support Vector Clustering!", score)
if score_knc > score:
    score = score_knc
    output = out_knc
    print("K Neighbors Classifier!", score)
if score_rfc > score:
    score = score_rfc
    output = out_rfc
    print("Random Forest Classifier!", score)
if score_dtc > score:
    score = score_dtc
    output = out_dtc
    print("Decision Tree Classifier!", score)
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
output.shape
output.head()
output.tail()