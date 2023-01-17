# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.shape
train.head()
train.dtypes
#Categorical
sorted(train['Survived'].unique())
#Categorical
sorted(train['Pclass'].unique())
#Categorical
len(train['Name'].unique())
#Categorical
train['Sex'].unique()
#Numerical
train['Age'].describe()
#Numerical
train['SibSp'].describe()
#Numerical
train['Parch'].describe()
#TicketID
"Num of unique tickets = "+str(len(train['Ticket'].unique()))
#Numerical
train["Fare"].describe()
#CabinID
"Num of unique cabins = " + str(len(train['Cabin'].unique()))
#Categorical
train['Embarked'].unique()
#Cabin has a lot of nan values
train.isna().sum()
#Drop nan values and some of columns
train = train.drop(columns=['Ticket', 'Cabin', 'Name'], errors = 'ignore').dropna()
train.shape
print("Survived: {}%".format(train[train['Survived']!=0]["Age"].count()/train["Age"].count()*100))
train["sex_encoded"] = train['Sex'].replace({'male':0,'female':1})
train["embarked_encoded"] = train["Embarked"].replace({'S':0,'C':1,'Q':2})
train.shape
#train = train.drop(columns = ["Sex", "Embarked"], errors = "ignore")
train.head(2)
train_cov = train.drop(columns = ["PassengerId"], errors = "ignore").cov()
train_cov
train_corr = train.drop(columns = ["PassengerId"], errors = "ignore").corr()
train_corr
fig = plt.figure(figsize=(10,7))
sns.heatmap(train_corr, annot = True)
fig.show()
fare = train[["Survived","Fare", "sex_encoded"]]
#Likelyhood of extreme events are very high 
fare['Fare'].kurtosis()
#Highly, positively skewed
fare['Fare'].skew()
#We definetely have outliers in a dataset :D
#Positively skewed
#We have some extra peaks around 25,55,80 values, need to dig deeper
fig = plt.figure(figsize=(10,8))
fig.title = ("KDE plot of Fare")
sns.distplot(fare['Fare'], rug = True, kde = True, hist = False)
sns.FacetGrid(fare, hue="Survived", height = 8).map(sns.distplot,"Fare")
plt.axvline(fare['Fare'].mean(), color = "g", label = 'mean')
plt.axvline(fare['Fare'].median(), color = "r", label = 'median')
plt.legend()
fig = plt.figure(figsize = (10,15))
sns.boxplot(x = "Survived", y = "Fare", data = train, hue = 'Sex')
fare_without_outliers = fare[fare['Fare']<200]
fare_without_outliers.describe()
#Positively skewed
fare_without_outliers['Fare'].skew()
#Extreme events likelyhood is slightly bigger, than a normal distribution without outlier
fare_without_outliers['Fare'].kurtosis()
#Distribution without outlier
fig = plt.figure(figsize=(10,8))
sns.distplot(fare_without_outliers['Fare'], rug = True, kde = True, hist = False)
plt.axvline(fare_without_outliers['Fare'].mean(), color = "g", label = 'mean')
plt.axvline(fare_without_outliers['Fare'].median(), color = "r", label = 'median')
plt.legend()
fig = plt.figure(figsize=(10,15))
sns.violinplot(x = "Survived", y = "Fare", data = fare_without_outliers, inner = None)
sns.swarmplot(x = "Survived", y = "Fare", data = fare_without_outliers, color = 'w')
#Have extra pick
sns.distplot(train['Age'], rug=True,kde=True,hist=False)
sns.FacetGrid(train, hue="Sex").map(sns.distplot, 'Age').add_legend()
sns.boxplot(x="Survived", y="Age", hue="Sex", data=train)
fig, ax = plt.subplots(2, 2, figsize = (15,15))
fig.tight_layout(pad=2.5)
fig.suptitle("Age")
#taking all people's age and averaging. Sorting by index i.e. [0,1,...20,80]
data = train['Age'].value_counts().sort_index()
#survived
data_survived = train[train['Survived']!=0]["Age"].value_counts().sort_index()
data_dead = train[train['Survived']!=1]["Age"].value_counts().sort_index()

#Plot frequency chart of all ages
ax[0][0].bar(data.index, data.values)
ax[0][0].set_title("Frequency of each individual age")
ax[0][0].set_ylim([0,data.max()])
#Plot frequency chart of all survived 
ax[1][0].bar(data_dead.index, data_dead.values, label = "died")
ax[1][0].bar(data_survived.index, data_survived.values, label = 'survived')
ax[1][0].set_title("Survived over dead")
ax[1][0].legend()
ax[1][0].set_ylim([0,data.max()])
#Plot percentage of survivors in lines
ax[0][1].plot(data.index, data.values, label = "all")
ax[0][1].plot(data_survived.index, data_survived.values, label = "survived")
ax[0][1].set_title("Num of survivals at certain age")
ax[0][1].legend()
#Plot them all together
ax[1][1].bar(data.index, data.values, label = 'all')
ax[1][1].bar(data_survived.index, data_survived.values, label = 'survived')
ax[1][1].plot(data.index, data.values, label = "all",color = 'r')
ax[1][1].plot(data_survived.index, data_survived.values, label = "survived", color = 'b')
ax[1][1].set_xlim([0,train['Age'].max()])
ax[1][1].set_title("Putting all together")
ax[1][1].legend()

#d = (train[train['Survived']!=0]["Age"].dropna().value_counts().sort_index()/train["Age"].dropna().value_counts().sort_index())
#Filling gaps of None with mask
#s1mask = np.isfinite(d.values)
#ax[2][0].plot(d.index[s1mask], d.values[s1mask])
#ax[2][0].set_title("Percenile of survival at certain age")

#d2 = (train[train['Survived']!=1]["Age"].dropna().value_counts().sort_index()/train["Age"].dropna().value_counts().sort_index())
#s2mask = np.isfinite(d2.values)
#ax[2][1].plot(d2.index[s2mask], d2.values[s2mask], label = "dead")
#ax[2][1].plot(d.index[s1mask], d.values[s1mask], label = 'alive')
#ax[2][1].legend()
#ax[2][1].set_title("Percenile of death at certain age")
plt.figure(figsize=(5,5))
sns.countplot(train["Sex"])
plt.show()
plt.figure(figsize=(5,5))
sns.countplot(train["Sex"], hue = train["Survived"])
plt.show()
gender_and_age = train[["Survived", "Sex", "Age"]].replace({'female':0, 'male':1}).round(0).groupby(['Age', 'Sex', 'Survived']).size()
colors = {0:'r', 1:'b'}
titles = {0:'female', 1:'male'}
fg,ax = plt.subplots(1,2)
fg.suptitle("Female and male (red - female, blue - male)")
ax[0].set_title("Survived")

ax[1].set_title("Died")
for age, sex, survived in gender_and_age.index:
    num = gender_and_age[age][sex][survived]
    colored = colors[sex]
    titled = titles[sex]
    if survived:
        ax[0].scatter(age, num, color = colored)
    else:
        ax[1].scatter(age, num, color = colored)

 


x_train = train.drop(columns=["PassengerId", "Survived"])
y_train = train[["PassengerId", "Survived"]]
