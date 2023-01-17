import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # for machine learning
import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
import seaborn as sns

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import collections as co
import math

from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
train_data
test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
test_data.describe()
train_data.describe()
fig1, ax = plt.subplots()
ax.pie(train_data.Survived.value_counts(),explode = (0.1, 0.1), labels=(0, 1), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

survivalRate = plt.title("Distribution of Survival, (1 = Survived)")
plt.show()
fig1, ax = plt.subplots()
ax.pie(train_data.Pclass.value_counts(), explode = (0.1, 0.1, 0.1), labels=('Class 3', 'Class 1', 'Class 2'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

classes = plt.title("Distribution of Classes, (Pclass)")
plt.show()

sns.kdeplot(train_data.Age.dropna(), shade=True, label='Age')
plt.axvline(train_data.Age.dropna().median(), label='Median', color = 'red')
plt.axvline(train_data.Age.dropna().mean(), label='Mean', color = 'blue')
plt.legend()
ageDistribution = plt.title("Distribution of Age, (Age)")
fig1, ax = plt.subplots()
ax.pie(train_data.Sex.value_counts(), explode=[0.1, 0.1],  labels=('male', 'female'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

genderPie = plt.title("Distribution of Gender, (Sex)")
plt.show()

fig1, ax = plt.subplots()
ax.pie(train_data.Embarked.value_counts(), explode=[0.1, 0.1, 0.1],  labels=('Southampton', 'Cherbourg', 'Queenstown'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

embarked = plt.title("Distribution of Embarked, (Embarked)")
plt.show()

sns.kdeplot(train_data.Fare[train_data.Pclass==1].apply(lambda x: 80 if x>80 else x), shade=True, label='1st Class')
sns.kdeplot(train_data.Fare[train_data.Pclass==2].apply(lambda x: 80 if x>80 else x), shade=True, label='2nd Class')
sns.kdeplot(train_data.Fare[train_data.Pclass==3].apply(lambda x: 80 if x>80 else x), shade=True, label='3rd Class')
plt.axvline(train_data.Fare.median(), label='Median', ls='dashed')
plt.axvline(train_data.Fare.mean(), label='Mean', ls='dotted')
plt.legend()
fare = plt.title("Fare Distribution by Class, (Fare)")
train_data.SibSp.value_counts().plot(kind='bar', color = 'blue')
train_data.Parch.value_counts().plot(kind='bar', color = 'red', bottom = train_data.SibSp.value_counts())
plt.figure(figsize=(8, 6), dpi = 100)
plt.legend(labels = ('No. Of Siblings and Spouses (SibSp)', 'No. of Parents and Children (Parch)'))
plt.ylabel('Passenger Count')
plt.xlabel('Number of Family Members')
siblings = plt.title("Number of Siblings, Spouses, Parents and Children")
sns.kdeplot(train_data.Age[train_data.Sex=='male'].dropna(), shade=True, label='Male')
sns.kdeplot(train_data.Age[train_data.Sex=='female'].dropna(), shade=True, label='Female')
plt.legend()
agesex = plt.title("Age Distribution by Sex")
Hefirst = plt.subplot2grid((20,30),(0,6),rowspan=10,colspan=3)
train_data.Survived[train_data.Pclass==1].value_counts().sort_index().plot(color = 'blue', kind='bar', alpha=0.85, label='1st Class')
plt.title("Survival in 1st Class")
plt.ylabel("Passenger Count")
plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])

second = plt.subplot2grid((20,30),(0,14),rowspan=10,colspan=3)
train_data.Survived[train_data.Pclass==2].value_counts().sort_index().plot(color = 'blue', kind='bar', alpha=0.85, label='2nd Class')
plt.title("Survival in 2nd Class")
plt.ylabel("Passenger Count")
plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])

third = plt.subplot2grid((20,30),(0,22),rowspan=10,colspan=3)              
train_data.Survived[train_data.Pclass==3].value_counts().sort_index().plot(color = 'blue', kind='bar', alpha=0.85, label='3rd Class')
plt.title("Survival in 3rd Class")
plt.ylabel("Passenger Count")
plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])
plt.show()

plt.bar(np.array([0,1])-0.25, train_data.Survived[train_data.Sex=='male'].value_counts().sort_index(), width=0.25, label='Male',alpha=0.85)
plt.bar(np.array([0,1]), train_data.Survived[train_data.Sex=='female'].value_counts().sort_index(), width=0.25, label='Female',alpha=0.85)
plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])
plt.legend()
sexSurvived = plt.title("Survival By Gender (Sex, Survived = 1)")
sns.kdeplot(train_data.Age[train_data.Survived==0].dropna(), shade=True, label='Died')
sns.kdeplot(train_data.Age[train_data.Survived==1].dropna(), shade=True, label='Survived')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Percentile')
ageSurvival = plt.title("Survival By Age")
embarkedC = plt.subplot2grid((20,30),(0,6),rowspan=10,colspan=5)
plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==0], color = 'red', label='')
plt.title("Survival By Embarked (C)")
plt.ylabel("Passenger Count")
embarkedS = plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])

embarkedQ = plt.subplot2grid((20,30),(0,16),rowspan=10,colspan=5)
plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==1], color = 'green',label='')
plt.title("Survival By Embarked (Q)")
plt.ylabel("Passenger Count")
embarkedS = plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])

embarkedS = plt.subplot2grid((20,30),(0,26),rowspan=10,colspan=5) 
plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==2], color = 'yellow', label='')
plt.title("Survival By Embarked (S)")
plt.ylabel("Passenger Count")
embarkedS = plt.xticks(np.arange(0, 2, 1), ['Died' ,'Survived'])
train_data['FamilySize'] = train_data.SibSp + train_data.Parch;
sns.kdeplot(train_data.FamilySize[train_data.Survived==0],shade=True,label='Died');
sns.kdeplot(train_data.FamilySize[train_data.Survived==1],shade=True,label='Survived');
plt.title('Survival By Family Size');
plt.ylabel('Survival Rate')
plt.xlabel('Family Size')
plt.legend();
plt.show();

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

train_data['Title'] = train_data["Name"].apply(get_title);
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona":9, "Lady": 9, "Countess": 9, "Jonkheer": 9, "Sir": 7, "Capt": 7, "Ms": 2}
train_data["TitleCat"] = train_data.loc[:,'Title'].map(title_mapping);

# Set the bar width
bar_width = 0.75
indexes = np.arange(9)

titleFreqSurvived = train_data.TitleCat[(train_data.Survived==1)].value_counts()
titleFreqDied = train_data.TitleCat[(train_data.Survived==0)].value_counts()

titleFreqSurvived.set_value(6, 0)
titleFreqDied.set_value(5, 0)
titleFreqDied.set_value(8, 0)

titleFreqSurvived = titleFreqSurvived.sort_index()
titleFreqDied = titleFreqDied.sort_index()

plt.bar(indexes,
        titleFreqDied,
        width=bar_width,
        label = "Died",
        color='blue')


plt.bar(indexes,
       titleFreqSurvived,
        width=bar_width,
        color='green',
        label = 'Survived',
       bottom = titleFreqDied)

    # set the x ticks with names

# Set the label and legends
plt.ylabel("Count")
plt.xlabel("Title")
legend1 = plt.legend(loc=4)
legend2 = plt.legend()
plt.xticks(indexes, np.arange(1, 10, 1))
plt.title("Survival by Title of Passengers")
plt.figure(dpi = 1000)
print("LEGEND OF TITLES")
print("1 - Mr")
print("2 - Miss, Ms")
print("3 - Master")
print("4 - Doctor")
print("5 - Reverent")
print("6 - Major")
print("7 - Colonel, Don, Sir, Captain")
print("8 - Mme, Mlle")
print("9 - Dona, Lady, Countess, Jonkheer")
plt.show();
trainNumeric = train_data.copy()
# delete all unimportant cells
del trainNumeric['PassengerId']
del trainNumeric['Name']
del trainNumeric['Title']

#make all ticket numbers string
ct = 0
for ticket in trainNumeric.Ticket:
    i = len(ticket)-1
    new = ""
    while(i >= 0):
        if(ticket[i] >= '0' and ticket[i] <= '9'):
            if(i == len(ticket)-1):
                new = ticket[i] + new
            else:
                if(ticket[i+1] >= '0' and ticket[i+1] <= '9'):
                    new = ticket[i] + new
        i-=1
    trainNumeric.set_value(ct, 'Ticket', new)
    ct+=1
trainNumeric

#make cabins to labels
cabins = trainNumeric['Cabin']

ct = 0
for cabin in cabins:
    if(not isinstance(cabin, str)):
        print('nan')
    else:
        i = len(cabin)-1
        while i >= 0:
            if('A' == cabin[i]):
                trainNumeric.set_value(ct, 'Cabin', 'A')
            elif('B' == cabin[i]):
                trainNumeric.set_value(ct, 'Cabin', 'B')
            elif('C' == cabin[i]):
                trainNumeric.set_value(ct, 'Cabin', 'C')
        i-=1