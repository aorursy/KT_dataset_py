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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV



# To ignore unwanted warnings

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



# Reading Data from CSV to DF - Jupyter

# train = pd.read_csv('data/train.csv')

# test = pd.read_csv('data/test.csv')

# submission = pd.read_csv('data/gender_submission.csv')
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='RdBu_r')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='RdBu_r')

ax[1].set_title('Test data');
train.isnull().sum()
test.isnull().sum()
#print number of females vs. males that survive

print("females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts())



print(" males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts())
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Male', 'Female'

sizes = [109, 233]

explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
for x in [1,2,3]:

    train.Survived[train.Pclass == x].plot(kind="kde")

plt.title("class wrt Survived")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.legend(("1st","2nd","3rd"))



plt.show
sns.barplot(x="Parch", y="Survived", data=train,palette="Blues")

plt.title('Parch Distribution by survived', fontsize=14)
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))

sns.set_palette(sns.color_palette(('RdBu_r')))

sns.barplot(train[ 'Pclass'], train['Survived'], ax=ax[0,0])

sns.barplot(train['Sex'], train[ 'Survived'], ax=ax[0,1]) 

sns.distplot(train['Age'],bins=24,ax=ax[1,0])

sns.distplot(train['Fare'],bins=24, ax=ax[1,1])

ax[0,0].set_title('The number of survived by Pclass', fontsize=10)

ax[0,1].set_title('The number of survived by Sex', fontsize=10)

ax[1,0].set_title('The number of survived by Age', fontsize=10)

ax[1,1].set_title('The number of survived by Fare', fontsize=10)

plt.show()
train.Embarked.value_counts()
test.Embarked.value_counts()
y = train['Survived']

train.drop('Survived', axis=1, inplace=True)

train.head()
test.set_index('PassengerId',inplace=True)

train.set_index('PassengerId',inplace=True)
test.head()
data_merge = pd.merge(train ,test,how='outer',left_index=False, right_index=False)
data_merge.head()
# Split the name column to two different columns (last_name & title)

data_merge['last_name'] = data_merge['Name'].apply(lambda x: x.split(',')[0])

data_merge['title'] = data_merge['Name'].apply(lambda x: x.split('.')[0].split(',')[1])



# We will also add parch and sibsp to get total of family memebers 

data_merge['family_size'] = data_merge['SibSp'] + data_merge['Parch'] + 1
data_merge['Embarked'] = data_merge['Embarked'].fillna('S')
data_merge['Fare'] = data_merge['Fare'].fillna(data_merge.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0])
data_merge['Age'] = data_merge.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
bins = [0, 5, 13, 19, 30, 60, 80]

labels = ['Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

data_merge['AgeGroup'] = pd.cut(data_merge["Age"], bins, labels = labels)
data_merge.drop('Age', axis = 1, inplace=True)
data_merge.drop('Name', axis=1, inplace=True)
data_merge.drop('SibSp', axis = 1, inplace=True)

data_merge.drop('Parch', axis = 1, inplace=True)
data_merge['Cabin'] = data_merge['Cabin'].fillna('M')
ax = plt.axes()

sns.heatmap(data_merge.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='RdBu_r')

ax.set_title('Merga DF');
data_merge['Cabin'] = data_merge['Cabin'].apply(lambda x: 'ABC' if x[0] in 'ABCT' else 'DE' if x[0] in 'DE' else 'FG' if x[0] in 'FG' else 'M')
data_merge
# DF used for plotting only

df_plot_train = data_merge[:891].copy()

# Seperating our target in a different DF

df_plot_train['Survived'] = y

df_plot_train
df_plot_train['title'] = df_plot_train['title'].apply(lambda x: x.strip(' '))

df_plot_train['title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms', inplace=True)

df_plot_train['title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy', inplace=True)
cat_features = ['title', 'Pclass', 'family_size','Cabin']



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(2, 2, i)

    sns.countplot(x=feature, hue='Survived', data=df_plot_train)

    

    plt.xlabel('{}'.format(feature), size=20)

    plt.ylabel('Passenger Count', size=20)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['UnSurvived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {}'.format(feature), size=20, y=1.05)



plt.show()
# Pie chart

labels = 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'

sizes = [44,27,93,392,313,22]

explode = (0, 0,0,0,0,0)  



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=20)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.tight_layout()

plt.show()
data_merge.drop('Ticket', axis = 1, inplace=True)

data_merge.drop('last_name', axis = 1, inplace=True)
# Changing Sex from male/female to 0/1

data_merge['Sex'] = data_merge['Sex'].map({'male':0, 'female':1})
data_merge_dummy = pd.get_dummies(data_merge, drop_first=True)
train_cleaned = data_merge_dummy[:891]

test_cleaned = data_merge_dummy[891:]
train_cleaned.tail()
test_cleaned.head()
X=train_cleaned
# Need to do some imports for modeling 

from IPython.display import Image  

from io import StringIO

from sklearn.tree import export_graphviz

import pydot # please install this package if you don't have it.

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn import svm

svm_m=svm.SVC()
# Fit Decision Trees

dt = DecisionTreeClassifier(max_depth = 6,random_state=8)

dt.fit(X,y)

# Getting model score

print('dt: ', dt.score(X,y))

# Print results 

s = cross_val_score(dt, X, y, cv=7 )

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Decision Tree", "Train", s.mean().round(3), s.std().round(3)))

print()





# Fit Bagging + Decision Trees

dt_bag = BaggingClassifier(max_features =16, n_estimators = 100,random_state=8)

dt_bag.fit(X,y)

# Getting model score

print('dt_bag: ', dt_bag.score(X,y))

# Print results 

s = cross_val_score(dt_bag, X, y, cv=7)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Decision Tree", "Train", s.mean().round(3), s.std().round(3)))

print()





# Fit Random Forest

rf = RandomForestClassifier(random_state=8)

rf.fit(X,y)

# Getting model score

print('rf: ', rf.score(X,y))

# Print results 

s = cross_val_score(rf, X, y, cv=7)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Decision Tree", "Train", s.mean().round(3), s.std().round(3)))

print()





# Fit Extra Trees

dt_et = ExtraTreesClassifier(random_state=8)

dt_et.fit(X,y)

# Getting model score

print('dt_et: ', dt_et.score(X,y))

# Print results 

s = cross_val_score(dt_et, X, y, cv=7)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Decision Tree", "Train", s.mean().round(3), s.std().round(3)))



# Fit SVM

svm_m.fit(X,y)

# Getting model score

print('svm_m: ', svm_m.score(X,y))

# Print results 

s = cross_val_score(svm_m, X, y, cv=7)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("svm_m", "Train", s.mean().round(3), s.std().round(3)))

print()
bag_pred = dt_bag.predict(test_cleaned)

bag_pred
submission['Survived'] = bag_pred

# Save CSV to path

submission.to_csv('submission_dt_bag.csv', index=False)

# Jupyter save to c

# submission.to_csv('data/submission_dt_bag_best_r, index=False)
dt_best = dt_bag.base_estimator_

dt_best.fit(X,y)
# Need to do some imports

from IPython.display import Image  

from io import StringIO

from sklearn.tree import export_graphviz

import pydot # please install this package if you don't have it.



dot_data = StringIO('/data/tree.dot')  

export_graphviz(decision_tree = dt_best,

                 out_file = dot_data,

                 feature_names = X.columns, 

                 filled=True, 

                 rounded=True) 



graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph[0].create_png()) 