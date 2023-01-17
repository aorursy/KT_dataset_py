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

# Set your own project id here

#PROJECT_ID = 'Titanic_Learning'

#from google.cloud import storage

#storage_client = storage.Client(project=PROJECT_ID)

#from google.cloud import bigquery

#bigquery_client = bigquery.Client(project=PROJECT_ID)'''
#读取数据

import pandas as pd

import numpy as np

train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

gender_data=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

full_data=[train_data,test_data]
#查看数据 搞清数据类型

#train_data.info()

#train_data.columns.values

#test_data.head()



train_data.tail()

#train_data.describe()

#train_data.describe(include=['O'])#统计categorical类型的特征，三个参数percentile，include，exclude，

#train_data.sample(6)

#import pandas_profiling

#train_data.profile_report()#强大的数据情况展示库

#train_data[train_data.Ticket.values=='113572']
#数据清洗，处理缺失值

#train_data.isnull().sum()

#Embarked的缺失值

train_data["Embarked"].value_counts() #S 最多，把两个缺失值用S代替

train_data[train_data.Embarked.isnull()] #查看缺失值

train_data.Embarked.fillna('C',inplace=True)#缺失值用C来代替，经过下边的可视化得来的

test_data.Embarked.fillna('C',inplace=True)
#对Cabin的处理不合适

#survivers=train_data.Survived

#train_data.drop(['Survived'],axis=1,inplace=True)

#all_data=pd.concat([train_data,test_data],ignore_index=False)

#all_data.Cabin.fillna('N',inplace=True)
'''all_data.Cabin=[i[0] for i in all_data.Cabin]#把船舱的首字母留下，舍去没有意义的数字

def cabin_estimator(i):

    """Grouping cabin feature by the first letter"""

    a = 0

    if i<16:

        a = "G"

    elif i>=16 and i<27:

        a = "F"

    elif i>=27 and i<38:

        a = "T"

    elif i>=38 and i<47:

        a = "A"

    elif i>= 47 and i<53:

        a = "E"

    elif i>= 53 and i<54:

        a = "D"

    elif i>=54 and i<116:

        a = 'C'

    else:

        a = "B"

    return a

with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]

##applying cabin estimator function. 

with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))

## getting back train. 

all_data = pd.concat([with_N, without_N], axis=0)

## PassengerId helps us separate train and test. 

all_data.sort_values(by = 'PassengerId', inplace=True)

## Separating train and test from all_data. 

train_data=all_data[:891]

test_data = all_data[891:]

# adding saved target variable with train. 

train_data['Survived'] = survivers'''
#all_data.Fare.isnull().sum() 就一个缺失值

#all_data[all_data.Fare.isnull()] 看看这一个缺失值长什么样子

missing_value = test_data[(test_data.Pclass == 3) & (test_data.Embarked == "S") & (test_data.Sex == "male")].Fare.mean()

## replace the test.fare null values with test.fare mean

test_data.Fare.fillna(missing_value, inplace=True)
###第一种方法，拥均值方差等来代替

#for dataset in full_data:

#    age_avg 	   = dataset['Age'].mean()

#   age_std 	   = dataset['Age'].std()

#    age_null_count = dataset['Age'].isnull().sum()

    

#    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

#    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

#    dataset['Age'] = dataset['Age'].astype(int)

    

#train['CategoricalAge'] = pd.cut(train['Age'], 5)

###第二种方法，用随机森林法则来推测年龄值 在下边的程序中进行展示
#数据可视化

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('darkgrid')

fig, ax = plt.subplots(figsize=(16,12),ncols=2)

ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_data, ax = ax[0]);

ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test_data, ax = ax[1]);

ax1.set_title("Training Set", fontsize = 18)

ax2.set_title('Test Set',  fontsize = 18)

# ## Fixing legends

# leg_1 = ax1.get_legend()

# leg_1.set_title("PClass")

# legs = leg_1.texts

# legs[0].set_text('Upper')

# legs[1].set_text('Middle')

# legs[2].set_text('Lower')

fig.show()  #（查看Embarked 和Pclass，Fare的关系，填充缺失值）
pal = {'male':"green", 'female':"Pink"}

sns.set(style="darkgrid")

plt.subplots(figsize = (15,8))

ax = sns.barplot(x = "Sex", 

                 y = "Survived", 

                 data=train_data, 

                 palette = pal,

                 linewidth=5,

                 order = ['female','male'],

                 capsize = .05)



plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25,loc = 'center')

plt.ylabel("% of passenger survived", fontsize = 15, )

plt.xlabel("Sex",fontsize = 15);
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(train_data.loc[(train_data['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')

ax=sns.kdeplot(train_data.loc[(train_data['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')

plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)

plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)

plt.xlabel("Fare", fontsize = 15, labelpad = 20);
#对Full_data做变化，train_data,test_data同步变化

full_data=[train_data,test_data]

for dataset in full_data:

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1

print(train_data[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean())

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)



print (train_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
import re

def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train_data['Title'], train_data['Sex']))
test_data.Cabin.value_counts()
# Mapping Cabin

    #dataset['Cabin'] = dataset['Cabin'].map( {'G':0,'F':1,'T':4,'A':6,'E':5,'D':7,'C':2,'B':3} ).astype(int)

#train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

#test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

for dataset in full_data:

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
drop_elements = ['PassengerId', 'Name','Ticket', 'SibSp','Cabin',\

                 'Parch', 'FamilySize']

train_data = train_data.drop(drop_elements, axis = 1)

train_data = train_data.drop(['CategoricalAge'], axis = 1)



test_data  = test_data.drop(drop_elements, axis = 1)



train_labels=train_data.Survived.values

train=train_data.drop('Survived',axis=1).values

test  = test_data.values
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression(),

    XGBClassifier()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = train

y = train_labels



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
candidate_classifier = XGBClassifier()

candidate_classifier.fit(train, train_labels)

result = candidate_classifier.predict(test)

StackingSubmission = pd.DataFrame({ 'PassengerId':gender_data.PassengerId.values,

                            'Survived': result })

StackingSubmission.to_csv("HS_Submission.csv", index=False)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train_data[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])