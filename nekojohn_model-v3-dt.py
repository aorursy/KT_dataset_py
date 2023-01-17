# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

pd.options.display.max_columns = None

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.name='train'
print(train.head(10))
test = pd.read_csv('../input/test.csv')
test.name='test'
print(test.head(10))
datasets = [train, test]

for data in datasets:
    print(data.name)
    print(data.shape)
    print(data.isnull().sum())
    print(data.info())
    print('A'*90)
    print(data.describe())
    print('B'*90)
    print(data.describe(include=['O']))
    #print('C'*90)
    #print(data.describe(include='all')) 
# handle missing values
# Embarked
print('{} before data cleaning {}'.format('*'*20,'*'*20))
for data in datasets:
    print(data.groupby(['Embarked']).size())
    
print('{} after data cleaning {}'.format('*'*20,'*'*20))
for data in datasets:
    #print(data['Embarked'].mode()[0]) #"S"
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
    data['Age'].fillna(data['Age'].mean(),inplace=True)
    data['Fare'].fillna(data['Fare'].mode()[0],inplace=True)
    data['FamilySize'] = data ['SibSp'] + data['Parch'] + 1
    data['Alone'] = np.where(data.FamilySize>1,0,1)
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    data['CabinCode'] = data['Cabin'].str[0]
    print(data.groupby(['Embarked']).size())
    print(data.groupby(['FamilySize']).size())
    print(data.groupby(['Alone']).size())
    print(data.groupby(['Title']).size())
    print(data.groupby(['CabinCode']).size())
    # print(data.groupby(['Ticket']).size()) # probably no use
# plot CabinCode
for data in [train]:
    plt_obj = data.groupby(['CabinCode'],as_index=False).mean()
    plt.subplot(2,1,1)
    sns.barplot(x='CabinCode',y='Fare',data=plt_obj)
    plt.subplot(2,1,2)
    sns.boxplot(x='CabinCode',y='Fare',data=data)
print(train[~train.Age.isnull()]['Age'])
# plot 
plt.figure(figsize=[16,12])
plt.subplot(2,3,1)
# fare box plot
plt.boxplot(train['Fare'],showmeans=True, meanline=True)

plt.subplot(2,3,4)
# fare by survived histogram
plt.hist(x=[train[train.Survived==1]['Fare'],train[train.Survived==0]['Fare']],stacked=True,color=['g','r'],label=['Survived','Dead'])
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(2,3,2)
# fare box plot
plt.boxplot(train[~train.Age.isnull()]['Age'],showmeans=True, meanline=True)

plt.subplot(2,3,5)
# fare by survived histogram
plt.hist(x=[train[(train.Survived==1)&(~train.Age.isnull())]['Age'],train[(train.Survived==0)&(~train.Age.isnull())]['Age']],stacked=True,color=['g','r'],label=['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(2,3,3)
# fare box plot
plt.boxplot(train[~train.FamilySize.isnull()]['FamilySize'],showmeans=True, meanline=True)

plt.subplot(2,3,6)
# fare by survived histogram
plt.hist(x=[train[(train.Survived==1)&(~train.FamilySize.isnull())]['FamilySize'],train[(train.Survived==0)&(~train.FamilySize.isnull())]['FamilySize']],stacked=True,color=['g','r'],label=['Survived','Dead'])
plt.xlabel('FamilySize')
plt.ylabel('# of Passengers')
plt.legend()
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=train, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train, ax = saxis[0,1])
sns.barplot(x = 'Alone', y = 'Survived', order=[1,0], data=train, ax = saxis[0,2])

sns.barplot(x = 'FamilySize', y = 'Survived', data=train,ax = saxis[1,0])
sns.barplot(x = 'Sex', y = 'Survived', data=train,ax = saxis[1,1])
# code categorical data
label = LabelEncoder()
for data in datasets:    
    data['SexCode'] = label.fit_transform(data['Sex']) # same results with data['sex_code'] = np.where(data['Sex']=='male',1,0)
    data['EmbCode'] = label.fit_transform(data['Embarked'])
    data['TitleCode'] = label.fit_transform(data['Title'])
    data['TitleCode'] = label.fit_transform(data['Title'])
    print(data.groupby(['EmbCode']).size())
    #print(data.groupby(['sex_code']).size())
    print(data.groupby(['Embarked']).size())
features = ['SexCode','EmbCode','TitleCode','Pclass','Fare','Age','FamilySize','Alone']

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train[~train.Age.isnull()][features])

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

seed = 7
test_size = 0.2
Y = train['Survived']
features = ['SexCode','EmbCode','TitleCode','Pclass','Fare','Age','FamilySize']
X = train[features]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
from graphviz import Source
from sklearn import tree
from IPython.display import SVG
graph = Source( tree.export_graphviz(model, out_file=None, feature_names=features))
SVG(graph.pipe(format='svg'))
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('accuracy =', accuracy)
print('population bad rate=',np.mean(y_test))
test_x = test[features]
pred_test = model.predict(test_x)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred_test
    })
submission.to_csv('titanic_v3_DT.csv', index=False)
