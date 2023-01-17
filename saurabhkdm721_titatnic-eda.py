# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn import svm,tree,linear_model,neighbors,naive_bayes,ensemble,discriminant_analysis,gaussian_process



from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

#from pandas.tools.plotting import scatter_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_raw=pd.read_csv('/kaggle/input/titanic/train.csv')

data_val=pd.read_csv('/kaggle/input/titanic/test.csv')

data1=data_raw.copy(deep=True)

data_cleaner=[data1,data_val]

print(data_raw.info())

print('data with nul value',data1.isnull().sum())

print('Test/Validation colmns with null',data_val.isnull().sum())





for dataset in data_cleaner:

    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

drop_cols=['PassengerId','Cabin','Ticket']

data1.drop(drop_cols,axis=1,inplace=True)

print(data1.isnull().sum())





for dataset in data_cleaner:

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1

    dataset['IsAlone']=1

    dataset['IsAlone'].loc[dataset['FamilySize']>1]=0

    dataset['Title']=dataset['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]

    dataset['FareBin']=pd.qcut(dataset['Fare'],4)

    dataset['Agebin']=pd.cut(dataset['Age'].astype(int),5)

stat_min=10

title_names=(data1['Title'].value_counts()<stat_min)

data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(data1['Title'].value_counts())

print(data1.info())



label=LabelEncoder()

for dataset in data_cleaner:

    dataset['Sex_Code']=label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code']=label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['Agebin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

Target=['Survived']

data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']

data1_xy=Target+data1_x

#define x variables for original w/bin features to remove continuous variables

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

data1_xy_bin = Target + data1_x_bin

print('Bin X Y: ', data1_xy_bin, '\n')





#define x and y variables for dummy features original

data1_dummy = pd.get_dummies(data1[data1_x])

data1_x_dummy = data1_dummy.columns.tolist()

data1_xy_dummy = Target + data1_x_dummy

print('Dummy X Y: ', data1_xy_dummy, '\n')

print(data1_dummy.head())

train1_x,test1_x,train1_y,test1_y=model_selection.train_test_split(data1[data1_x_calc],data1[Target],random_state=0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)



print("Data1 Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))



train1_x_bin.head()



for x in data1_x:

    if data1[x].dtype!='float64':

        print('Survival Correlation by',x)

        print(data1[[x,Target[0]]].groupby(x,as_index=False).mean())

        print('-'*10,'\n')

print(data1)

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])

sns.pointplot(x = 'Agebin', y = 'Survived',  data=data1, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])

# Any results you write to the current directory are saved as output.