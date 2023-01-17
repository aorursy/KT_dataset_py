

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import sklearn

import random



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Common Model Algorithms

from sklearn import svm, ensemble

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix



#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data_cleaner = [train, test]

train_copy = train.copy(deep=True)
print(train.info())

train.sample(5)
#CORRELATION ON DATA
print('Train'.center(30,'='))

print(train.corr())

print('Test'.center(30,'='))

print(test.corr())
print('Train'.center(30,'+'))

print('Train columns with null values:\n', train.isnull().sum())



print('Test'.center(30,'+'))

print('Test/Validation columns with null values:\n', test.isnull().sum())



train.describe(include = 'all')
for dataset in data_cleaner:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

#delete the cabin feature/column and others previously stated to exclude in train dataset

drop_column = ['PassengerId','Cabin', 'Ticket']

train.drop(drop_column, axis=1, inplace = True)



print(train.isnull().sum())

print()

print(test.isnull().sum())
###CREATE: Feature Engineering for train and test/validation dataset

for dataset in data_cleaner:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



    #quick and dirty code split title from name

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



    #Continuous variable bins; qcut vs cut

    #Fare Bins/Buckets using qcut or frequency bins

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



    #Age Bins/Buckets using cut or value bins

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    

#cleanup rare title names

#print(data1['Title'].value_counts())

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (train['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(train['Title'].value_counts())

print("-"*10)



#preview data again

train.info()

test.info()

train.sample(10)
#code categorical data

label = LabelEncoder()

for dataset in data_cleaner:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])





#define y variable aka target/outcome

Target = ['Survived']



#define x variables for original features aka feature selection

train_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

train_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

train_xy =  Target + train_x

print('Original X Y: ', train_xy, '\n')





#define x variables for original w/bin features to remove continuous variables

train_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

train_xy_bin = Target + train_x_bin

print('Bin X Y: ', train_xy_bin, '\n')





#define x and y variables for dummy features original

train_dummy = pd.get_dummies(train[train_x])

train_x_dummy = train_dummy.columns.tolist()

train_xy_dummy = Target + train_x_dummy

print('Dummy X Y: ', train_xy_dummy, '\n')





#print(data1_x_bin.head())

train_dummy.head()
"""

groupby関数がやっていることはただのグループ分けで、その後の処理は我々の方で自由に設定可能。

"""



for x in train_x:

    if train[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(train[[x, Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')

        



print(pd.crosstab(train['Title'],train[Target[0]]))
#graph distribution of quantitative data

plt.figure(figsize=[16,12])



"""

o is treated as a Outlier.

minimun

25パーセンタイル	第一四分位数

50パーセンタイル	第二四分位数（中央値）

75パーセンタイル	第三四分位数

maximum

"""

l = ['Fare','Age','FamilySize']

c = 230

for i in l:

    c += 1

    plt.subplot(c)

    plt.boxplot(x=train[i], showmeans = True, meanline = True)

    plt.title('{0} Boxplot'.format(i))

    plt.ylabel('{0}'.format(i))



for i in l:

    c += 1

    plt.subplot(c)

    plt.hist(x = [train[train['Survived']==1][i], train[train['Survived']==0][i]], 

             stacked=True, color = ['g','r'],label = ['Survived','Dead'])

    plt.title('{} Histogram by Survival'.format(i))

    plt.xlabel('{}'.format(i))

    plt.ylabel('# of Passengers')

    plt.legend()
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

l1 = ['Embarked','Pclass','IsAlone']

l2 = ['FareBin','AgeBin','FamilySize']

c=0

for i in l1:

    sns.barplot(x = i, y = 'Survived', data=train, ax = saxis[0,c])

    c +=1

c=0

for i in l2:

    sns.pointplot(x = i, y = 'Survived',  data=train, ax = saxis[1,c])

    c += 1
#graph distribution of qualitative data: Pclass

#we know class mattered in survival, now let's compare class and a 2nd feature

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = train, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = train, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
#graph distribution of qualitative data: Sex

#we know sex mattered in survival, now let's compare sex and a 2nd feature

fig, qaxis = plt.subplots(1,3,figsize=(14,12))

l = ['Embarked','Pclass','IsAlone']

c = 0

for i in l:

    sns.barplot(x = 'Sex', y = 'Survived', hue = i, data=train, ax = qaxis[c])

    axis1.set_title('Sex vs {} Survival Comparison'.format(i))

    c += 1
#more side-by-side comparisons

fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))



#how does family size factor with sex & survival compare



l = ['FamilySize','Pclass']

for i,m in zip(l,[maxis1, maxis2]):

    sns.pointplot(x=i, y="Survived", hue="Sex", data=train,

                  palette={"male": "blue", "female": "pink"},

                  markers=["*", "o"], linestyles=["-", "--"], ax = m)
e = sns.FacetGrid(train, col = 'Embarked')

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')

e.add_legend()
a = sns.FacetGrid( train, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , train['Age'].max()))

a.add_legend()
h = sns.FacetGrid(train, row = 'Sex', col = 'Pclass', hue = 'Survived')

h.map(plt.hist, 'Age', alpha = .75)

h.add_legend()
#pair plots of entire dataset

pp = sns.pairplot(train, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

pp.set(xticklabels=[])
#correlation heatmap of dataset

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



correlation_heatmap(train)
corr_matrix = train.corr()



fig.axis = plt.subplots(figsize=(16,12))

y = pd.DataFrame(corr_matrix['Survived'].sort_values(ascending=False))

sns.barplot(x = y.index, y = 'Survived', data=y)
train.hist()
#male female

male_df = train[train.Sex == 'male']

female_df = train[train.Sex == 'female']



male_age_survived_ratio_list = []

female_age_survived_ratio_list = []

for i in range(0, int(max(train.Age))+1, 10):

    male_df_of_age = male_df[(male_df.Age >= i) & (male_df.Age < i+9)]

    female_df_of_age = female_df[(female_df.Age >= i) & (female_df.Age < i+9)]



    male_s = len(male_df_of_age[male_df_of_age.Survived == 1])

    female_s = len(female_df_of_age[female_df_of_age.Survived == 1])



    male_total = len(male_df_of_age)

    female_total = len(female_df_of_age)



    if male_total  == 0:

        male_age_survived_ratio_list.append(0.5)

    else:

        male_age_survived_ratio_list.append(male_s/male_total)



    if female_total == 0:

        female_age_survived_ratio_list.append(0.5)

    else:

        female_age_survived_ratio_list.append(female_s/female_total)



print(male_age_survived_ratio_list, female_age_survived_ratio_list)



x_labels = []

for i in range(0, int(max(train.Age))+1, 10):

    x_labels.append(str(i) + '-' + str(i+9))



plt.figure(figsize=(16,8))

x1 = [i for i in range(0, int(max(train.Age))+ 1, 10)]

x2 = [i + 2 for i in range(0, int(max(train.Age))+ 1, 10)]

plt.bar(x1, male_age_survived_ratio_list, tick_label=x_labels, width=3, color='blue')

plt.bar(x2,female_age_survived_ratio_list, tick_label=x_labels, width=3, color='red')

plt.tick_params(labelsize = 15)
#male female

p1_df = train[train.Pclass == 1]

p2_df = train[train.Pclass == 2]

p3_df = train[train.Pclass == 3]

print(p1_df.head())

#Devided total???

l1,l2,l3 = len(p1_df),len(p2_df),len(p3_df)

print(l1,l2,l3)

p1_age_survived_ratio_list = []

p2_age_survived_ratio_list = []

p3_age_survived_ratio_list = []

for i in range(0, int(max(train.Age))+1, 10):

    p1_df_of_age = p1_df[(p1_df.Age >= i) & (p1_df.Age < i+9)]

    p2_df_of_age = p2_df[(p2_df.Age >= i) & (p2_df.Age < i+9)]

    p3_df_of_age = p3_df[(p3_df.Age >= i) & (p3_df.Age < i+9)]



    p1_s = len(p1_df_of_age[p1_df_of_age.Survived == 1]) / l1

    p2_s = len(p2_df_of_age[p2_df_of_age.Survived == 1]) / l2

    p3_s = len(p3_df_of_age[p3_df_of_age.Survived == 1]) / l3



    p1_total = len(p1_df_of_age)

    p2_total = len(p2_df_of_age)

    p3_total = len(p3_df_of_age)



    if p1_total  == 0:

        p1_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p1_age_survived_ratio_list.append(p1_s/(p1_s+p2_s+p3_s))

        except:

            p1_age_survived_ratio_list.append(0)

            

    if p2_total  == 0:

        p2_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p2_age_survived_ratio_list.append(p2_s/(p1_s+p2_s+p3_s))

        except:

            p2_age_survived_ratio_list.append(0)

            

    if p3_total  == 0:

        p3_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p3_age_survived_ratio_list.append(p3_s/(p1_s+p2_s+p3_s))

        except:

            p3_age_survived_ratio_list.append(0)

        



print(p1_age_survived_ratio_list, p2_age_survived_ratio_list, p3_age_survived_ratio_list)



x_labels = []

for i in range(0, int(max(train.Age))+1, 10):

    x_labels.append(str(i) + '-' + str(i+9))



plt.figure(figsize=(16,8))

x1 = [i for i in range(0, int(max(train.Age))+ 1, 10)]

x2 = [i + 2 for i in range(0, int(max(train.Age))+ 1, 10)]

x3 = [i + 4 for i in range(0, int(max(train.Age))+ 1, 10)]

plt.bar(x1, p1_age_survived_ratio_list, tick_label=x_labels, width=3, color='blue')

plt.bar(x2,p2_age_survived_ratio_list, tick_label=x_labels, width=3, color='red')

plt.bar(x3,p3_age_survived_ratio_list, tick_label=x_labels, width=3, color='green')

plt.tick_params(labelsize = 15)
#male female

p1_df = train[train.Pclass == 1]

p2_df = train[train.Pclass == 2]

p3_df = train[train.Pclass == 3]

print(p1_df.head())

#Devided total???

l1,l2,l3 = len(p1_df),len(p2_df),len(p3_df)

print(l1,l2,l3)

p1_age_survived_ratio_list = []

p2_age_survived_ratio_list = []

p3_age_survived_ratio_list = []

for i in range(0, int(max(train.Age))+1, 10):

    p1_df_of_age = p1_df[(p1_df.Age >= i) & (p1_df.Age < i+9)]

    p2_df_of_age = p2_df[(p2_df.Age >= i) & (p2_df.Age < i+9)]

    p3_df_of_age = p3_df[(p3_df.Age >= i) & (p3_df.Age < i+9)]



    p1_s = len(p1_df_of_age[p1_df_of_age.Survived == 1])

    p2_s = len(p2_df_of_age[p2_df_of_age.Survived == 1])

    p3_s = len(p3_df_of_age[p3_df_of_age.Survived == 1])



    p1_total = len(p1_df_of_age)

    p2_total = len(p2_df_of_age)

    p3_total = len(p3_df_of_age)



    if p1_total  == 0:

        p1_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p1_age_survived_ratio_list.append(p1_s/(p1_s+p2_s+p3_s))

        except:

            p1_age_survived_ratio_list.append(0)

            

    if p2_total  == 0:

        p2_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p2_age_survived_ratio_list.append(p2_s/(p1_s+p2_s+p3_s))

        except:

            p2_age_survived_ratio_list.append(0)

            

    if p3_total  == 0:

        p3_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p3_age_survived_ratio_list.append(p3_s/(p1_s+p2_s+p3_s))

        except:

            p3_age_survived_ratio_list.append(0)

        



print(p1_age_survived_ratio_list, p2_age_survived_ratio_list, p3_age_survived_ratio_list)



x_labels = []

for i in range(0, int(max(train.Age))+1, 10):

    x_labels.append(str(i) + '-' + str(i+9))



plt.figure(figsize=(16,8))

x1 = [i for i in range(0, int(max(train.Age))+ 1, 10)]

x2 = [i + 2 for i in range(0, int(max(train.Age))+ 1, 10)]

x3 = [i + 4 for i in range(0, int(max(train.Age))+ 1, 10)]

plt.bar(x1, p1_age_survived_ratio_list, tick_label=x_labels, width=3, color='blue')

plt.bar(x2,p2_age_survived_ratio_list, tick_label=x_labels, width=3, color='red')

plt.bar(x3,p3_age_survived_ratio_list, tick_label=x_labels, width=3, color='green')

plt.tick_params(labelsize = 15)
#male female

p1_df = train[train.Embarked == 'S']

p2_df = train[train.Embarked == 'C']

p3_df = train[train.Embarked == 'Q']

print(p1_df.head())

l1,l2,l3 = len(p1_df),len(p2_df),len(p3_df)

print(l1,l2,l3)

p1_age_survived_ratio_list = []

p2_age_survived_ratio_list = []

p3_age_survived_ratio_list = []

for i in range(0, int(max(train.Age))+1, 10):

    p1_df_of_age = p1_df[(p1_df.Age >= i) & (p1_df.Age < i+9)]

    p2_df_of_age = p2_df[(p2_df.Age >= i) & (p2_df.Age < i+9)]

    p3_df_of_age = p3_df[(p3_df.Age >= i) & (p3_df.Age < i+9)]



    p1_s = len(p1_df_of_age[p1_df_of_age.Survived == 1]) / l1

    p2_s = len(p2_df_of_age[p2_df_of_age.Survived == 1]) / l2

    p3_s = len(p3_df_of_age[p3_df_of_age.Survived == 1]) / l3



    p1_total = len(p1_df_of_age)

    p2_total = len(p2_df_of_age)

    p3_total = len(p3_df_of_age)

    

    if p1_total  == 0:

        p1_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p1_age_survived_ratio_list.append(p1_s/(p1_s+p2_s+p3_s))

        except:

            p1_age_survived_ratio_list.append(0)

            

    if p2_total  == 0:

        p2_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p2_age_survived_ratio_list.append(p2_s/(p1_s+p2_s+p3_s))

        except:

            p2_age_survived_ratio_list.append(0)

            

    if p3_total  == 0:

        p3_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p3_age_survived_ratio_list.append(p3_s/(p1_s+p2_s+p3_s))

        except:

            p3_age_survived_ratio_list.append(0)

        



print(p1_age_survived_ratio_list, p2_age_survived_ratio_list, p3_age_survived_ratio_list)



x_labels = []

for i in range(0, int(max(train.Age))+1, 10):

    x_labels.append(str(i) + '-' + str(i+9))



plt.figure(figsize=(16,8))

x1 = [i for i in range(0, int(max(train.Age))+ 1, 10)]

x2 = [i + 2 for i in range(0, int(max(train.Age))+ 1, 10)]

x3 = [i + 4 for i in range(0, int(max(train.Age))+ 1, 10)]

plt.bar(x1, p1_age_survived_ratio_list, tick_label=x_labels, width=3, color='blue')

plt.bar(x2,p2_age_survived_ratio_list, tick_label=x_labels, width=3, color='red')

plt.bar(x3,p3_age_survived_ratio_list, tick_label=x_labels, width=3, color='green')

plt.tick_params(labelsize = 15)

print(len(p1_df),len(p2_df),len(p3_df))
#male female

p1_df = train[train.Embarked == 'S']

p2_df = train[train.Embarked == 'C']

p3_df = train[train.Embarked == 'Q']

print(p1_df.head())

l1,l2,l3 = len(p1_df),len(p2_df),len(p3_df)

print(l1,l2,l3)

p1_age_survived_ratio_list = []

p2_age_survived_ratio_list = []

p3_age_survived_ratio_list = []

for i in range(0, int(max(train.Age))+1, 10):

    p1_df_of_age = p1_df[(p1_df.Age >= i) & (p1_df.Age < i+9)]

    p2_df_of_age = p2_df[(p2_df.Age >= i) & (p2_df.Age < i+9)]

    p3_df_of_age = p3_df[(p3_df.Age >= i) & (p3_df.Age < i+9)]



    p1_s = len(p1_df_of_age[p1_df_of_age.Survived == 1])

    p2_s = len(p2_df_of_age[p2_df_of_age.Survived == 1])

    p3_s = len(p3_df_of_age[p3_df_of_age.Survived == 1])



    p1_total = len(p1_df_of_age)

    p2_total = len(p2_df_of_age)

    p3_total = len(p3_df_of_age)

    

    if p1_total  == 0:

        p1_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p1_age_survived_ratio_list.append(p1_s/(p1_s+p2_s+p3_s))

        except:

            p1_age_survived_ratio_list.append(0)

            

    if p2_total  == 0:

        p2_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p2_age_survived_ratio_list.append(p2_s/(p1_s+p2_s+p3_s))

        except:

            p2_age_survived_ratio_list.append(0)

            

    if p3_total  == 0:

        p3_age_survived_ratio_list.append(0.33333)

    else:

        try:

            p3_age_survived_ratio_list.append(p3_s/(p1_s+p2_s+p3_s))

        except:

            p3_age_survived_ratio_list.append(0)

        



print(p1_age_survived_ratio_list, p2_age_survived_ratio_list, p3_age_survived_ratio_list)



x_labels = []

for i in range(0, int(max(train.Age))+1, 10):

    x_labels.append(str(i) + '-' + str(i+9))



plt.figure(figsize=(16,8))

x1 = [i for i in range(0, int(max(train.Age))+ 1, 10)]

x2 = [i + 2 for i in range(0, int(max(train.Age))+ 1, 10)]

x3 = [i + 4 for i in range(0, int(max(train.Age))+ 1, 10)]

plt.bar(x1, p1_age_survived_ratio_list, tick_label=x_labels, width=3, color='blue')

plt.bar(x2,p2_age_survived_ratio_list, tick_label=x_labels, width=3, color='red')

plt.bar(x3,p3_age_survived_ratio_list, tick_label=x_labels, width=3, color='green')

plt.tick_params(labelsize = 15)

print(len(p1_df),len(p2_df),len(p3_df))
#handmade data model using brain power (and Microsoft Excel Pivot Tables for quick calculations)

def mytree(df):

    

    #initialize table to store predictions

    Model = pd.DataFrame(data = {'Predict':[]})

    male_title = ['Master'] #survived titles



    for index, row in df.iterrows():



        #Question 1: Were you on the Titanic; majority died

        Model.loc[index, 'Predict'] = 0



        #Question 2: Are you female; majority survived

        if (df.loc[index, 'Sex'] == 'female'):

                  Model.loc[index, 'Predict'] = 1



        #Question 3A Female - Class and Question 4 Embarked gain minimum information



        #Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0       

        if ((df.loc[index, 'Sex'] == 'female') & 

            (df.loc[index, 'Pclass'] == 3) & 

            (df.loc[index, 'Embarked'] == 'S')  &

            (df.loc[index, 'Fare'] > 8)



           ):

                  Model.loc[index, 'Predict'] = 0



        #Question 3B Male: Title; set anything greater than .5 to 1 for majority survived

        if ((df.loc[index, 'Sex'] == 'male') &

            (df.loc[index, 'Title'] in male_title)

            ):

            Model.loc[index, 'Predict'] = 1

        

        #Question : FamilySize

        if df.loc[index, 'FamilySize'] > 4:

            Model.loc[index, 'Predict'] = 0

        

        if ((df.loc[index, 'Pclass'] == 1) & 

            (df.loc[index, 'Embarked'] == 'S')  &

            (df.loc[index, 'Age'] > 70)



           ):

                Model.loc[index, 'Predict'] = 1



            

        if ((df.loc[index, 'Pclass'] == 1) & 

            (df.loc[index, 'Embarked'] == 'C')  &

            (df.loc[index, 'Sex'] > 'male')



           ):

                Model.loc[index, 'Predict'] = 1

        if ((df.loc[index, 'Pclass'] == 2) & 

            (df.loc[index, 'Embarked'] == 'C')  &

            (df.loc[index, 'Sex'] > 'male')



           ):

                Model.loc[index, 'Predict'] = 1



        if ((df.loc[index, 'Pclass'] == 1) & 

            (df.loc[index, 'Embarked'] == 'Q')  &

            (df.loc[index, 'Sex'] > 'male')



           ):

                Model.loc[index, 'Predict'] = 0



        if ((df.loc[index, 'Age'] >= 60) & 

            (df.loc[index, 'Age'] == 69)  &

            (df.loc[index, 'Sex'] > 'female')



           ):

                Model.loc[index, 'Predict'] = 0

    return Model





#model data

Tree_Predict = mytree(train)

print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(train['Survived'], Tree_Predict)*100))



#Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score

#And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

print(metrics.classification_report(train['Survived'], Tree_Predict))



"""

Previous(Online)

Decision Tree Model Accuracy/Precision Score: 82.04%



             precision    recall  f1-score   support



          0       0.82      0.91      0.86       549

          1       0.82      0.68      0.75       342



avg / total       0.82      0.82      0.82       891

"""
test['Survived'] = mytree(test).astype(int)

submit = test[['PassengerId','Survived']]

submit.to_csv("../working/new_handmade_submit.csv", index=False)