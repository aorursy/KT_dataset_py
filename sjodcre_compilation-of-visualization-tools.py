import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

#from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



# Load dataset.

dataset = pd.read_csv('../input/titanic/train.csv')



#fill NaN values in the age column with the median of that column

dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



#fill NaN values in the embarked column with the mode of that column

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



#fill NaN values in the fare column with the median of that column

dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)



#delete the cabin feature/column and others 

drop_column = ['PassengerId','Cabin', 'Ticket']

dataset.drop(drop_column, axis=1, inplace = True)



#create a new column which is the combination of the sibsp and parch column

dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



#create a new column and initialize it with 1

dataset['IsAlone'] = 1 #initialize to yes/1 is alone

dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



#quick and dirty code split title from the name column

dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)



#so create stat_min and any titles less than 10 will be put into Misc category

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (dataset['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



#convertion from categorical data to dummy variables

label = LabelEncoder()  

dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

dataset['Title_Code'] = label.fit_transform(dataset['Title'])

dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])



#define y variable aka target/outcome

Target = ['Survived']
dataset.columns.values
dataset.info()
dataset.describe(include = 'all')
dataset.isnull().sum()
dataset.sample(10)
dataset.head()
dataset.tail()
pd.crosstab(dataset['Title'],dataset[Target[0]])
dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#to organize our graphics will use figure: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure

#subplot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot

#and subplotS: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=matplotlib%20pyplot%20subplots#matplotlib.pyplot.subplots



#graph distribution of quantitative data

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=dataset['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(dataset['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(dataset['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [dataset[dataset['Survived']==1]['Fare'], dataset[dataset['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [dataset[dataset['Survived']==1]['Age'], dataset[dataset['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [dataset[dataset['Survived']==1]['FamilySize'], dataset[dataset['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
#we will use seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html



#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=dataset, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=dataset, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=dataset, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=dataset, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=dataset, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=dataset, ax = saxis[1,2])
#graph distribution of qualitative data: Pclass

#we know class mattered in survival, now let's compare class and a 2nd feature

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = dataset, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = dataset, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = dataset, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
#graph distribution of qualitative data: Sex

#we know sex mattered in survival, now let's compare sex and a 2nd feature

fig, qaxis = plt.subplots(1,3,figsize=(14,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=dataset, ax = qaxis[0])

axis1.set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=dataset, ax  = qaxis[1])

axis1.set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=dataset, ax  = qaxis[2])

axis1.set_title('Sex vs IsAlone Survival Comparison')
#more side-by-side comparisons

fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))



#how does family size factor with sex & survival compare

sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=dataset,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)



#how does class factor with sex & survival compare

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=dataset,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
#how does embark port factor with class, sex, and survival compare

#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

e = sns.FacetGrid(dataset, col = 'Embarked')

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')

e.add_legend()
#plot distributions of age of passengers who survived or did not survive

a = sns.FacetGrid( dataset, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , dataset['Age'].max()))

a.add_legend()
#histogram comparison of sex, class, and age by survival

h = sns.FacetGrid(dataset, row = 'Sex', col = 'Pclass', hue = 'Survived')

h.map(plt.hist, 'Age', alpha = .75)

h.add_legend()
#pair plots of entire dataset

pp = sns.pairplot(dataset, hue = 'Survived', palette = 'deep', height=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

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

    

correlation_heatmap(dataset)