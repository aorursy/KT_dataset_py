

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session







# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
#Read data from csv file

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
print('The shape of our training set: ',df_train.shape[0], 'Passengers', 'and', df_train.shape[1] -1 , 'features' ,df_train.shape[1] , 'columns' )

print('The shape of our testing set: ',df_test.shape[0], 'Passengers', 'and', df_test.shape[1], 'features')

print('The testing set has 1 column  less than the training set, which is Survived , the target to predict  ')
print(df_train.columns.values)
# preview the data from head

df_train.head(3)
# preview the data from tail

df_train.tail(3)
# split data train into Numeric and Categorocal

numeric = df_train.select_dtypes(exclude='object')

categorical = df_train.select_dtypes(include='object')
print("\nNumber of categorical features : ",(len(categorical.axes[1])))

print("\n", categorical.axes[1])

categorical.head()
df_train.describe(include=['O'])
print("\nNumber of numeric features : ",(len(numeric.axes[1])))

print("\n", numeric.axes[1])
df_train.describe()
df_train.info()

print('_'*50)

df_test.info()
# Isolate the numeric features and check his relevance



num_corr = numeric.corr()

table = num_corr['Survived'].sort_values(ascending=False).to_frame()

cm = sns.light_palette("green", as_cmap=True)

tb = table.style.background_gradient(cmap=cm)

tb
#Survived correlation matrix

k = 5 #number of variables for heatmap

cols = df_train.corr().nlargest(k, 'Survived')['Survived'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
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

        linewidths=0.1,vmax=1.0, linecolor='blue',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(df_train)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#print('_'*50)
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)



df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


df_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
fig1, ax1 = plt.subplots()

ax1.pie(df_train['Survived'].groupby(df_train['Survived']).count(), 

        labels = ['Not Survived', 'Survived'], autopct = '%1.1f%%')

ax1.axis('equal')



plt.show()
g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Fare', bins=30)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(df_train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()




# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(df_train, row='Pclass', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# Countplot 



sns.catplot(x ="Sex", hue ="Survived", kind ="count", data = df_train) 
# Countplot 



sns.catplot(x ="Embarked", hue ="Survived", kind ="count", data = df_train) 
# Countplot 



sns.catplot(x ="Pclass", hue ="Survived", kind ="count", data = df_train) 
# Countplot 



sns.catplot(x ="SibSp", hue ="Survived", kind ="count", data = df_train) 
# Countplot 



sns.catplot(x ="Parch", hue ="Survived", kind ="count", data = df_train) 



plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data=df_train)

plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass',y='Fare',data=df_train)

plt.figure(figsize=(10,7))

sns.boxplot(x='Survived',y='Fare',data=df_train)

#Embarked

plt.figure(figsize=(10,7))

sns.boxplot(x='Embarked',y='Fare',data=df_train)

group = df_train.groupby(['Pclass', 'Survived']) 

pclass_survived = group.size().unstack() 

  

sns.heatmap(pclass_survived, annot = True, fmt ="d") 
group = df_train.groupby(['Embarked', 'Survived']) 

pclass_survived = group.size().unstack() 

  

sns.heatmap(pclass_survived, annot = True, fmt ="d") 
group = df_train.groupby(['Sex', 'Survived']) 

pclass_survived = group.size().unstack() 

  

sns.heatmap(pclass_survived, annot = True, fmt ="d")
#missing data in Traing examples

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(12)

print(df_train.shape)

print(df_test.shape)
na = df_train.shape[0] #na is the number of rows of the original training set

nb = df_test.shape[0]  #nb is the number of rows of the original test set

y_target = df_train['Survived'].to_frame()

#Combine train and test sets

c1 = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)

#Drop the target "Survived" and Id columns

c1.drop(['Survived'], axis=1, inplace=True)

c1.drop(['PassengerId'], axis=1, inplace=True)

print("Total size for train and test sets is :",c1.shape)
##msv1 method to visualize missing values per columns

def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3): 

    """

    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

    """

    

    plt.figure(figsize=(width,height))

    percentage=(data.isnull().mean())*100

    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)

    plt.axhline(y=thresh, color='r', linestyle='-')

    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )

    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, 'Columns with more than %s%s missing values' %(thresh, '%'), fontsize=12, color='crimson',

         ha='left' ,va='top')

    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, 'Columns with less than %s%s missing values' %(thresh, '%'), fontsize=12, color='green',

         ha='left' ,va='top')

    plt.xlabel('Columns', size=15, weight='bold')

    plt.ylabel('Missing values percentage')

    plt.yticks(weight ='bold')

    

    return plt.show()
#missing data in Traing examples and test set 

total = c1.isnull().sum().sort_values(ascending=False)

percent = (c1.isnull().sum()/c1.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(12)
#call method msv1 to visualization missing value per columns 

msv1(c1, 77, color=('silver', 'gold', 'lightgreen', 'skyblue', 'lightpink'))
# drop columns (features ) with > 79% missing vales

c=c1.dropna(thresh=len(c1)*0.15, axis=1)

print('We dropped ',c1.shape[1]-c.shape[1], ' features in the combined set')
print('The shape of the combined dataset after dropping features with more than 21% M.V.', c.shape)
allna = (c.isnull().sum() / len(c))*100

allna = allna.drop(allna[allna == 0].index).sort_values()





##msv2 method to visualize missing values per columns less than threshold 

def msv2(data, width=12, height=8, color=('silver', 'gold','lightgreen','skyblue','lightpink'), edgecolor='black'):

    """

    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

    """

    fig, ax = plt.subplots(figsize=(width, height))



    allna = (data.isnull().sum() / len(data))*100

    tightout= 0.008*max(allna)

    allna = allna.drop(allna[allna == 0].index).sort_values().reset_index()

    mn= ax.barh(allna.iloc[:,0], allna.iloc[:,1], color=color, edgecolor=edgecolor)

    ax.set_title('Missing values percentage per column', fontsize=15, weight='bold' )

    ax.set_xlabel('Percentage', weight='bold', size=15)

    ax.set_ylabel('Features with missing values', weight='bold')

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    for i in ax.patches:

        ax.text(i.get_width()+ tightout, i.get_y()+0.1, str(round((i.get_width()), 2))+'%',

            fontsize=10, fontweight='bold', color='grey')

    return plt.show()
#call method msv2 to visualization missing value per columns less than threshold

msv2(c)
NA=c[allna.index.to_list()]
NAcat=NA.select_dtypes(include='object')

NAnum=NA.select_dtypes(exclude='object')

print('We have :',NAcat.shape[1],'categorical features with missing values')

print('We have :',NAnum.shape[1],'numerical features with missing values')
NAnum.head()
NANUM= NAnum.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



NANUM = NANUM.style.background_gradient(cmap=cm)

NANUM
#complete missing age with median



c['Age']=c.Age.fillna(c.Age.median())



#complete missing Fare (ticket price) with median



c['Fare']=c.Age.fillna(c.Fare.median())

bb=c[allna.index.to_list()]

nan=bb.select_dtypes(exclude='object')

N= nan.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



N= N.style.background_gradient(cmap=cm)

N
NAcat.head()
NAcat1= NAcat.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



NAcat1 = NAcat1.style.background_gradient(cmap=cm)

NAcat1
#complete embarked with mode

    

c['Embarked'].fillna(c['Embarked'].mode()[0], inplace = True)
# Replace the Cabin number by the type of cabin 'X' if not

c["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in c['Cabin'] ])
FillNA=c[allna.index.to_list()]







FillNAcat=FillNA.select_dtypes(include='object')



FC= FillNAcat.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



FC= FC.style.background_gradient(cmap=cm)

FC
FillNAnum=FillNA.select_dtypes(exclude='object')



FM= FillNAnum.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



FM= FM.style.background_gradient(cmap=cm)

FM
c.isnull().sum().sort_values(ascending=False).head(10)
c.shape
c.head(1)
c['Title'] = c['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)





c_t=pd.concat([c, y_target], axis=1)
pd.crosstab(c_t['Title'], c_t['Sex'])
#we will just 'None' in categorical features

#Categorical missing values

NAcols=c.columns

for col in NAcols:

    if c[col].dtype == "object":

        c[col] = c[col].fillna("None")

c['Title'] = c['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')





c_t=pd.concat([c, y_target], axis=1)

c_t[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#for dataset in c:









c['Title'] = c['Title'].replace('Mlle', 'Miss')

c['Title'] = c['Title'].replace('Ms', 'Miss')

c['Title'] = c['Title'].replace('Mme', 'Mrs')





c_t=pd.concat([c, y_target], axis=1)

c_t[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()





#for dataset in c:

c['FamilySize'] = c['SibSp'] + c['Parch'] + 1







c_t=pd.concat([c, y_target], axis=1)

c_t[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#c_t[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
c['IsAlone'] = 0

c.loc[c['FamilySize'] == 1, 'IsAlone'] = 1





c_t=pd.concat([c, y_target], axis=1)

print (c_t[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
c.loc[ c['Fare'] <= 7.91, 'Fare'] = 0

c.loc[(c['Fare'] > 7.91) & (c['Fare'] <= 14.454), 'Fare'] = 1

c.loc[(c['Fare'] > 14.454) & (c['Fare'] <= 31), 'Fare']   = 2

c.loc[ c['Fare'] > 31, 'Fare'] = 3
#c['CategoricalFare'] = pd.qcut(c['Fare'], 4)

c_t=pd.concat([c, y_target], axis=1)

print (c_t[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean())
  

c.loc[c['Age']<16,'Age'] = 0

c.loc[(c['Age']>16) & (c['Age']<=32),'Age'] =1

c.loc[(c['Age']>32) & (c['Age']<=48),'Age'] =2

c.loc[(c['Age']>48) & (c['Age']<=64),'Age'] =3

c.loc[c['Age']>64,'Age'] =4

    

#c['CategoricalAge'] = pd.qcut(c['Age'], 4)

c_t=pd.concat([c, y_target], axis=1)

print (c_t[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())    

    

    

    

c_t.head(3)
c.loc[c['Ticket']=='LINE']
c['Ticket'] = c['Ticket'].replace('LINE','LINE 0')
# remove dots and slashes

c['Ticket'] = c['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())

def get_prefix(ticket):

    lead = ticket.split(' ')[0][0]

    if lead.isalpha():

        return ticket.split(' ')[0]

    else:

        return 'NoPrefix'

    

c['Prefix'] = c['Ticket'].apply(lambda x: get_prefix(x))
c['TNumeric'] = c['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)

c['TNlen'] = c['TNumeric'].apply(lambda x : len(str(x)))

c['LeadingDigit'] = c['TNumeric'].apply(lambda x : int(str(x)[0]))

c['TGroup'] = c['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))



c.head()
#c['CategoricalAge'] = pd.qcut(c['Age'], 4)

c_t=pd.concat([c, y_target], axis=1)

print (c_t[['LeadingDigit', 'Survived']].groupby(['LeadingDigit'], as_index=False).mean())    

    
#c['CategoricalAge'] = pd.qcut(c['Age'], 4)

c_t=pd.concat([c, y_target], axis=1)

print (c_t[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean())    

    
c['Cabin'] = c['Cabin'].replace('A', 'Y')

c['Cabin'] = c['Cabin'].replace('B', 'Y')

c['Cabin'] = c['Cabin'].replace('C', 'Y')

c['Cabin'] = c['Cabin'].replace('D', 'Y')

c['Cabin'] = c['Cabin'].replace('E', 'Y')

c['Cabin'] = c['Cabin'].replace('F', 'Y')

c['Cabin'] = c['Cabin'].replace('G', 'Y')

c['Cabin'] = c['Cabin'].replace('T', 'Y')

c['Cabin'] = c['Cabin'].replace('X', 'X')
#c['CategoricalAge'] = pd.qcut(c['Age'], 4)

c_t=pd.concat([c, y_target], axis=1)

print (c_t[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean())   
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



c['Title'] = c['Title'].map(title_mapping)

c['Title'] = c['Title'].fillna(0)







c['Sex'] = c['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



c.head()



c['Embarked'] = c['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



c.head(10)


#c['Sex_Class_Embark'] = 0 

    

c.loc[(c['Sex'] == 1) & ((c['Pclass'] == 1) | (c['Pclass'] == 2) ) & ((c['Embarked'] == 0)  | (c['Embarked'] == 1)  | (c['Embarked'] == 2)),'Sex_Class_Embark'] = 0

    

    

c.loc[(c['Sex'] == 1) & (c['Pclass'] == 3) & ((c['Embarked'] == 1)  | (c['Embarked'] == 2)),'Sex_Class_Embark'] = 1

    



c.loc[(c['Sex'] == 0) & (c['Pclass'] == 1) & ((c['Embarked'] == 0)  | (c['Embarked'] == 1)),'Sex_Class_Embark'] = 2

c.loc[(c['Sex'] == 1) & (c['Pclass'] == 3) & (c['Embarked'] == 0),'Sex_Class_Embark'] = 2

    

    

c.loc[(c['Sex'] == 0) & ((c['Pclass'] == 2)  | (c['Pclass'] == 3) ) & ((c['Embarked'] == 0)  | (c['Embarked'] == 1)  | (c['Embarked'] == 2)),'Sex_Class_Embark'] = 3

    

    

c.loc[(c['Sex'] == 0) & ((c['Pclass'] == 1)  |(c['Pclass'] == 2) ) & (c['Embarked'] == 2),'Sex_Class_Embark'] = 4

    

    
#c['CategoricalFare'] = pd.qcut(c['Fare'], 4)

c_t=pd.concat([c, y_target], axis=1)

print (c_t[['Sex_Class_Embark', 'Survived']].groupby(['Sex_Class_Embark'], as_index=False).mean())




c.head(2)
print (c_t[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)



print (c_t[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)





print (c_t[['Sex_Class_Embark', 'Survived']].groupby(['Sex_Class_Embark'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)

#print (c_t[['Name', 'Survived']].groupby(['Name'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#print('_'*60)



print (c_t[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)



print (c_t[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)





#print (c_t[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#print('_'*60)





print (c_t[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)







print (c_t[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)



print (c_t[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)





print (c_t[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)





print (c_t[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)





print (c_t[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)



print (c_t[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('_'*60)





#c = c.drop([ 'Parch' ,'SibSp','FamilySize',], axis = 1)

#c = c.drop(['Name','Ticket' ,'Parch', 'SibSp' ,'FamilySize' ,'Sex_Class_Embark' ], axis = 1)

#c = c.drop(['Name','Ticket' ,'Parch', 'SibSp' ,'FamilySize' ,'Sex_Class_Embark'], axis = 1)

#c = c.drop(['Name','Ticket','TNumeric','FamilySize' ,'Parch', 'SibSp' ], axis = 1)

c = c.drop(['Name','Ticket','TNumeric','FamilySize'], axis = 1)
c.isnull().sum().sort_values(ascending=False).head(20)
c.describe()
c.describe(include=['O'])


c.head(2) 
#c = c.drop(['Name','Ticket','TNumeric','FamilySize' ,'Parch' , 'SibSp','Sex_Class_Embark'], axis = 1)

#c['Sex_Class_Embark'] = c['Sex_Class_Embark'].astype(str)

#c = c.drop(['Name','Ticket','TNumeric','FamilySize'], axis = 1)

c['LeadingDigit'] = c['LeadingDigit'].astype(str)

c['TNlen'] = c['TNlen'].astype(str)

c['IsAlone'] = c['IsAlone'].astype(str)

#c['FamilySize'] = c['FamilySize'].astype(str)

c['Title'] = c['Title'].astype(str)

c['Embarked'] = c['Embarked'].astype(str)

c['Fare'] = c['Fare'].astype(str)

c['Parch'] = c['Parch'].astype(str)

c['SibSp'] = c['SibSp'].astype(str)



c['Age'] = c['Age'].astype(str)

c['Sex'] = c['Sex'].astype(str)

c['Pclass'] = c['Pclass'].astype(str)

c['Cabin'] = c['Cabin'].astype(str)

c['Sex_Class_Embark'] = c['Sex_Class_Embark'].astype(str) 



 
c.describe(include=['O'])
c.shape
c.describe()
cb=pd.get_dummies(c)

print("the shape of the original dataset",c.shape)

print("the shape of the encoded dataset",cb.shape)

print("We have ",cb.shape[1]- c.shape[1], 'new encoded features')
cb.head(2)
cb1=pd.get_dummies(c,drop_first=True)

print("the shape of the original dataset",c.shape)

print("the shape of the encoded dataset",cb1.shape)

print("We have ",cb1.shape[1]- c.shape[1], 'new encoded features')
cb1.head(2)
# dummy with  dont drop frist

Train = cb[:na]  #na is the number of rows of the original training set

                 

Test = cb[na:]  #testset  after clean missing values and feature engineering and encoder  we do NOT apply outliers on it



# dummy with  drop frist

train_1 = cb1[:na] 

test_1 =  cb1[na:]
print(train_1.shape)

print(y_target.shape)

print(test_1.shape)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder



# NAIBE BAYES

from sklearn.naive_bayes import GaussianNB

#KNN

from sklearn.neighbors import KNeighborsClassifier

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

#SVM

from sklearn.svm import SVC

#DECISON TREE

from sklearn.tree import DecisionTreeClassifier

#XGBOOST

from xgboost import XGBClassifier

#AdaBoosting Classifier

from sklearn.ensemble import AdaBoostClassifier

#GradientBoosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

#HistGradientBoostingClassifier

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier



from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV









from sklearn.preprocessing import StandardScaler ,Normalizer , MinMaxScaler, RobustScaler 

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")
X_train,X_test,y_train,y_test = train_test_split(train_1,y_target,test_size=0.30 ,  shuffle=True, random_state=42)
sk_fold = StratifiedKFold(10,shuffle=True, random_state=42)

#sc =StandardScaler()



sc =StandardScaler()

#sc =Normalizer()

#sc = MinMaxScaler()





X_train= sc.fit_transform(X_train)



X_train_1= sc.transform(train_1.values)



X_test= sc.transform(X_test)



X_submit= sc.transform(test_1.values)





g_nb = GaussianNB()

knn = KNeighborsClassifier()

ran_for  = RandomForestClassifier()

log_reg = LogisticRegression()

svc = SVC()

tree= DecisionTreeClassifier()

xgb = XGBClassifier()



ada_boost = AdaBoostClassifier()

grad_boost = GradientBoostingClassifier(n_estimators=100)

hist_grad_boost = HistGradientBoostingClassifier()









clf = [("Naive Bayes",g_nb,{}),\

       ("K Nearest",knn,{"n_neighbors":[3,5,8],"leaf_size":[25,30,35]}),\

       ("Random Forest",ran_for,{"n_estimators":[100],"random_state":[42],"min_samples_leaf":[5,10,20,40,50],"bootstrap":[False]}),\

       ("Logistic Regression",log_reg,{"penalty":['l2'],"C":[100, 10, 1.0, 0.1, 0.01] , "solver":['saga']}),\

       ("Support Vector",svc,{"kernel": ["linear","rbf"],"gamma":['auto'],"C":[0.1, 1, 10, 100, 1000]}),\

       ("Decision Tree", tree, {}),\

       ("XGBoost",xgb,{"n_estimators":[200],"max_depth":[3,4,5],"learning_rate":[.01,.1,.2],"subsample":[.8],"colsample_bytree":[1],"gamma":[0,1,5],"lambda":[.01,.1,1]}),\

       

       ("Adapative Boost",ada_boost,{"n_estimators":[100],"learning_rate":[.6,.8,1]}),\

       ("Gradient Boost",grad_boost,{}),\

     

       ("Histogram GB",hist_grad_boost,{"loss":["binary_crossentropy"],"min_samples_leaf":[5,10,20,40,50],"l2_regularization":[0,.1,1]})]





stack_list=[]

train_scores = pd.DataFrame(columns=["Name","Train Score","Test Score"])



i=0

for name,clf1,param_grid in clf:

    clf = GridSearchCV(clf1,param_grid=param_grid,scoring="accuracy",cv=sk_fold,return_train_score=True)

    clf.fit(X_train,y_train) #.reshape(-1,1)

    y_pred = clf.best_estimator_.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    

    #train_scores.loc[i]= [name,cross_val_score(clf,X_train,y_train,cv=sk_fold,scoring="accuracy").mean(),(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    train_scores.loc[i]= [name,clf.best_score_,(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    stack_list.append(clf.best_estimator_)

    i=i+1

    

est = [("g_nb",stack_list[0]),\

       ("knn",stack_list[1]),\

       ("ran_for",stack_list[2]),\

       ("log_reg",stack_list[3]),\

       ("svc",stack_list[4]),\

       ("dec_tree",stack_list[5]),\

       ("XGBoost",stack_list[6]),\

       ("ada_boost",stack_list[7]),\

       ("grad_boost",stack_list[8]),\

       ("hist_grad_boost",stack_list[9])]







sc = StackingClassifier(estimators=est,final_estimator = None,cv=sk_fold,passthrough=False)

sc.fit(X_train,y_train)

y_pred = sc.predict(X_test)

cm1 = confusion_matrix(y_test,y_pred)

y_pred_train = sc.predict(X_train)

cm2 = confusion_matrix(y_train,y_pred_train)

train_scores.append(pd.Series(["Stacking",(cm2[0,0]+cm2[1,1,])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]),(cm1[0,0]+cm1[1,1,])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])],index=train_scores.columns),ignore_index=True)

#Import Libraries

from sklearn.ensemble import VotingClassifier

#----------------------------------------------------

#Applying VotingClassifier Model 

'''

ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)

'''

#loading Voting Classifier

VotingClassifierModel = VotingClassifier(estimators=[("grad_boost",stack_list[8]),("svc",stack_list[4]) , ("ada_boost",stack_list[7])], voting='hard')

VotingClassifierModel.fit(X_train, y_train)





#Calculating Details

print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))

print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))

print('----------------------------------------------------')
y=df_train['Survived'].to_frame()

print(X_train_1.shape)

print(y.shape)

print(X_submit.shape)



#Calculating Prediction



y_submit = VotingClassifierModel.predict(X_submit)

submit = pd.DataFrame({

        "PassengerId": df_test.PassengerId,

        "Survived": y_submit

    })


submit.PassengerId = submit.PassengerId.astype(int)

submit.Survived = submit.Survived.astype(int)

submit.to_csv("titanic_submit.csv", index=False)
print('Predicted Value for VotingClassifierModel is : ' , y_submit[:10])

 



'''

# dummy with  dont drop frist

Train = cb[:na]  #na is the number of rows of the original training set

                 

Test = cb[na:]  #testset  after clean missing values and feature engineering and encoder  we do NOT apply outliers on it



# dummy with  drop frist

train_1 = cb1[:na] 

test_1 =  cb1[na:]



print(train_1.shape)

print(y_target.shape)

print(test_1.shape)

'''
'''

from sklearn.decomposition import PCA



pca = PCA(0.95, whiten=True)

pca_train_data = pca.fit_transform(train_1)

print(pca_train_data.shape,'\n')



explained_variance = pca.explained_variance_ratio_ 

print(explained_variance)

'''
'''

train_1=pd.DataFrame(pca_train_data)

test_1=pd.DataFrame(pca.transform(test_1))



print(train_1.shape)

print(y_target.shape)

print(test_1.shape)

'''
'''

X_train,X_test,y_train,y_test = train_test_split(train_1,y_target,test_size=0.30, random_state=42)

'''
'''

sk_fold = StratifiedKFold(10,shuffle=True, random_state=42)

#sc =StandardScaler()



sc =StandardScaler()

#sc =Normalizer()

#sc = MinMaxScaler()





X_train= sc.fit_transform(X_train)



X_train_1= sc.transform(train_1.values)



X_test= sc.transform(X_test)



X_submit= sc.transform(test_1.values)





g_nb = GaussianNB()

knn = KNeighborsClassifier()

ran_for  = RandomForestClassifier()

log_reg = LogisticRegression()

svc = SVC()

tree= DecisionTreeClassifier()

xgb = XGBClassifier()



ada_boost = AdaBoostClassifier()

grad_boost = GradientBoostingClassifier(n_estimators=100)

hist_grad_boost = HistGradientBoostingClassifier()









clf = [("Naive Bayes",g_nb,{}),\

       ("K Nearest",knn,{"n_neighbors":[3,5,8],"leaf_size":[25,30,35]}),\

       ("Random Forest",ran_for,{"n_estimators":[100],"random_state":[42],"min_samples_leaf":[5,10,20,40,50],"bootstrap":[False]}),\

       ("Logistic Regression",log_reg,{"penalty":['l2'],"C":[100, 10, 1.0, 0.1, 0.01] , "solver":['saga']}),\

       ("Support Vector",svc,{"kernel": ["linear","rbf"],"gamma":['auto'],"C":[0.1, 1, 10, 100, 1000]}),\

       ("Decision Tree", tree, {}),\

       ("XGBoost",xgb,{"n_estimators":[200],"max_depth":[3,4,5],"learning_rate":[.01,.1,.2],"subsample":[.8],"colsample_bytree":[1],"gamma":[0,1,5],"lambda":[.01,.1,1]}),\

       

       ("Adapative Boost",ada_boost,{"n_estimators":[100],"learning_rate":[.6,.8,1]}),\

       ("Gradient Boost",grad_boost,{}),\

     

       ("Histogram GB",hist_grad_boost,{"loss":["binary_crossentropy"],"min_samples_leaf":[5,10,20,40,50],"l2_regularization":[0,.1,1]})]





stack_list=[]

train_scores = pd.DataFrame(columns=["Name","Train Score","Test Score"])



i=0

for name,clf1,param_grid in clf:

    clf = GridSearchCV(clf1,param_grid=param_grid,scoring="accuracy",cv=sk_fold,return_train_score=True)

    clf.fit(X_train,y_train) #.reshape(-1,1)

    y_pred = clf.best_estimator_.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    

    #train_scores.loc[i]= [name,cross_val_score(clf,X_train,y_train,cv=sk_fold,scoring="accuracy").mean(),(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    train_scores.loc[i]= [name,clf.best_score_,(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    stack_list.append(clf.best_estimator_)

    i=i+1

    

est = [("g_nb",stack_list[0]),\

       ("knn",stack_list[1]),\

       ("ran_for",stack_list[2]),\

       ("log_reg",stack_list[3]),\

       ("svc",stack_list[4]),\

       ("dec_tree",stack_list[5]),\

       ("XGBoost",stack_list[6]),\

       ("ada_boost",stack_list[7]),\

       ("grad_boost",stack_list[8]),\

       ("hist_grad_boost",stack_list[9])]









'''
'''

sc = StackingClassifier(estimators=est,final_estimator = None,cv=sk_fold,passthrough=False)

sc.fit(X_train,y_train)

y_pred = sc.predict(X_test)

cm1 = confusion_matrix(y_test,y_pred)

y_pred_train = sc.predict(X_train)

cm2 = confusion_matrix(y_train,y_pred_train)

train_scores.append(pd.Series(["Stacking",(cm2[0,0]+cm2[1,1,])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]),(cm1[0,0]+cm1[1,1,])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])],index=train_scores.columns),ignore_index=True)



'''
'''

# dummy with  dont drop frist

train = cb[:na]  #na is the number of rows of the original training set

                 

test = cb[na:]  #testset  after clean missing values and feature engineering and encoder  we do NOT apply outliers on it



# dummy with  drop frist

train_1 = cb1[:na] 

test_1 =  cb1[na:]



print(train_1.shape)

print(y_target.shape)

print(test_1.shape)

print("****************************************************")

print(train.shape)

print(y_target.shape)

print(test.shape)



'''
'''

# Importing libraries

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.mixture import GaussianMixture

from sklearn.svm import SVC



X = np.r_[train,test]

print('X shape :',X.shape)

print('\n')



# USING THE GAUSSIAN MIXTURE MODEL 



#The Bayesian information criterion (BIC) can be used to select the number of components in a Gaussian Mixture in an efficient way. 

#In theory, it recovers the true number of components only in the asymptotic regime

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)



#The GaussianMixture comes with different options to constrain the covariance of the difference classes estimated: 

# spherical, diagonal, tied or full covariance.

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(X)

        bic.append(gmm.aic(X))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

            

best_gmm.fit(X)

gmm_train = best_gmm.predict_proba(train)

gmm_test = best_gmm.predict_proba(test)

'''
'''

X_train,X_test,y_train,y_test = train_test_split(gmm_train,y_target,test_size=0.30, random_state=101)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

'''
'''

sk_fold = StratifiedKFold(10,shuffle=True, random_state=42)





X_train= X_train





X_train_1= pd.DataFrame(gmm_train).values



X_test= X_test



X_submit =  pd.DataFrame(gmm_test).values



g_nb = GaussianNB()

knn = KNeighborsClassifier()

ran_for  = RandomForestClassifier()

log_reg = LogisticRegression()

svc = SVC()

tree= DecisionTreeClassifier()

xgb = XGBClassifier()



ada_boost = AdaBoostClassifier()

grad_boost = GradientBoostingClassifier(n_estimators=100)

hist_grad_boost = HistGradientBoostingClassifier()









clf = [("Naive Bayes",g_nb,{}),\

       ("K Nearest",knn,{"n_neighbors":[3,5,6,7,8,9,10],"leaf_size":[25,30,35]}),\

       ("Random Forest",ran_for,{"n_estimators":[10, 50, 100, 200,400],"max_depth":[3, 10, 20, 40],"random_state":[99],"min_samples_leaf":[5,10,20,40,50],"bootstrap":[False]}),\

       ("Logistic Regression",log_reg,{"penalty":['l2'],"C":[100, 10, 1.0, 0.1, 0.01] , "solver":['saga']}),\

       ("Support Vector",svc,{"kernel": ["linear","rbf"],"gamma":[0.05,0.0001,0.01,0.001],"C":[0.1, 1, 10, 100, 1000]},),\

      

       ("Decision Tree", tree, {}),\

       ("XGBoost",xgb,{"n_estimators":[200],"max_depth":[3,4,5],"learning_rate":[.01,.1,.2],"subsample":[.8],"colsample_bytree":[1],"gamma":[0,1,5],"lambda":[.01,.1,1]}),\

       

       ("Adapative Boost",ada_boost,{"n_estimators":[100],"learning_rate":[.6,.8,1]}),\

       ("Gradient Boost",grad_boost,{}),\

     

       ("Histogram GB",hist_grad_boost,{"loss":["binary_crossentropy"],"min_samples_leaf":[5,10,20,40,50],"l2_regularization":[0,.1,1]})]





stack_list=[]

train_scores = pd.DataFrame(columns=["Name","Train Score","Test Score"])



i=0

for name,clf1,param_grid in clf:

    clf = GridSearchCV(clf1,param_grid=param_grid,scoring="accuracy",cv=sk_fold,return_train_score=True)

    clf.fit(X_train,y_train) #.reshape(-1,1)

    y_pred = clf.best_estimator_.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    

    #train_scores.loc[i]= [name,cross_val_score(clf,X_train,y_train,cv=sk_fold,scoring="accuracy").mean(),(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    train_scores.loc[i]= [name,clf.best_score_,(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    stack_list.append(clf.best_estimator_)

    i=i+1

    

est = [("g_nb",stack_list[0]),\

       ("knn",stack_list[1]),\

       ("ran_for",stack_list[2]),\

       ("log_reg",stack_list[3]),\

       ("svc",stack_list[4]),\

       ("dec_tree",stack_list[5]),\

       ("XGBoost",stack_list[6]),\

       ("ada_boost",stack_list[7]),\

       ("grad_boost",stack_list[8]),\

       ("hist_grad_boost",stack_list[9])]









'''

'''

sc = StackingClassifier(estimators=est,final_estimator = None,cv=sk_fold,passthrough=False)

sc.fit(X_train,y_train)

y_pred = sc.predict(X_test)

cm1 = confusion_matrix(y_test,y_pred)

y_pred_train = sc.predict(X_train)

cm2 = confusion_matrix(y_train,y_pred_train)

train_scores.append(pd.Series(["Stacking",(cm2[0,0]+cm2[1,1,])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]),(cm1[0,0]+cm1[1,1,])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])],index=train_scores.columns),ignore_index=True)



'''
'''

y=df_train['Survived'].to_frame()

'''
'''

print(X_train_1.shape)

print(y.shape)

print(X_submit.shape)

'''





'''

stack_list[7].fit(X_train_1,y)

y_submit = stack_list[7].predict(X_submit)

submit = pd.DataFrame({

        "PassengerId": df_test.PassengerId,

        "Survived": y_submit

    })

    

'''
    

'''

submit.PassengerId = submit.PassengerId.astype(int)

submit.Survived = submit.Survived.astype(int)

submit.to_csv("titanic_submit.csv", index=False)

    

'''
    

'''

submit.PassengerId = submit.PassengerId.astype(int)

submit.Survived = submit.Survived.astype(int)

submit.to_csv("titanic_submit.csv", index=False)

    

'''