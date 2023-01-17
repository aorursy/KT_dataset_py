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
''' loading the csv files of train set , test set '''
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
'''Ignore deprecation and future, and user warnings.'''

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning) 
'''Importing Data Manipulattion Moduls'''

import numpy as np

import pandas as pd

# pandaprofile is used to generate the profile report

import pandas_profiling

import scipy.stats as stats # for stats function like skew and kurtosis



import pylab as pl

import scipy.optimize as opt
# to display the output below cell code within frontend(jupyter notebook)

%matplotlib inline

# you could also use %matplotlib notebook for more interactive plots
'''Seaborn and Matplotlib Visualization'''

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('bmh')                    

sns.set_style({'axes.grid':False}) 

'''plotly Visualization'''

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)
# importing one hot encoder from sklearn 

from sklearn.preprocessing import OneHotEncoder

# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
pd.set_option('display.max_columns', None) # to display all columns
pkmn_type_colors = ['#78C850',  # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                   ]
train.head()
test.head()
print("train shape: ",train.shape)

print("test shape: ",test.shape)
print("gender_submission : ",gender_submission.shape)

gender_submission.head()
train.describe(include='all')
test.describe()
''' merge the train set and test set for data cleaning ie. imputation missing values and correcting 

outliers, feature engineering etc'''
merged= pd.concat([train,test],axis=0,sort=False)

print('merged shape : ',merged.shape)

merged.sample(5)
merged.describe(include='all')
#lets focus on name varible first and clean it and create new variable 'Title'

merged['Title']=merged.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

merged.head()
merged['Title'].value_counts()
merged['Title'].unique()
merged[merged['Title'].isin(

    ['Dona','Don','Sir','Dr','Rev','Col','Ms','Mlle','Major','Lady','Jonkheer','Mme','the Countess','Capt'])]
# changing title of Dr based on sex and Age

merged['Title']=np.where((merged.Title=='Dr') & (merged.Sex=='female'),'Mrs',merged.Title)

merged['Title']=np.where((merged.Title=='Dr') & (merged.Sex=='male'),'Mr',merged.Title)

#for Rev

merged.loc[merged.Title=='Rev','Title']='Mr'

# based on sex and age and title

merged.loc[(merged.Title=='Lady')|(merged.Title== 'the Countess')|(merged.Title=='Dona'),'Title']='Mrs'

merged.loc[(merged.Title=='Mme')|(merged.Title== 'Ms'),'Title']='Miss'



merged.loc[(merged.Title=='Col')|(merged.Title== 'Sir')|(merged.Title=='Major')|(merged.Title=='Jonkheer')|(merged.Title=='Capt')|(merged.Title=='Don'),'Title']='Mr'

# based on age and if husband name is mention in the (brackets) with passenger name

merged.loc[merged.PassengerId==642,'Title']='Miss'

merged.loc[merged.PassengerId==711,'Title']='Mrs' #husband name is there although spouse=0, he is not on ship with her
merged['Title'].value_counts()

# we successfully cleaned the title column although we added some bias
fig, ax=plt.subplots(figsize=(10,6))

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.swarmplot(x="Age", y="Title", hue="Survived",

              palette=["r", "c", "y"], data=merged)

plt.show()
sns.catplot(x="Title", y="Survived", hue="Pclass", kind="bar", data=merged)

plt.show()
sns.catplot(y="Title", hue="Survived", kind="count",

            palette="pastel", edgecolor=".6",

            data=merged)

plt.show()
g = sns.catplot(x="Title", y="Survived", col="Pclass",data=merged, saturation=.5,kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

.set_titles("{col_name} {col_var}")

.set(ylim=(0, 1))

.despine(left=True))  

plt.show()
sns.catplot(y="Title", hue="Survived", kind="count",

            palette="pastel", edgecolor=".6",

            data=merged)

plt.show()
merged['Cabin'].describe()
df_cabin=merged.loc[pd.notnull(merged.Cabin)]

df_cabin.head()
df_cabinstrip=df_cabin.Cabin.str.split(expand=True,)

df_cabin
df_cabinstrip.rename(columns={0:'one',1:'two',2:'three',3:'four'},inplace=True)
df_cabinstrip['one']=df_cabinstrip['one'].str[0]

df_cabinstrip['two']=df_cabinstrip['two'].str[0]

df_cabinstrip['three']=df_cabinstrip['three'].str[0]

df_cabinstrip['four']=df_cabinstrip['four'].str[0]
df_cabinstrip.head()
# pandas_profiling.ProfileReport(df_cabinstrip) . i am not executing this statement now 
df_cabinstrip.describe()
pd.notnull(df_cabinstrip).sum().sum() #total number of cabins alloted
df1=pd.concat([df_cabin,df_cabinstrip],axis=1)
df1
df1['one']=df1.one.str[0].str.strip() # we are keeping only cabin category ie A, B, C etc 

df1
print(pd.crosstab(df1['Survived'],df1['one']),"\n")

print(pd.crosstab(df1['Survived'],df1['two']),"\n")

print(pd.crosstab(df1['Survived'],df1['three']),"\n")

print(pd.crosstab(df1['Survived'],df1['four']),"\n")
merged['Cabin']=merged.Cabin.str[0].str.strip()

merged.head()
merged['Cabin'].value_counts()
pd.crosstab(merged['Survived'],[merged['Pclass'],merged['Cabin']])
g = sns.catplot(x="Cabin", y="Survived", col="Pclass",data=merged, saturation=.5,kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

.set_titles("{col_name} {col_var}")

.set(ylim=(0, 1))

.despine(left=True))  

plt.show()
merged.loc[(merged.Cabin=='G')|(merged.Cabin=='A')|(merged.Cabin=='F')|(merged.Cabin=='T'),'Cabin']= 'other'

#we keep T cabin out as its very less in number

merged.loc[pd.isnull(merged['Cabin']),'Cabin']='none'

merged.loc[(merged['Cabin']=='none')&(merged['Pclass']==1),'Cabin']='other'
sns.countplot(x='Cabin', data=merged, hue='Survived')

# Rotate x-labels

plt.xticks(rotation=-45)

plt.show()
# Swarm plot with Pokemon color palette

sns.swarmplot(x='Pclass', y='Fare', data=merged,hue='Cabin',palette=pkmn_type_colors)

plt.show()
# Swarm plot with Pokemon color palette

sns.swarmplot(x='Cabin', y='Fare', data=merged,hue='Survived',palette=pkmn_type_colors)

plt.show()
print('mean age for title-master : ',merged.loc[(pd.notnull(merged['Age']))&(merged.Title== 'Master')]['Age'].mean())

print('mean age for title-mr : ',merged.loc[(pd.notnull(merged['Age']))&(merged.Title== 'Mr')]['Age'].mean())

print('mean age for title-miss : ',merged.loc[(pd.notnull(merged['Age']))&(merged.Title== 'Miss')]['Age'].mean())

print('mean age for title-mrs : ',merged.loc[(pd.notnull(merged['Age']))&(merged.Title== 'Mrs')]['Age'].mean())
''' okay now lets impute age based on title'''

merged.loc[(pd.isnull(merged['Age']))&(merged.Title== 'Master'),'Age']=5

merged.loc[(pd.isnull(merged['Age']))&(merged.Title== 'Mr'),'Age']=32

merged.loc[(pd.isnull(merged['Age']))&(merged.Title== 'Miss'),'Age']=22

merged.loc[(pd.isnull(merged['Age']))&(merged.Title== 'Mrs'),'Age']=36

# since max age is 80 so we took age limit on bins as 80 with 6 steps for binning

bins=np.arange(0,85,6).tolist()

print('bins : ',bins)

labels=np.arange(3,85,6).tolist()

print('labels : ',labels)

merged['Agegroup']=pd.cut(merged['Age'],bins,labels=labels)

merged.head(5)
# Swarm plot with Pokemon color palette

sns.swarmplot(x='Agegroup', y='Pclass', data=merged,hue='Survived',palette=pkmn_type_colors)

plt.show()
#since fare should not be zero so lets impute it based on class but lets dive into outliers
plt.figure(figsize = (12, 8))

sns.boxplot(x = 'Pclass', y = 'Fare', data=merged)

sns.swarmplot(x = 'Pclass', y = 'Fare',hue='Survived', data = merged)

plt.title("Box plots of pclass vs fare")

plt.xlabel("pclass")

plt.ylabel("fare")

plt.show()

#lets set fare>100 as fare = 100 since there are outliers in fare category 
merged.loc[merged.Fare>100,'Fare']=100

merged.loc[(merged.Fare<25)& (merged.Pclass==1),'Fare']=25
plt.figure(figsize = (12, 8))

sns.boxplot(x = 'Pclass', y = 'Fare', data=merged)

sns.swarmplot(x = 'Pclass', y = 'Fare',hue='Survived', data = merged)

plt.title("Box plots of pclass vs fare")

plt.xlabel("pclass")

plt.ylabel("fare")

plt.show()

#lets set fare>100 as fare = 100 since there are outliers in fare category 
print('mean fare for class 1 is : ',merged.loc[(merged.Pclass==1)]['Fare'].mean())

print('mean fare for class 2 is : ',merged.loc[(merged.Pclass==2)]['Fare'].mean())

print('mean fare for class 3 is : ',merged.loc[(merged.Pclass==3)]['Fare'].mean())
merged.loc[merged.Fare==0]
merged.loc[pd.isnull(merged.Fare)]
merged.loc[pd.isnull(merged.Fare),'Fare']=13 # as it is in class 3
# imputing fare==0 to fare.mean of that class

merged.loc[(merged.Fare==0)& (merged.Pclass==3),'Fare']=12 # fare mean for Pclass=3 

merged.loc[(merged.Fare==0)& (merged.Pclass==2),'Fare']=21 # fare mean for Pclass=2 

merged.loc[(merged.Fare==0)& (merged.Pclass==1),'Fare']=60 # fare mean for Pclass=1 
bins=np.arange(0,101,4).tolist()

print('bins : ',bins)

labels=np.arange(2,101,4).tolist()

print('labels : ',labels)

merged['faregp']=pd.cut(merged['Fare'],bins,labels=labels)

merged.sample(5)
fig, ax = plt.subplots(figsize=(16,8))

sns.set(style="whitegrid", palette="muted")

sns.swarmplot(x="Pclass", y="faregp", hue="Survived",

              palette=["r", "c", "y"], data=merged)

plt.show()
merged.loc[pd.isnull(merged['Embarked'])]
# since both missing values of embarked is in pclass 1

merged.loc[(merged.Pclass==1)&(merged.faregp==78)&(merged.Sex=='female')&(merged.Survived==1)]['Embarked'].value_counts()
# lets impute it by 'S' , though we could have imputed it by 'C' also 

merged.loc[(merged.PassengerId==62)|(merged.PassengerId==830),'Embarked']='S'
merged.isnull().sum()
# lets drop irrelevant columns now , although we didnt take certain things into consideration like in column 'name' , we didnt give importance to surname and in 'ticket' column we imputed values on certain assumption 

# so we have already added a certain degree of bias by ignoring them
final=merged.drop(columns=['Name','Age','Fare','Ticket'])
print('final shape : ', final.shape, '\n')

final.head()
fig=plt.figure(figsize=(10,6))

plt.subplot2grid((2,3),(1,0),colspan=2)

for x in [1,2,3]:

    merged.Age[merged.Pclass==x].plot(kind='kde')

    

plt.title('Class wrt Age')

plt.legend(('1st','2nd','3rd'))

plt.show()
fig=plt.figure(figsize=(10,6))

#rich man

plt.subplot2grid((4,4),(0,0))

merged.Survived[(merged.Sex=='male') & (merged.Pclass==1)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='b')

plt.title('Rich Man Survived')



#poor man

plt.subplot2grid((4,4),(0,1))

merged.Survived[(merged.Sex=='male') & (merged.Pclass==3)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='r')

plt.title('Poor Man Survived')



#rich women

plt.subplot2grid((4,4),(0,2))

merged.Survived[(merged.Sex=='female') & (merged.Pclass==1)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='b')

plt.title('Rich Women Survived')



#poor  woman

plt.subplot2grid((4,4),(0,3))

merged.Survived[(merged.Sex=='female') & (merged.Pclass==3)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='g')

plt.title('Poor Women Survived')

plt.show()
f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(merged.corr(),annot=True,linewidths=.5,fmt='.0%',ax=ax)

plt.show()
merged.Survived.dropna(inplace = True)

labels = merged.Survived.value_counts().index

colors = ["red","lightblue"]

explode = [0,0]

sizes = merged.Survived.value_counts().values



# visual cp

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('People According to Survived',color = 'blue',fontsize = 15)

plt.show()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(merged, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked')

fig=plt.figure(figsize=(16,6))

grid = sns.FacetGrid(merged, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
sns.pairplot(train)
cat_columns = ['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Title','Agegroup','faregp']

merged_processed = pd.get_dummies(final, prefix_sep="_",columns=cat_columns)

merged_processed
df_train=merged_processed.loc[merged_processed['PassengerId']<892]

df_test=merged_processed.loc[merged_processed['PassengerId']>891]
x_train=df_train.drop(columns=['PassengerId','Survived'])

y_train=df_train['Survived']
x=np.asarray(x_train)

x
y=np.asarray(y_train)

y[0:5]
# importing one hot encoder from sklearn 

from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

x[0:1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)