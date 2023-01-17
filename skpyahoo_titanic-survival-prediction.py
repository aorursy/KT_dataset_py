import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVC

import warnings

warnings.filterwarnings('ignore')
#Load the Train set

train_set = pd.read_csv('../input/titanic/train.csv')

train_set.tail(10)
#Load the Test set

test_set = pd.read_csv('../input/titanic/test.csv')

test_set.head(10)
# inspect the structure etc.

print(train_set.info(), "\n")

print(train_set.shape)
print(test_set.info(), "\n")

print(test_set.shape)
train_set['Survived'].value_counts()
# missing values in Train set df

train_set.isnull().sum()
round(100*(test_set.isnull().sum().sort_values(ascending=False)/len(test_set.index)), 2)
# summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(train_set.isnull().sum().sort_values(ascending=False)/len(train_set.index)), 2)
train_set.Age.describe()
train_set['Title']=train_set['Name'].map(lambda x: x.split(',')[1].split('.')[0].lstrip())

train_set.head()
train_set['Title'].value_counts()
print(train_set.info())
#Check the list of values in title column

train_set.Title.unique()
# lets sort the remaining other categories in title to various sub category of Mr, Miss, mrs, train_set

title_list=['Mrs', 'Mr', 'Master', 'Miss']

train_set.loc[~train_set['Title'].isin(title_list),['Age','Sex','Title']]
# function to bucket other titles into major 4

def fix_title(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme','Lady','Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
train_set['Title']=train_set.apply(fix_title, axis=1)

train_set['Title'].value_counts()
train_set.Age.isnull().sum()
#Check mean  on title Subclass w.r.t Age

train_set.groupby(['Title'])['Age'].describe()
train_set.groupby(['Title'])['Age'].median()
round(100*(train_set.isnull().sum().sort_values(ascending=False)/len(train_set.index)), 2)
# Total Nullvalues in Age Column

train_set.Age.isnull().sum()
print('Value of Master Class with Null Values {other}'.format(other=train_set.loc[(train_set.Title=='Master'),['Age']].isnull().sum()))

print('Value of Miss Class with Null Values {other}'.format(other=train_set.loc[(train_set.Title=='Miss'),['Age']].isnull().sum()))

print('Value of Mrs Class with Null Values {other}'.format(other=train_set.loc[(train_set.Title=='Mrs'),['Age']].isnull().sum()))

print('Value of Mr Class with Null Values {other}'.format(other=train_set.loc[(train_set.Title=='Mr'),['Age']].isnull().sum()))
train_set.loc[(train_set.Title=='Master'),['Age']].isnull().sum()
#Impute Missing values in Age Column

master_median=train_set.loc[(train_set.Title=='Master') & ~(train_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

mr_median=train_set.loc[(train_set.Title=='Mr') & ~(train_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

miss_median=train_set.loc[(train_set.Title=='Miss') & ~(train_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

mrs_median=train_set.loc[(train_set.Title=='Mrs') & ~(train_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')
train_set.loc[(train_set.Title=='Master') & (train_set.Age.isnull()),'Age']=train_set.loc[(train_set.Title=='Master') & (train_set.Age.isnull()),'Age'].replace(np.nan,master_median.median())

train_set.loc[(train_set.Title=='Miss') & (train_set.Age.isnull()),'Age']=train_set.loc[(train_set.Title=='Miss') & (train_set.Age.isnull()),'Age'].replace(np.nan,miss_median.median())

train_set.loc[(train_set.Title=='Mrs') & (train_set.Age.isnull()),'Age']=train_set.loc[(train_set.Title=='Mrs') & (train_set.Age.isnull()),'Age'].replace(np.nan,mrs_median.median())

train_set.loc[(train_set.Title=='Mr') & (train_set.Age.isnull()),'Age']=train_set.loc[(train_set.Title=='Mr') & (train_set.Age.isnull()),'Age'].replace(np.nan,mr_median.median())
# After Imputation on Age Colums verify the null values

train_set.Age.isnull().sum()
# Again summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(train_set.isnull().sum().sort_values(ascending=False)/len(train_set.index)), 2)
# GEt the unique set of Value of Cabin

train_set.Cabin.unique()
train_set.Cabin.isnull().sum()
# Lets see the cabin with passenger class

class_cabin=train_set.groupby(['Pclass'])['Cabin'].count()

class_cabin
# No Pclass

train_set.Pclass.value_counts()
print('Value of 1st passenger Class with Null Values {other}'.format(other=train_set.loc[(train_set.Pclass==1),['Cabin']].isnull().sum()))

print('Value of 2nd passenger Class with Null Values {other}'.format(other=train_set.loc[(train_set.Pclass==2),['Cabin']].isnull().sum()))

print('Value of 3rd passenger Class with Null Values {other}'.format(other=train_set.loc[(train_set.Pclass==3),['Cabin']].isnull().sum()))

train_set.loc[(train_set.Pclass==1) & ~(train_set.Cabin.isnull()),['Cabin']]
# Lets have Deck # as separte Columns and null as GNR

pd.Series(train_set.loc[(train_set.Pclass==1) & ~(train_set.Cabin.isnull()),['Cabin']].values.flatten()).astype('str').str[0]
# Lets have Deck # as separte Columns and null as GNR

train_set['Deck']=pd.Series(train_set.loc[~(train_set.Cabin.isnull()),['Cabin']].values.flatten()).astype('str').str[0]
train_set.head()
# Lets see the unique value and count of Deck Column

train_set['Deck'].value_counts()
train_set.Deck.unique()
train_set.Deck.isnull().sum()
# REplace Nan in Decek to GNR

train_set.loc[(train_set.Deck.isnull()),'Deck']=train_set.loc[ (train_set.Deck.isnull()),'Deck'].replace(np.nan,'GNR')

train_set.Deck.isnull().sum()
# Remove Cabin Column

train_set.drop('Cabin',axis=1,inplace=True)
train_set.head()
# Now lets check the column with null values

train_set.isnull().sum()
# Value of Embarked on various categories

train_set.Embarked.value_counts()
train_set.Embarked.isnull().sum()
#Lets impute 2 null records of Embarked with value 'S' as it have max occurance

train_set.loc[(train_set.Embarked.isnull()),'Embarked']=train_set.loc[ (train_set.Embarked.isnull()),'Embarked'].replace(np.nan,'S')

train_set.Embarked.isnull().sum()
# Check if any null columns are present

train_set.isnull().sum()
sns.distplot(train_set['Age'])
# pairplot

sns.pairplot(train_set)

plt.show()
sns.countplot(x="Pclass", data=train_set)
sns.distplot(train_set['Fare'])
sns.boxplot(y=train_set['Fare'])
# Checking for Outlier

train_set.Fare.describe(percentiles=[.25, .5, .75, .90, .95, .99])
sns.countplot(x="Survived", data=train_set)
### Checking the Survival Rate Rate

survival = (sum(train_set['Survived'])/len(train_set['Survived'].index))*100

survival
# Remove name and Passenger Id Column

train_set.drop(['PassengerId', 'Name'],axis=1,inplace=True)
#Checking the Correlation Matrix

plt.figure(figsize = (15,10))

sns.heatmap(train_set.corr(),annot = True)

plt.show()
#Check the Survival rate by Paasaenger Class

print(train_set [['Pclass','Survived']].groupby('Pclass').mean())
sns.countplot(x="Pclass", hue="Survived", data=train_set)
#Check the Survival rate by Paasaenger Class

sep="---------------------------------------------------------------"

print( round(train_set [['Sex','Survived']].groupby(['Sex']).mean()*100,1),'\n',sep)

print(train_set [['Pclass','Sex','Survived']].groupby(['Pclass','Sex']).agg(['count','mean']))
#tracking th Survival on the basis of Sex and PClass

g = sns.catplot(x="Pclass", hue="Sex", col="Survived",

                data=train_set, kind="count",

                height=4, aspect=.7);
# check the impact of Embarked Colum on Survival

print( round(train_set [['Embarked','Survived']].groupby(['Embarked']).mean()*100,1),'\n',sep)

print(train_set [['Pclass','Embarked','Survived']].groupby(['Embarked','Pclass']).agg(['count','mean']),'\n',sep)

print(train_set [['Pclass','Embarked','Survived','Sex']].groupby(['Embarked','Pclass','Sex']).agg(['count','mean']))
sns.countplot(x="Parch", hue="Survived", data=train_set)
sns.countplot(x="SibSp", hue="Survived", data=train_set)
train_set['Family']=train_set['SibSp']+train_set['Parch']+1

train_set.head()
df = train_set.groupby(['Ticket']).size().reset_index(name='count')

print( df)
master=pd.concat([train_set, test_set])

master.head()
# New column for Ticket Head Count on teh complete data

train_set['TicketHeadCount']=train_set['Ticket'].map(master['Ticket'].value_counts())

train_set.head()
#Let take fair per Person as per Ticket head Count

train_set['FairPerPerson']=train_set['Fare']/train_set['TicketHeadCount']

train_set[['FairPerPerson']].describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Lets check the distribution

sns.distplot(train_set['FairPerPerson'])
#Check the impact of Fair on chances of Survival

plt.figure(figsize = (15,10))

sns.violinplot(x="Family", y="FairPerPerson", hue="Survived",

                    data=train_set, palette="muted")

plt.show()
sns.violinplot(x="Pclass", y="FairPerPerson", hue="Sex",

                    data=train_set, palette="muted")
sns.violinplot(x="Survived", y="Age", hue="Sex",

                    data=train_set, palette="muted")
plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>0.0) & (train_set.Age<15.0)],jitter=True,palette='Set1')

plt.subplot(2,3,2)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>15.0) & (train_set.Age<40.0)],jitter=True,palette='Set1')

plt.subplot(2,3,3)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>40.0) & (train_set.Age<60.0)],jitter=True,palette='Set1')

plt.subplot(2,3,4)

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>60.0) & (train_set.Age<80.0)],jitter=True,palette='Set1')



plt.show()
# Lets check on graph the survival of male aginst Female within age 15-40 years

sns.stripplot(x="Survived",y="Age",data=train_set.loc[(train_set['Age']>15.0) & (train_set.Age<40.0)],jitter=True,hue='Sex',palette='Set1')
#tracking th Survival on the basis of Family Size and Sex

g = sns.catplot(x="Family", hue="Sex", col="Survived",

                data=train_set, kind="count",

                height=4, aspect=.7);
plt.figure(figsize=(16, 6))

sns.catplot(x="Family", col="Survived",data=train_set.loc[train_set.Family==1], kind="count",height=4, aspect=.7)

plt.figure(figsize=(16, 6))

sns.catplot(x="Family", col="Survived",

                data=train_set.loc[train_set.Family.between(2,4)], kind="count",

                height=4, aspect=.7)

plt.figure(figsize=(16, 6))

sns.catplot(x="Family", col="Survived",data=train_set.loc[train_set.Family>4], kind="count",height=4, aspect=.7)
print("Casuality Rate on Solo Travellers on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Family']==1) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Family']==1)])*100,2),'%')

print("Casuality Rate on family Group between 2 to 4 on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Family'].between(2,4)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Family'].between(2,4))])*100,2),'%')

print("Casuality Rate on Group of 5 and above on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Family']>4) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Family']>4)])*100,2),'%')



# Same as we can see in Graph too the Rate of Death on 1st class with repect to family Size

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set.Family==1) & (train_set.Pclass==1)], kind="count",height=4, aspect=.7)

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Pclass']==1) & (train_set['Family'].between(2,4))], kind="count",height=4, aspect=.7)

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set.Family>4) & (train_set.Pclass==1)], kind="count",height=4, aspect=.7)
print("Casuality Rate on Solo Travellers on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Family']==1) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Family']==1)])*100,2),'%')

print("Casuality Rate on family Group between 2 to 4 on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Family'].between(2,4)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Family'].between(2,4))])*100,2),'%')

print("Casuality Rate on Group of 5 and above on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Family']>4) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Family']>4)])*100,2),'%')
# Same as we can see in Graph too the Rate of Death on 2nd class with repect to family Size

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set.Family==1) & (train_set.Pclass==2)], kind="count",height=4, aspect=.7)

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Pclass']==2) & (train_set['Family'].between(2,4))], kind="count",height=4, aspect=.7)

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set.Family>4) & (train_set.Pclass==2)], kind="count",height=4, aspect=.7)
print("Casuality Rate on Solo Travellers on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Family']==1) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Family']==1)])*100,2),'%')

print("Casuality Rate on family Group between 2 to 4 on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Family'].between(2,4)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Family'].between(2,4))])*100,2),'%')

print("Casuality Rate on Group of 5 and above on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Family']>4) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Family']>4)])*100,2),'%')
# Same as we can see in Graph too the Rate of Death on 3rd class with repect to family Size

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set.Family==1) & (train_set.Pclass==3)], kind="count",height=4, aspect=.7)

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Pclass']==3) & (train_set['Family'].between(2,4))], kind="count",height=4, aspect=.7)

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set.Family>4) & (train_set.Pclass==3)], kind="count",height=4, aspect=.7)
print("Casuality Rate on Age Group 0-15 on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(0,15)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(0,15))])*100,2),'%')

print("Casuality Rate on Age Group between 15 to 40 on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(15,40)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(15,40))])*100,2),'%')

print("Casuality Rate on Group of 40-60 on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(40,60)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(40,60))])*100,2),'%')

print("Casuality Rate on Group of 60-80 on 1st Class", round(len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(60,80)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(60,80))])*100,2),'%')
# Same as we can see in Graph too the Rate of Death on 1st class with repect to Age Group

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(0,15)) & (train_set.Pclass==1)], kind="count",height=4, aspect=.7, hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Pclass']==1) & (train_set['Age'].between(15,40))], kind="count",height=4, aspect=.7,hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(40,60)) & (train_set.Pclass==1)], kind="count",height=4, aspect=.7,hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(60,80)) & (train_set.Pclass==1)], kind="count",height=4, aspect=.7,hue='Sex')
print("Casuality Rate on Age Group 0-15 on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(0,15)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(0,15))])*100,2),'%')

print("Casuality Rate on Age Group between 15 to 40 on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(15,40)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(15,40))])*100,2),'%')

print("Casuality Rate on Group of 40-60 on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(40,60)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(40,60))])*100,2),'%')

print("Casuality Rate on Group of 60-80 on 2nd Class", round(len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(60,80)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(60,80))])*100,2),'%')
# Same as we can see in Graph too the Rate of Death on 2nd class with repect to Age Group

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(0,15)) & (train_set.Pclass==2)], kind="count",height=4, aspect=.7, hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Pclass']==2) & (train_set['Age'].between(15,40))], kind="count",height=4, aspect=.7,hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(40,60)) & (train_set.Pclass==2)], kind="count",height=4, aspect=.7,hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(60,80)) & (train_set.Pclass==2)], kind="count",height=4, aspect=.7,hue='Sex')
print("Casuality Rate on Age Group 0-15 on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(0,15)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(0,15))])*100,2),'%')

print("Casuality Rate on Age Group between 15 to 40 on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(15,40)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(15,40))])*100,2),'%')

print("Casuality Rate on Group of 40-60 on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(40,60)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(40,60))])*100,2),'%')

print("Casuality Rate on Group of 60-80 on 3rd Class", round(len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(60,80)) & (train_set['Survived']==0)])/len(train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(60,80))])*100,2),'%')
# Same as we can see in Graph too the Rate of Death on 3rd class with repect to Age Group

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(0,15)) & (train_set.Pclass==3)], kind="count",height=4, aspect=.7, hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Pclass']==3) & (train_set['Age'].between(15,40))], kind="count",height=4, aspect=.7,hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(40,60)) & (train_set.Pclass==3)], kind="count",height=4, aspect=.7,hue='Sex')

sns.catplot(x="Family", col="Survived",data=train_set.loc[(train_set['Age'].between(60,80)) & (train_set.Pclass==3)], kind="count",height=4, aspect=.7,hue='Sex')
sns.boxplot(x = 'Pclass', y = 'FairPerPerson',hue='Survived', data = train_set)
# Group the Deck by Class

print(train_set.groupby([ 'Pclass','Deck'])['Survived'].agg(['count','mean']))
# Lets Check the pattern of Deck on Age

sns.swarmplot(x="Deck",y="Age",hue='Sex',data=train_set,palette="Set1", split=True)
sns.swarmplot(x="Deck",y="FairPerPerson",hue='Pclass',data=train_set,palette="Set1", split=True)
plt.figure(figsize=(25, 14))

sns.catplot(x="Deck", col="Survived",data=train_set, kind="count",height=4, aspect=.7, hue='Pclass')

plt.show()
# Lets now check the records where we have Fare=0

print(train_set[(train_set['Fare']==0)].groupby(['Pclass', 'Ticket']).agg(['count']))
print(train_set.loc[(train_set['FairPerPerson']==0),['Embarked','Ticket','Age','Sex','Family','TicketHeadCount']])
train_set['FairPerPerson'].describe(percentiles=[.25, .5, .75, .90, .95, .99])
train_set.info()
train_set.loc[(train_set.Pclass==1) & (train_set.FairPerPerson>=60)]['FairPerPerson'].describe(percentiles=[.25, .5, .75, .90, .95, .99])
quantile_1, quantile_3 = np.percentile(train_set.FairPerPerson, [25, 75])
print(quantile_1, quantile_3)
iqr_value = quantile_3 - quantile_1

iqr_value
lower_bound_val = quantile_1 - (1.5 * iqr_value)

upper_bound_val = quantile_3 + (1.5 * iqr_value)

print(lower_bound_val, upper_bound_val)
plt.figure(figsize = (10, 5))

sns.kdeplot(train_set.FairPerPerson)

plt.axvline(x=-2.0, color = 'red')

plt.axvline(x=23.32, color = 'red')
train_set[(train_set.FairPerPerson >= lower_bound_val) & (train_set.FairPerPerson <= upper_bound_val)].info()
round(100*(train_set[(train_set.FairPerPerson >= lower_bound_val) & (train_set.FairPerPerson <= upper_bound_val)].count()/len(train_set.index)), 2)
round(100*(train_set[(train_set.FairPerPerson >= 0) & (train_set.FairPerPerson <= 60)].count()/len(train_set.index)), 2)
train_set_copy=train_set.loc[(train_set.FairPerPerson>0) & (train_set.FairPerPerson<=60)]

train_set_copy.head()
train_set_copy.info()
sns.boxplot(x = 'Pclass', y = 'FairPerPerson',hue='Survived', data = train_set_copy)
#train_set.drop(['Parch', 'Fare','Title','Deck','SibSp','TicketHeadCount'],axis=1,inplace=True)

train_set_copy.drop(['Parch','Ticket','Fare','Title','Deck','SibSp','TicketHeadCount'],axis=1,inplace=True)

train_set_copy.head()
#convert Pclass as category type

train_set_copy['Pclass'] = train_set_copy['Pclass'].map( {1: 'FirstClass', 2: 'SecondClass', 3:'ThirdClass'} ).astype('category')

train_set_copy.head()    
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(train_set_copy[['Sex','Pclass','Embarked']], drop_first=True)



# Adding the results to the master dataframe

train_set_copy = pd.concat([train_set_copy, dummy1], axis=1)

train_set_copy.head()
#Drop Original Columns

train_set_copy.drop(['Pclass','Sex', 'Embarked'],axis=1,inplace=True)
train_set_copy.info()
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = train_set_copy.drop(['Survived'], axis=1)



X.head()
# Putting response variable to y

y = train_set_copy['Survived']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train[['Age','FairPerPerson','Family']] = scaler.fit_transform(X_train[['Age','FairPerPerson','Family']])



X_train.head()
### Checking the Survival Rate

Survival = (sum(train_set_copy['Survived'])/len(train_set_copy['Survived'].index))*100

Survival
plt.figure(figsize = (20,10))

sns.heatmap(X_train.corr(),annot = True)

plt.show()
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
col = X_train.columns
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Embarked_Q', 1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('FairPerPerson', 1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
col = col.drop('Embarked_S', 1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})



y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score



# define function to calculate and print model metrics.

def printMetrics(y_test,y_pred):

    cp = confusion_matrix(y_test,y_pred)

    sensitivity = cp[1,1]/(cp[1,0]+cp[1,1])

    specificity =  cp[0,0]/(cp[0,1]+cp[0,0])

    precision = cp[1,1]/(cp[0,1]+cp[1,1])

    print('Confusion Matrix: \n',cp)

    print("Sensitivity: ", sensitivity)

    print("Specificity: ",specificity)

    print("AUC Score: ", roc_auc_score(y_test,y_pred)) 

    print("Precision: ",precision)

    print("f1 Score: ",f1_score(y_test,y_pred))
printMetrics(y_train_pred_final.Survived, y_train_pred_final.predicted)
from sklearn import metrics

# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.37 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
printMetrics(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
X_test[['Age','FairPerPerson','Family']] = scaler.transform(X_test[['Age','FairPerPerson','Family']])
X_test = X_test[col]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

y_test_df.head()
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Survived_Prob'})

y_pred_final
y_pred_final['final_predicted'] = y_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.37 else 0)

y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)
printMetrics(y_pred_final.Survived, y_pred_final.final_predicted)
# Importing random forest classifier from sklearn library

from sklearn.ensemble import RandomForestClassifier



# Running the random forest with default parameters.

rfc = RandomForestClassifier()
# Splitting the data into train and test, 

#we are doing a split again beacuse on top for Logistic we have done feature scalling. Here we will observe without Feature Scalling.

X_random_train, X_random_test, y__random_train, y_random_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
rfc.fit(X_random_train,y__random_train)
# Making predictions

predictions = rfc.predict(X_random_test)
# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,accuracy_score
# Let's check the report of our default model

print(classification_report(y_test,predictions))
# Printing confusion matrix

print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(2, 20, 5)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_random_train, y__random_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(100, 1500, 200)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=3)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_random_train, y__random_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with n_estimators

plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_n_estimators"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal max_features

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_features': [2, 4, 5, 7]}



# instantiate the model

rf = RandomForestClassifier(max_depth=3)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_random_train, y__random_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_features

plt.figure()

plt.plot(scores["param_max_features"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_features"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_features")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(30, 100, 20)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_random_train, y__random_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(50, 300, 50)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_random_train, y__random_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_split

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [2,3,4],

    'min_samples_leaf': range(30, 100, 20),

    'min_samples_split': range(50, 300, 50),

    'n_estimators': [200,400,600,900], 

    'max_features': [3,4,5]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
grid_search.fit(X_random_train, y__random_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=4,

                             min_samples_leaf=30, 

                             min_samples_split=150,

                             max_features=3,

                             n_estimators=900)
rfc.fit(X_random_train, y__random_train)
# predict

predictions = rfc.predict(X_random_test)
print(classification_report(y_random_test,predictions))
print(confusion_matrix(y_random_test,predictions))
print(metrics.accuracy_score(y_random_test, predictions))
X_svc_train, X_svc_test, y_svc_train, y_svc_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
scaler = StandardScaler()



X_svc_train[['Age','FairPerPerson','Family']] = scaler.fit_transform(X_svc_train[['Age','FairPerPerson','Family']])
X_svc_test[['Age','FairPerPerson','Family']] = scaler.transform(X_svc_test[['Age','FairPerPerson','Family']])
# Model building

from sklearn.svm import SVC

# instantiate an object of class SVC()

# note that we are using cost C=1

model = SVC(C = 1)



# fit

model.fit(X_svc_train, y_svc_train)



# predict

y_pred = model.predict(X_svc_test)
metrics.confusion_matrix(y_true=y_svc_test, y_pred=y_pred)
# print other metrics



# accuracy

print("accuracy", metrics.accuracy_score(y_svc_test, y_pred))



# precision

print("precision", metrics.precision_score(y_svc_test, y_pred))



# recall/sensitivity

print("recall", metrics.recall_score(y_svc_test, y_pred))

printMetrics(y_svc_test,y_pred)
# creating a KFold object with 5 splits 

from sklearn.model_selection import validation_curve

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# instantiating a model with cost=1

model = SVC(C = 1)



# computing the cross-validation scores 

# note that the argument cv takes the 'folds' object, and

# we have specified 'accuracy' as the metric



cv_results = cross_val_score(model, X_svc_train, y_svc_train, cv = folds, scoring = 'accuracy') 
# print 5 accuracies obtained from the 5 folds

print(cv_results)

print("mean accuracy = {}".format(cv_results.mean()))
# specify range of parameters (C) as a list

params = {"C": [0.1, 1, 10, 100, 1000]}



model = SVC()



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV(estimator = model, param_grid = params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                       return_train_score=True)      
model_cv.fit(X_svc_train, y_svc_train)
# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plot of C versus train and test scores



plt.figure(figsize=(8, 6))

plt.plot(cv_results['param_C'], cv_results['mean_test_score'])

plt.plot(cv_results['param_C'], cv_results['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
best_score = model_cv.best_score_

best_C = model_cv.best_params_['C']



print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))
# model with the best value of C

model = SVC(C=1)



# fit

model.fit(X_svc_train, y_svc_train)



# predict

#y_pred = model.predict(X_svc_test)
# print other metrics

from sklearn import metrics

# accuracy

print("accuracy", metrics.accuracy_score(y_svc_test, y_pred))



# precision

print("precision", metrics.precision_score(y_svc_test, y_pred))



# recall/sensitivity

print("recall", metrics.recall_score(y_svc_test, y_pred))



printMetrics(y_svc_test,y_pred)




dataModel = {'Model': ['Linear SVC','Random Forest','Logistic Regression'],

        'Accuracy': [0.8084,0.777,0.794],

        'Precision':[0.8118,0.78,0.691],

        'Recall':[0.725,0.75,0.79],

        'F1 Score':[0.766,0.73,0.736]

        }



dfm = pd.DataFrame(dataModel, columns = ['Model', 'Accuracy','Precision','Recall', 'F1 Score'])



dfm.sort_values(by = ['Accuracy'], ascending = False, inplace = True)    

dfm
plt.figure(figsize=(20, 10))

plt.subplot(2,4,1)

sns.barplot(x="Model", y="Precision",data=dfm,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Precision Comparison')

plt.subplot(2,4,2)

sns.barplot(x="Model", y="Accuracy",data=dfm,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Test Accuracy Comparison')

plt.subplot(2,4,3)

sns.barplot(x="Model", y="Recall",data=dfm,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Recall Comparison')

plt.subplot(2,4,4)

sns.barplot(x="Model", y="F1 Score",data=dfm,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA F1 Score Comparison')

plt.show()
test_set.Age.describe()
# summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(test_set.isnull().sum().sort_values(ascending=False)/len(test_set.index)), 2)
test_set['Title']=test_set['Name'].map(lambda x: x.split(',')[1].split('.')[0].lstrip())

test_set.head()
test_set['Title'].value_counts()
test_set['Title']=test_set.apply(fix_title, axis=1)

test_set['Title'].value_counts()
test_set.groupby(['Title'])['Age'].median()
test_set.groupby(['Title'])['Age'].describe()
#Impute Missing values in Age Column

master_median=test_set.loc[(test_set.Title=='Master') & ~(test_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

mr_median=test_set.loc[(test_set.Title=='Mr') & ~(test_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

miss_median=test_set.loc[(test_set.Title=='Miss') & ~(test_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')

mrs_median=test_set.loc[(test_set.Title=='Mrs') & ~(test_set.Age.isnull()),['Age']].median(axis=0, skipna=True).astype('float')
test_set.loc[(test_set.Title=='Master') & (test_set.Age.isnull()),'Age']=test_set.loc[(test_set.Title=='Master') & (test_set.Age.isnull()),'Age'].replace(np.nan,master_median.median())

test_set.loc[(test_set.Title=='Miss') & (test_set.Age.isnull()),'Age']=test_set.loc[(test_set.Title=='Miss') & (test_set.Age.isnull()),'Age'].replace(np.nan,miss_median.median())

test_set.loc[(test_set.Title=='Mrs') & (test_set.Age.isnull()),'Age']=test_set.loc[(test_set.Title=='Mrs') & (test_set.Age.isnull()),'Age'].replace(np.nan,mrs_median.median())

test_set.loc[(test_set.Title=='Mr') & (test_set.Age.isnull()),'Age']=test_set.loc[(test_set.Title=='Mr') & (test_set.Age.isnull()),'Age'].replace(np.nan,mr_median.median())
test_set.Age.isnull().sum()
# Again summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(test_set.isnull().sum().sort_values(ascending=False)/len(test_set.index)), 2)
test_set['Fare'].median()
test_set['Fare'].fillna(test_set['Fare'].median(), inplace=True)
# Again summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(test_set.isnull().sum().sort_values(ascending=False)/len(test_set.index)), 2)
test_set['Family']=test_set['SibSp']+test_set['Parch']

test_set.head()
# New column for Ticket Head Count on teh complete data

test_set['TicketHeadCount']=test_set['Ticket'].map(master['Ticket'].value_counts())

test_set.head()
#Let take fair per Person as per Ticket head Count

test_set['FairPerPerson']=test_set['Fare']/test_set['TicketHeadCount']

test_set[['FairPerPerson']].describe(percentiles=[.25, .5, .75, .90, .95, .99])
test_set_copy=test_set.loc[(test_set.FairPerPerson>0) & (test_set.FairPerPerson<=60)]

test_set_copy.head()
#train_set.drop(['Parch', 'Fare','Title','Deck','SibSp','TicketHeadCount'],axis=1,inplace=True)

test_set_copy.drop(['Parch','Ticket','Fare','Title','Cabin','SibSp','TicketHeadCount'],axis=1,inplace=True)

test_set_copy.head()
test_set_copy.drop(['PassengerId','Name'],axis=1,inplace=True)

test_set_copy.head()
#convert Pclass as category type

test_set_copy['Pclass'] = test_set_copy['Pclass'].map( {1: 'FirstClass', 2: 'SecondClass', 3:'ThirdClass'} ).astype('category')

test_set_copy.head()    
# Creating a dummy variable for some of the categorical variables and dropping the first one.

testdummy1 = pd.get_dummies(test_set_copy[['Sex','Pclass','Embarked']], drop_first=True)



# Adding the results to the master dataframe

test_set_copy = pd.concat([test_set_copy, testdummy1], axis=1)

test_set_copy.head()
#Drop Original Columns

test_set_copy.drop(['Pclass','Sex', 'Embarked'],axis=1,inplace=True)
test_set_copy[['Age','FairPerPerson','Family']] = scaler.transform(test_set_copy[['Age','FairPerPerson','Family']])


# predict

y_test_pred = model.predict(test_set_copy)

y_test_pred
test_set_copy_dummy=test_set.loc[(test_set.FairPerPerson>0) & (test_set.FairPerPerson<=60)]

predicted_df = pd.DataFrame({'PassengerId': test_set_copy_dummy['PassengerId'], 'Survived': y_test_pred})

predicted_df.head()
predicted_df.to_csv('final_kaggle_submission.csv', index=False)