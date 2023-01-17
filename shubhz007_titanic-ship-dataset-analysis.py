#Loading all the necessary libraries

import unicodecsv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for visualisation

import seaborn as sns #for visualisation

%matplotlib inline
#Reading data from Comma Separated Values files

#Creating the dataframe to get the clean structure of the data and its description, pandas will help us save time in this

titanic_df = pd.read_csv('../input/train.csv')



col_names=titanic_df.columns.tolist()

print("column names:")

print(col_names)
print("Sample data:")

titanic_df.head(6)
#Dataframe Information

titanic_df.describe()

#From the description we can see that we have mean of all the columns, since we have mean of age also then we can replace the missing values of age with its mean value
#Dataframe datatype information

titanic_df.dtypes
titanic_df['PassengerId'].count()
unique_PassengerId = titanic_df['PassengerId'].unique()

len(unique_PassengerId)
#Method to find out missing records

def missingData(param):

    return titanic_df[(titanic_df[param].isnull())]['PassengerId'].count()

    

    

def finding_missing_record():

    missingPassengers = missingData('PassengerId') 

    missingSurvived = missingData('Survived')

    missingPclass = missingData('Pclass')

    missingName = missingData('Name')

    missingSex = missingData('Sex')

    missingAge = missingData('Age')

    missingSibSp = missingData('SibSp')

    missingParch = missingData('Parch')

    missingTicket = missingData('Ticket')

    missingFare = missingData('Fare')

    #missingCabin = missingData('Cabin')

    missingEmbarked = missingData('Embarked')

    missing_records=pd.Series([missingPassengers,missingSurvived,missingPclass,missingName,missingSex,missingAge,missingSibSp,missingParch,missingTicket,missingFare,missingEmbarked],index=['missingPassengers','missingSurvived','missingPclass','missingName','missingSex','missingAge','missingSibSp','missingParch','missingTicket','missingFare','missingEmbarked'])

    missing_records_df=pd.DataFrame(missing_records,columns=['No. of missing records'])

    return missing_records_df
#Finding the missing records



finding_missing_record()
#This graph shows ages of passengers on ship

titanic_df['Age'].plot(kind="hist",title = "Ages of all the passengers on ship",figsize = (10,10)).set_xlabel("Agesg in years")
#This graph shows ages of passengers on ship

sns.set_style("whitegrid")

sns.violinplot(x=titanic_df["Age"])
titanic_df.groupby('Pclass')['Age'].mean()
titanic_df.groupby('Pclass')['Age'].mean().plot(kind = 'bar', figsize=(10,10), title="Mean age of passengers travelling").set_ylabel("Mean")
titanic_df.groupby(['Pclass', 'Survived'])['Age'].mean()
titanic_df.groupby(['Pclass', 'Survived'])['Age'].mean().plot(kind = 'bar', figsize=(10,10) , legend="True", title="Mean age of passengers survived or perished from each class").set_ylabel("Mean")
#Method to replace values

def replace_all_null(grp, param):

    grp[param] = np.where(((grp[param] ==0) | (grp[param].isnull())), grp[param].mean(),grp[param])

    return grp
#This graph shows ages of passengers on ship

titanic_df['Age'].plot(kind="hist",title = "Ages of all the passengers on ship before replacing null ",figsize = (10,10)).set_xlabel("Agesg in years")
titanic_df = titanic_df.groupby(['Pclass','Survived']).apply(replace_all_null, "Age")
#This graph shows ages of passengers on ship after replacing nulls

titanic_df['Age'].plot(kind="hist",title = "Ages of all the passengers on ship after replacing null",figsize = (10,10)).set_xlabel("Agesg in years")
titanic_df.groupby(['Pclass', 'Embarked'])['Fare'].mean()
titanic_df.groupby(['Pclass', 'Embarked'])['Fare'].mean().plot(kind = 'bar', figsize=(10,10) , title="Mean fare of of each class from each station").set_ylabel("Mean Fare")
titanic_df[(titanic_df['Embarked'].isnull())]
titanic_df[((titanic_df['Fare'] > 79.50 ) & (titanic_df['Fare'] < 80.50) & (titanic_df['Pclass'] == 1) & (titanic_df['Cabin'].notnull()))]
titanic_df[(titanic_df['Embarked'].isnull())].fillna('S')
#Finding the missing records



finding_missing_record()
#Dataframe datatype information

print ("Data types before improvising")

titanic_df.dtypes
#changing the data types of columns Age and Survived

titanic_df['Survived'] = titanic_df['Survived'].astype(bool)

titanic_df['Age'] = titanic_df[('Age')].astype(int)
#Dataframe datatype information

print ("After improving datatypes")

titanic_df.dtypes
#Method to find out records with value as 0

def zeroValueData(param):

    return titanic_df[(titanic_df[param] == 0)]['PassengerId'].count()

    

    

def zeroValueData_record():

    zeroValuePassengers = zeroValueData('PassengerId') 

    zeroValuePclass = zeroValueData('Pclass')

    zeroValueAge = zeroValueData('Age')

    zeroValueTicket = zeroValueData('Ticket')

    zeroValueFare = zeroValueData('Fare')

    zeroValue_records=pd.Series([zeroValuePassengers,zeroValuePclass,zeroValueAge,zeroValueTicket,zeroValueFare],index=['zeroValuePassengers','zeroValuePclass','zeroValueAge','zeroValueTicket','zeroValueFare'])

    zero_records_df=pd.DataFrame(zeroValue_records,columns=['No. of zero value records'])

    return zero_records_df
#Finding the 0 value records

zeroValueData_record()
titanic_df.groupby(['Pclass'])['Fare'].mean()
titanic_df.groupby(['Pclass'])['Fare'].mean().plot(kind = 'bar', figsize=(10,10) , title="Mean fare of of each class").set_ylabel("Mean Fare")
titanic_df = titanic_df.groupby(['Pclass']).apply(replace_all_null, "Fare")

titanic_df = titanic_df.groupby(['Pclass','Survived']).apply(replace_all_null, "Age")
#Finding the 0 value records

zeroValueData_record()
#Again printing the sample data

print("Sample data:")

titanic_df.head(50)
#Total number of passengers travelling

print ('Total number of passengers travelling = ',titanic_df['PassengerId'] .count())



#Total number of people srvived

print ('Total number of passengers survived = ',(titanic_df['Survived'] == 1).sum())



#Total number of casualities

print ('Total number of passengers died = ',(titanic_df['Survived'] == 0).sum())



#Mean of passengers srvived

print ('Mean of passengers srvived = ',(titanic_df['Survived'] == 1).mean())



#Mean of casualities

print ('Mean of passengers died = ',(titanic_df['Survived'] == 0).mean())



#Total number of females travelling

print ('Total number of females travelling', (titanic_df['Sex'] == 'female').sum())



#Total number of males travelling

print('Total number of males travelling', (titanic_df['Sex'] == 'male').sum())



#Total number of females survived

print ('Total number of females survived', ((titanic_df['Sex'] == 'female') & (titanic_df['Survived'] == 1)).sum())



#Total number of females died

print ('Total number of females died', ((titanic_df['Sex'] == 'female') & (titanic_df['Survived'] == 0)).sum())



#Total number of males survived

print ('Total number of males survived', ((titanic_df['Sex'] == 'male') & (titanic_df['Survived'] == 1)).sum())



#Total number of males died

print ('Total number of males died', ((titanic_df['Sex'] == 'male') & (titanic_df['Survived'] == 0)).sum())
#This graph shows number of people survived and casualities

titanic_df.groupby(['Survived'])['PassengerId'].count().plot(kind="bar", figsize = (10,10), grid = 10 ,logy = 0, title = "No of survivors and casualities ").set_ylabel("Frequency")
sns.set(style="ticks", color_codes=True)

sns.pairplot(titanic_df, vars=["Age", "Survived","Pclass"] , size =3, diag_kind="kde")
titanic_df.groupby('Sex')['Survived'].sum().plot(kind="bar" ,figsize = (10,10), grid = 10 ,logy = 0, title = "No of survivors and their sex ").set_ylabel("Frequency")
titanic_df.groupby('Embarked')['Survived'].sum().plot(kind="pie", autopct='%1.1f%%' , legend="True")
titanic_df.groupby('Embarked')['Survived'].sum().plot(kind="bar" ,figsize = (10,10), grid = 10 ,logy = 0, title = "No of survivors from each station ").set_ylabel("Frequency")
titanic_df.groupby('Pclass')['Survived'].sum().plot(kind="pie", autopct='%1.1f%%')
titanic_df.groupby('Pclass')['Survived'].sum().plot(kind="bar" ,figsize = (10,10), grid = 10 ,logy = 0, title = "No of survivors from each Pclass ").set_ylabel("Frequency")
titanic_df[(titanic_df['Age'] < 10)].groupby('Survived')['Age'].plot(kind="hist", alpha =0.7,legend = "True", figsize = (10,10),title = "No of survivors/casualities under age 10 ")
titanic_df[(titanic_df['Age'] > 10) & (titanic_df['Age'] < 20)].groupby('Survived')['Age'].plot(kind="hist",figsize = (10,10), legend="True",alpha=0.7,title = "No of survivors/casualities between age 10-20 ")
titanic_df[(titanic_df['Age'] > 20) & (titanic_df['Age'] < 40)].groupby('Survived')['Age'].plot(kind="hist",figsize = (10,10), legend="True",alpha=0.7,title = "No of survivors/casualities between age 20-40 ")
titanic_df[(titanic_df['Age'] > 40) & (titanic_df['Age'] < 60)].groupby('Survived')['Age'].plot(kind="hist",alpha=0.7,figsize = (10,10), legend="True",title = "No of survivors/casualities between age 40-60 ")
titanic_df[(titanic_df['Age'] > 60) ].groupby('Survived')['Age'].plot(kind="hist", alpha=0.7,legend="True",figsize = (10,10),use_index="True",title = "No of survivors/casualities above age 60 ")
#This graph shows that 

titanic_df.groupby(['Pclass','Sex'])['Survived'].sum().plot(kind="bar", figsize = (10,10), grid = 10 ,logy = 0, title = "No of people survived along with their Pclass and Sex").set_ylabel("Frequency")
# Fetching the data for survivors and casualities

survivor = titanic_df[(titanic_df['Survived'] == 1)]['Age']



casualities = titanic_df[(titanic_df['Survived'] == 0)]['Age']
# Giving the description of surivors

survivor.describe()
# Giving the description of casualities

casualities.describe()
survivor.plot(kind="hist" , figsize=(10,10), title = "Survivors and their age").set_xlabel("Age in years")
casualities.plot(kind="hist" , figsize=(10,10), title = "Casualities and their age").set_xlabel("Age in years")
SE = np.sqrt((survivor.var()/survivor.count())+(casualities.var()/casualities.count()))

SE
T = (survivor.mean() - casualities.mean() )/SE

T
DegreeOfFreedom = survivor.count() + casualities.count() - 2

DegreeOfFreedom