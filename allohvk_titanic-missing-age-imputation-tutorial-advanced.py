import pandas as pd

import numpy as np

pd.set_option("display.max_rows", None, "display.max_columns", None)



test_data = pd.read_csv ('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")



combined=train_data.append(test_data)



##We discussed the below command in detail the data exploration notebook

print('Missing values Percentage: \n\n', round (combined.isnull().sum().sort_values(ascending=False)/len(combined)*100,1))
display(combined[(combined.Age.isnull()) & (combined.Name.str.contains('Master'))])
print(train_data[train_data.Name.str.contains('Master')]['Age'].mean())
display((combined[(combined.Age.isnull()) & (combined.Name.str.contains('Master')) & (combined.Parch==0)]))
##So there are cases (just 1) where a child is travelling without either parents..

##Probably (travelling with nanny or relatives. We will just assume that the Child

##is little senior in age and cannot be 5. We will assign the max value of Master 

##which is around 14 for such cases.

test_data.loc[test_data.PassengerId==1231,'Age']=14
train_data['Title'], test_data['Title'] = [df.Name.str.extract \

        (' ([A-Za-z]+)\.', expand=False) for df in [train_data, test_data]]
train_data.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count'])
TitleDict = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty", \

             "Don": "Royalty", "Sir" : "Royalty","Dr": "Royalty","Rev": "Royalty", \

             "Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs","Mr" : "Mr", \

             "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}
train_data['Title'], test_data['Title'] = [df.Title.map(TitleDict) for df in [train_data, test_data]]



##Let us now reprint the groups

train_data.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count'])
combined=train_data.append(test_data)

display(train_data[train_data.Title.isnull()])

display(test_data[test_data.Title.isnull()])



##There is Dona which is royalty which is not covered in test_data. Update the same

test_data.at[414,'Title'] = 'Royalty'
print ("Avg age of 'Miss' Title", round(train_data[train_data.Title=="Miss"]['Age'].mean()))



print ("Avg age of 'Miss' Title travelling without Parents", round(train_data[(train_data.Title=="Miss") & (train_data.Parch==0)]['Age'].mean()))



print ("Avg age of 'Miss' Title travelling with Parents", round(train_data[(train_data.Title=="Miss") & (train_data.Parch!=0)]['Age'].mean()), '\n')
##Let us turn our attention to the missing fare

display(combined[combined.Fare.isnull()])



##Let us get fare per person

for df in [train_data, test_data, combined]:

    df['PeopleInTicket']=df['Ticket'].map(combined['Ticket'].value_counts())

    df['FarePerPerson']=df['Fare']/df['PeopleInTicket']

##Valuecounts is the swissknife of Pandas and is deeply explained in my earlier notebook



##Just take the mean fare for the PORT S and the Pclass & fill it. Remember to consider FarePerPerson and not Fare

print('Mean fare for this category: ', train_data[(train_data.Embarked=='S') & (train_data.Pclass==3)]['FarePerPerson'].mean())
test_data.loc[test_data.Fare.isnull(), ['Fare','FarePerPerson']] = round(train_data[(train_data.Embarked=='S') & (train_data.Pclass==3) & (train_data.PeopleInTicket==1)]['Fare'].mean(),1)
display(combined[combined.Embarked.isnull()])
##Fare is 40 per person (80 for 2 people) for Pclass 1 for 2 adults. Where could they have Embarked from?



##Let us groupby Embarked and check some statistics

train_data[(train_data.Pclass==1)].groupby('Embarked').agg({'FarePerPerson': 'mean', 'Fare': 'mean', 'PassengerId': 'count'})
##Only 1 family got on at Q. Also fare is 30 per person and this is definitely not the case

##From the data below, it seems fairly obvious that the fareperperson of 40 for the 2 missing cases maps to Port C



##Let us check same data for groups of 2 adults

train_data[(train_data.Pclass==1) & (train_data.PeopleInTicket==2) & (train_data.Age>18)].groupby('Embarked').agg({'FarePerPerson': 'mean', 'Fare': 'mean', 'PassengerId': 'count'})
print(train_data[(~train_data.Cabin.isnull()) & (train_data.Pclass==1) & (train_data.PeopleInTicket==2) & (train_data.Sex=="female") & (train_data.Age>18)].groupby('Embarked').agg({'FarePerPerson': 'mean', 'Fare': 'mean', 'PassengerId': 'count'}))



##Still port C comes out as a winner in all cases. We will go ahead with this

train_data.Embarked.fillna('C', inplace=True)
print(train_data.groupby(['Pclass','Sex','Title'])['Age'].agg({'mean', 'median', 'count'}))



for df in [train_data, test_data, combined]:

    df.loc[(df.Title=='Miss') & (df.Parch!=0) & (df.PeopleInTicket>1), 'Title']="FemaleChild"



display(combined[(combined.Age.isnull()) & (combined.Title=='FemaleChild')])
##[df['Age'].fillna(df.groupby(['Pclass','Sex','Title'])['Age'].transform('mean'), inplace=True) for df in [train_data, test_data]]
##Define a group containing all the parameters you want, do a mean

##You can print the below group. This will be our lookup table

grp = train_data.groupby(['Pclass','Sex','Title'])['Age'].mean()

print(grp)
##Though it looks like a nice lookup table this will be difficult to

##'lookup'. This is because this table is actually just a series object

##like a list of ages and the index is Pclass, Sex, Title.

print('\n', 'This so called lookup table is actually similar to a list: ', type(grp))

##So the below kind of lookup will fail miserably with an error

##Try: print(grp[(grp.Pclass==2) & (grp.Sex=='male') & (grp.Title=='Master')]['Age'])
##So let us convert this 'series' object into a 'dataframe' 

##We use the re-index feature. This is an important tool

grp = train_data.groupby(['Pclass','Sex','Title'])['Age'].mean().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]



print('\n', 'We converted the series object to: ', type(grp))
##Now below statement works almost like a charm

print('\n', 'Lookup works like a charm now but not quite: ', grp[(grp.Pclass==2) & (grp.Sex=='male') & (grp.Title=='Master')]['Age'])
##There is still one minor change. The above lookup returns a series object

##You can print the type() and see for yourself.

##Even though the series object has only ONE row, however Python does not know

##all that and if you try assigning that series object to the 'age' col of a 

##'single' row, it will crib BIG-TIME. So we do one last thing..read the value

##of the first (and only row). This will be a single number

print('\n', 'Aah! Perfect: ', grp[(grp.Pclass==2) & (grp.Sex=='male') & (grp.Title=='Master')]['Age'].values[0])
##Now the above lookup works perfectly. Pass it the Pclass, Sex, Title

##It can then tell you the (mean) age for that group. Let us use it



##Define a function called fill_age. This will lookup the combination

##passed to it using above lookup table and return the value of the age associated

def fill_age(x):

    return grp[(grp.Pclass==x.Pclass)&(grp.Sex==x.Sex)&(grp.Title==x.Title)]['Age'].values[0]

##Here 'x' is the row containing the missing age. We look up the row's Pclass

##Sex and Title against the lookup table as shown previously and return the Age

##Now we have to call this fill_age function for every missing row for test, train



train_data['Age'], test_data['Age'] = [df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1) for df in [train_data, test_data]]

##This line is explained in the next cell



##End by combining the test and training data

combined=train_data.append(test_data)