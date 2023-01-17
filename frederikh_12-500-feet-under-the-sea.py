# Standard Libraries

import numpy as np

import pandas as pd



# Regex for name exploration

import re as re



# Visualzation

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns



# Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_set = pd.read_csv('../input/train.csv')



#Display first 5 rows (4+description)

train_set.head(5)
test_set = pd.read_csv("../input/test.csv")

test_set_orig = test_set.copy()

#Display first 5 rows (4+description)

test_set.head(5)
# First entry Name of the collumn, second entry number of row (excluding the title row)

print(train_set['Name'][3])
# Check Age for runaway data

fig1, axis1 = plt.subplots(1,1,figsize=(18,4))

sns.regplot(x=train_set["PassengerId"], y=train_set["Age"])

plt.show()



fig2, axis2 = plt.subplots(1,1,figsize=(18,4))

sns.regplot(x=test_set["PassengerId"], y=test_set["Age"])

plt.show()
# Check Fare for runaway data



# use the function regplot to make a scatterplot

fig1, axis1 = plt.subplots(1,1,figsize=(18,4))

sns.regplot(x=train_set["PassengerId"], y=train_set["Fare"])

plt.show()



fig2, axis2 = plt.subplots(1,1,figsize=(18,4))

sns.regplot(x=test_set["PassengerId"], y=test_set["Fare"])

plt.show()
train_set.loc[train_set['Fare'] > 300]
test_set.loc[test_set['Fare'] > 300]
# Show with outliers

PClass_palette = {1:"b", 2:"y", 3:"r"}

fig1, axis1 = plt.subplots(1,1,figsize=(10,8))

sns.boxplot( x=train_set["Pclass"], y=train_set["Fare"], palette=PClass_palette, showfliers=True)

plt.show()



# Show without outliers

fig2, axis2 = plt.subplots(1,1,figsize=(10,8))

sns.boxplot( x=train_set["Pclass"], y=train_set["Fare"], palette=PClass_palette, showfliers=False)

plt.show()
# Make default histogram of Fare

fig1, axis1 = plt.subplots(1,1,figsize=(10,8))

sns.distplot( train_set.dropna()["Fare"], bins=25)

plt.xlim(0,)

plt.show()
# Show with outliers

PClass_palette = {1:"b", 2:"y", 3:"r"}

fig1, axis1 = plt.subplots(1,1,figsize=(10,8))

sns.boxplot( x=test_set["Pclass"], y=test_set["Fare"], palette=PClass_palette,  showfliers=True)

plt.show()



# Show without outliers

fig2, axis2 = plt.subplots(1,1,figsize=(10,8))

sns.boxplot( x=test_set["Pclass"], y=test_set["Fare"], palette=PClass_palette, showfliers=False)

plt.show()
# Color Palette

PClass_palette = {1:"b", 2:"y", 3:"r"}



fig1, axis2 = plt.subplots(1,1,figsize=(10,8))



train_set.Fare[train_set.Pclass == 1].plot(kind='kde')    

train_set.Fare[train_set.Pclass == 2].plot(kind='kde')

train_set.Fare[train_set.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Fare")    

plt.title("Fare Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')

# Limit to important x

plt.xlim(0,300)

plt.ylim(0,)

plt.palette=PClass_palette
PClass1Fare = train_set[train_set['Pclass'] == 1]



fig1, axis1 = plt.subplots(1,1,figsize=(18,4))

sns.regplot(x=PClass1Fare["PassengerId"], y=PClass1Fare["Fare"])

plt.title('Fare in First Class')

plt.show()
train_set.isnull().sum()
test_set.isnull().sum()
fig1, axis2 = plt.subplots(1,1,figsize=(10,8))



sns.boxplot(x=train_set["Embarked"], y=train_set["Fare"], hue=train_set["Pclass"],

            palette=PClass_palette,showfliers=False)

plt.plot([80, -80], [80, 80], linewidth=3, linestyle='--')

plt.show()
fig1, axis2 = plt.subplots(1,1,figsize=(10,8))

sns.boxplot(x=test_set["Embarked"], y=test_set["Fare"], hue=test_set["Pclass"],

            palette=PClass_palette,showfliers=False)

plt.plot([80, -80], [80, 80], linewidth=3, linestyle='--')

plt.show()
train_set[pd.isnull(train_set['Embarked'])]
# Change NaN values in Embarked to 'C'

train_set[['Embarked']] = train_set[['Embarked']].fillna(value='C')
train_set.iloc[[61]]
train_set.iloc[[829]]
test_set[pd.isnull(test_set['Fare'])]
meanFare = train_set[train_set['Embarked'] == 'S'][train_set['Pclass'] == 3][train_set['Age'] >= 40][train_set['SibSp'] == 0][train_set['Parch'] == 0]

meanFare['Fare'].describe()
# Change NaN value in Fare to 7.5

test_set[['Fare']] = test_set[['Fare']].fillna(value=7.5)
test_set.iloc[[152]]
train_set.isnull().sum()
test_set.isnull().sum()
# Add an empty collumn with the title 'AgeBand'

train_set['AgeBand'] = np.nan

test_set['AgeBand'] = np.nan
#Unknown Age ranges before classification:

print("[Train] Unknown Entries for 'AgeBand' before classification:",

      train_set.isnull().sum(axis=0)['AgeBand'])

print("[Test] Unknown Entries for 'AgeBand' before classification:",

      test_set.isnull().sum(axis=0)['AgeBand'])



print("")





# If Age is known, add the right tag

for i in train_set['Age'].iteritems():

    if i[1]<=15:

        train_set.loc[i[0],'AgeBand'] = 'Underage'

    elif 15<i[1]<=30:

        train_set.loc[i[0],'AgeBand'] = 'Young'

    elif 30<i[1]<=45:

        train_set.loc[i[0],'AgeBand'] = 'MidAge'

    elif 45<i[1]<=60:

        train_set.loc[i[0],'AgeBand'] = 'Old'

    elif 60<i[1]:

        train_set.loc[i[0],'AgeBand'] = 'VeryOld'

    else:

        pass

    

for i in test_set['Age'].iteritems():

    if i[1]<=15:

        test_set.loc[i[0],'AgeBand'] = 'Underage'

    elif 15<i[1]<=30:

        test_set.loc[i[0],'AgeBand'] = 'Young'

    elif 30<i[1]<=45:

        test_set.loc[i[0],'AgeBand'] = 'MidAge'

    elif 45<i[1]<=60:

        test_set.loc[i[0],'AgeBand'] = 'Old'

    elif 60<i[1]:

        test_set.loc[i[0],'AgeBand'] = 'VeryOld'

    else:

        pass



#Unknown Age ranges after using 'age':

print("[Train] Unknown Entries for 'AgeBand' after using known 'Age':",

      train_set.isnull().sum(axis=0)['AgeBand'])

print("[Test] Unknown Entries for 'AgeBand' after using known 'Age':",

      test_set.isnull().sum(axis=0)['AgeBand'])
fig1, axis2 = plt.subplots(1,1,figsize=(15,4))

sns.boxplot(x=train_set["AgeBand"], y=train_set["SibSp"],

            hue=train_set["Pclass"], palette=PClass_palette,

            showfliers=True,order=["Underage","Young","MidAge","Old","VeryOld"]).set_title("Train Set") ;



fig2, axis1 = plt.subplots(1,1,figsize=(15,4))

sns.boxplot(x=test_set["AgeBand"], y=test_set["SibSp"],

            hue=test_set["Pclass"], palette=PClass_palette,

            showfliers=True,order=["Underage","Young","MidAge","Old","VeryOld"]).set_title("Test Set") ;
# First Method:  If there are more than two Spouses/Siblings it is likely,

#                that it is an underage passenger with many siblings.

#                Adult passengers are mostly alone, with one sibling/spouse

#                or with many children rather than siblings.

for i in train_set['AgeBand'].iteritems():

    if type(train_set['AgeBand'][i[0]]) == float and train_set['SibSp'][i[0]] > 2:

        train_set.loc[i[0],'AgeBand'] = 'Underage'

        

for i in test_set['AgeBand'].iteritems():

    if type(test_set['AgeBand'][i[0]]) == float and test_set['SibSp'][i[0]] > 2:

        test_set.loc[i[0],'AgeBand'] = 'Underage'

        

#Unknown Age ranges after Method One:

print("[Train] Unknown Entries for 'AgeBand' after using method one:",

      train_set.isnull().sum(axis=0)['AgeBand'])

print("[Test] Unknown Entries for 'AgeBand' after using method one:",

      test_set.isnull().sum(axis=0)['AgeBand'])
fig1, axis2 = plt.subplots(1,1,figsize=(15,4))

sns.boxplot(x=train_set["AgeBand"], y=train_set["Parch"],

            hue=train_set["Pclass"], palette=PClass_palette,

            showfliers=True,order=["Underage","Young","MidAge","Old","VeryOld"]).set_title("Train Set") ;



fig2, axis1 = plt.subplots(1,1,figsize=(15,4))

sns.boxplot(x=test_set["AgeBand"], y=test_set["Parch"],

            hue=test_set["Pclass"], palette=PClass_palette,

            showfliers=True,order=["Underage","Young","MidAge","Old","VeryOld"]).set_title("Test Set") ;
# Second Method: If there are two Parents/Children the passenger will be categorized as 'underage'



for i in train_set['AgeBand'].iteritems():

    if type(train_set['AgeBand'][i[0]]) == float and train_set['Parch'][i[0]] == 2:

        train_set.loc[i[0],'AgeBand'] = 'Underage'

        

for i in test_set['AgeBand'].iteritems():

    if type(test_set['AgeBand'][i[0]]) == float and test_set['Parch'][i[0]] == 2:

        test_set.loc[i[0],'AgeBand'] = 'Underage'

        

#Unknown Age ranges after Method One:

print("[Train] Unknown Entries for 'AgeBand' after using method two:",

      train_set.isnull().sum(axis=0)['AgeBand'])

print("[Test] Unknown Entries for 'AgeBand' after using method two:",

      test_set.isnull().sum(axis=0)['AgeBand'])
# plot real ages

fig1, axis1 = plt.subplots(1,1,figsize=(10,8));

sns.distplot( train_set.dropna()["Age"], bins=20);

plt.xlim(0,);

plt.show();



# plot ageband

fig2, axis2 = plt.subplots(1,1,figsize=(10,8));

sns.countplot(x=train_set["AgeBand"],order=["Underage","Young","MidAge","Old","VeryOld"]);

train_set.dropna()["Age"].describe()
train_set.groupby('AgeBand')['Age'].mean()
# eeny meeny

eeny = np.random.randint(train_set['Age'].mean() - train_set['Age'].std(),

                         train_set['Age'].mean() + train_set['Age'].std(),

                         size = train_set['Age'].isnull().sum())

meeny = np.random.randint(train_set['Age'].mean() - train_set['Age'].std(),

                         train_set['Age'].mean() + train_set['Age'].std(),

                         size = test_set['Age'].isnull().sum())



# assign nan values of age with eeny and meeny

train_set['Age'][np.isnan(train_set['Age'])] = eeny;

test_set['Age'][np.isnan(test_set['Age'])] = meeny;



# make it into integers

train_set['Age'] = train_set['Age'].astype(int);

test_set['Age'] = test_set['Age'].astype(int);
# Repeat assignment of AgeBand using Ages

for i in train_set['Age'].iteritems():

    if type(train_set['AgeBand'][i[0]]) == float:

        if i[1]<=15:

            train_set.loc[i[0],'AgeBand'] = 'Underage'

        elif 15<i[1]<=30:

            train_set.loc[i[0],'AgeBand'] = 'Young'

        elif 30<i[1]<=45:

            train_set.loc[i[0],'AgeBand'] = 'MidAge'

        elif 45<i[1]<=60:

            train_set.loc[i[0],'AgeBand'] = 'Old'

        elif 60<i[1]:

            train_set.loc[i[0],'AgeBand'] = 'VeryOld'

        else:

            pass

    

for i in test_set['Age'].iteritems():

    if type(test_set['AgeBand'][i[0]]) == float:

        if i[1]<=15:

            test_set.loc[i[0],'AgeBand'] = 'Underage'

        elif 15<i[1]<=30:

            test_set.loc[i[0],'AgeBand'] = 'Young'

        elif 30<i[1]<=45:

            test_set.loc[i[0],'AgeBand'] = 'MidAge'

        elif 45<i[1]<=60:

            test_set.loc[i[0],'AgeBand'] = 'Old'

        elif 60<i[1]:

            test_set.loc[i[0],'AgeBand'] = 'VeryOld'

        else:

            pass



#Unknown Age ranges after using 'age':

print("[Train] Unknown Entries for 'AgeBand' after using random 'Age':",

      train_set.isnull().sum(axis=0)['AgeBand'])

print("[Test] Unknown Entries for 'AgeBand' after using random 'Age':",

      test_set.isnull().sum(axis=0)['AgeBand'])
# Total number of male and female passengers

male_total = (train_set['Sex'] == 'male').sum()

female_total = (train_set['Sex'] == 'female').sum()



# Survivors per gender

male_survived = (train_set[train_set['Survived']==1]['Sex']=='male').sum()

female_survived = (train_set[train_set['Survived']==1]['Sex']=='female').sum()



# Deaths per gender

male_dead = (train_set[train_set['Survived']==0]['Sex']=='male').sum()

female_dead = (train_set[train_set['Survived']==0]['Sex']=='female').sum()



# Plotting

Gender_data = train_set.groupby(['Sex', 'Survived'])['Sex'].count().unstack('Survived')

plot1 = Gender_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'], figsize=(10,8))

plot1.legend(["Dead","Survived"])



# Percentage survived by gender

print(100/male_total*male_survived,"% of the male passengers survived.")

print(100/female_total*female_survived,"% of the the female passengers survived.")
# Total number of children and adults

children_total = (train_set['Age'].dropna(axis=0, how='any') <= 15).sum()

adult_total = (train_set['Age'].dropna(axis=0, how='any') > 15).sum()



# Survivors per agegroup

children_survived = (train_set[train_set['Survived']==1]['Age']<=15).sum()

adult_survived = (train_set[train_set['Survived']==1]['Age']>15).sum()



#percentage of survivors by age above and below 14

print(100/adult_total*adult_survived,"% of the adult passengers survived.")

print(100/children_total*children_survived,"% of the the passengers below the age of 15 survived.")
fig2, axis2 = plt.subplots(1,1,figsize=(10,8))

sns.barplot( x="AgeBand",

            y="Survived",

            order=["Underage","Young","MidAge","Old","VeryOld"],

           data=train_set)
# Plotting total

Money_data = train_set.groupby(['Pclass', 'Survived'])['Pclass'].count().unstack('Survived')

plot1 = Money_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])

plot1.legend(["Dead","Survived"])
# Plotting gender and class

Money_data = train_set.groupby(['Pclass', 'Survived', 'Sex'])['Pclass'].count().unstack('Survived')

plot1 = Money_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])

plot1.legend(["Dead","Survived"])
# Total number of children in third class

children_rich_total = (train_set[train_set['Pclass']==1]['Age']<=15).sum()

children_mid_total = (train_set[train_set['Pclass']==2]['Age']<=15).sum()

children_poor_total = (train_set[train_set['Pclass']==3]['Age']<=15).sum()



# Survivors of third class children

children_rich_survived = ((train_set['Survived']==1) & (train_set['Age']<=15) & (train_set['Pclass']==1)).sum()

children_mid_survived = ((train_set['Survived']==1) & (train_set['Age']<=15) & (train_set['Pclass']==2)).sum()

children_poor_survived = ((train_set['Survived']==1) & (train_set['Age']<=15) & (train_set['Pclass']==3)).sum()



#percentage of survivors by age above and below 14

print(100/children_rich_total*children_rich_survived,"% of the the passengers below the age of 15 in the first class survived.")

print(100/children_mid_total*children_mid_survived,"% of the the passengers below the age of 15 in the second class survived.")

print(100/children_poor_total*children_poor_survived,"% of the the passengers below the age of 15 in the third class survived.")
train_set["Title"] = train_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_set["Title"] = test_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_set['Title'], train_set['Sex'])
pd.crosstab(test_set['Title'], test_set['Sex'])
train_set['Title'] = train_set['Title'].replace('Mlle', 'Miss')

train_set['Title'] = train_set['Title'].replace('Ms', 'Miss')

train_set['Title'] = train_set['Title'].replace('Mme', 'Mrs')



test_set['Title'] = test_set['Title'].replace('Mlle', 'Miss')

test_set['Title'] = test_set['Title'].replace('Ms', 'Miss')

test_set['Title'] = test_set['Title'].replace('Mme', 'Mrs')
pd.crosstab(train_set['Title'], train_set['Sex'])
train_set['Title'] = train_set['Title'].replace(['Lady', 'Countess','Capt',

                                                 'Col','Don', 'Dr', 'Major',

                                                 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



test_set['Title'] = test_set['Title'].replace(['Lady', 'Countess','Capt',

                                                 'Col','Don', 'Dr', 'Major',

                                                 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
pd.crosstab(train_set['Title'], train_set['Sex'])
# Plotting Survival and title

Title_data = train_set.groupby(['Title', 'Survived'])['Title'].count().unstack('Survived')

plot1 = Title_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])

plot1.legend(["Dead","Survived"])
train_set["Surname"] = train_set.Name.str.split(',').str.get(0)

test_set["Surname"] = test_set.Name.str.split(',').str.get(0)



# Add an empty collumn with the title 'Ethnicity'

train_set['Ethnicity'] = np.nan

test_set['Ethnicity'] = np.nan
train_set['Surname'].head(15)
train_set.isnull().sum(axis=0)['Ethnicity']
# Training Set

for i in train_set['Surname'].iteritems():

    # English

    if i[1].endswith('ley') or i[1].endswith("ers") or i[1].endswith("en") or i[1].endswith("ton") or i[1].endswith("ing") or i[1].endswith("ings"):

        train_set.loc[i[0],'Ethnicity'] = 'English'

    # Irish

    if i[1].startswith('Mc') or i[1].startswith("O'") or i[1].startswith("Fitz") or i[1].endswith("ty"):

        train_set.loc[i[0],'Ethnicity'] = 'Irish'

    # Scandinavian

    if i[1].endswith('son') or i[1].endswith('sen') or i[1].endswith('la') or i[1].endswith('lä') or i[1].endswith('nen'):

        train_set.loc[i[0],'Ethnicity'] = 'Scandinavian'

    # Scottish

    if i[1].startswith('Mac') or i[1].startswith('Mag'):

        train_set.loc[i[0],'Ethnicity'] = 'Scottish'

    # EasterEuropean

    if i[1].endswith('ski') or i[1].endswith('vic') or i[1].endswith('off') or i[1].endswith('cki') or i[1].endswith('dzki') or i[1].endswith('wicz') or i[1].endswith('czyk') or i[1].endswith('czak') or i[1].endswith('czek') or i[1].endswith('ek') or i[1].endswith('ak') or i[1].endswith('vić') or i[1].endswith('ov') or i[1].endswith('yev') or i[1].endswith('enko') or i[1].endswith('shin'):

        train_set.loc[i[0],'Ethnicity'] = 'EasternEuropean'

    # SouthernEuropean

    if i[1].startswith('De ') or i[1].startswith('Di ') or i[1].startswith('D ') or i[1].endswith('as') or i[1].endswith('is') or i[1].endswith('us') or i[1].endswith('es') or i[1].endswith('ez') or i[1].endswith('akis') or i[1].endswith('idis') or i[1].endswith('opoulos') or i[1].endswith('ni') or i[1].endswith('no') or i[1].endswith('zzi') or i[1].endswith('tti') or i[1].endswith('ero') or i[1].endswith('eri') or i[1].endswith('elli') or i[1].endswith('er') or i[1].endswith('ossi') or i[1].endswith('aldi'):

        train_set.loc[i[0],'Ethnicity'] = 'SouthernEuropean'

    # CentralEuropean

    if i[1].startswith('van ') or i[1].startswith('von ') or i[1].endswith('che') or i[1].endswith('elle') or i[1].endswith('er') or i[1].endswith('stein') or i[1].endswith('baum') or i[1].endswith('berg'):

        train_set.loc[i[0],'Ethnicity'] = 'CentralEuropean'



# Test Set 

for i in test_set['Surname'].iteritems():

    # English

    if i[1].endswith('ley') or i[1].endswith("ers") or i[1].endswith("en") or i[1].endswith("ton") or i[1].endswith("ing") or i[1].endswith("ings"):

        test_set.loc[i[0],'Ethnicity'] = 'English'

    # Irish

    if i[1].startswith('Mc') or i[1].startswith("O'") or i[1].startswith("Fitz") or i[1].endswith("ty"):

        test_set.loc[i[0],'Ethnicity'] = 'Irish'

    # Scandinavian

    if i[1].endswith('son') or i[1].endswith('sen') or i[1].endswith('la') or i[1].endswith('lä') or i[1].endswith('nen'):

        test_set.loc[i[0],'Ethnicity'] = 'Scandinavian'

    # Scottish

    if i[1].startswith('Mac') or i[1].startswith('Mag'):

        test_set.loc[i[0],'Ethnicity'] = 'Scottish'

    # EasterEuropean

    if i[1].endswith('ski') or i[1].endswith('vic') or i[1].endswith('off') or i[1].endswith('cki') or i[1].endswith('dzki') or i[1].endswith('wicz') or i[1].endswith('czyk') or i[1].endswith('czak') or i[1].endswith('czek') or i[1].endswith('ek') or i[1].endswith('ak') or i[1].endswith('vić') or i[1].endswith('ov') or i[1].endswith('yev') or i[1].endswith('enko') or i[1].endswith('shin'):

        test_set.loc[i[0],'Ethnicity'] = 'EasternEuropean'

    # SouthernEuropean

    if i[1].startswith('De ') or i[1].startswith('Di ') or i[1].startswith('D ') or i[1].endswith('as') or i[1].endswith('is') or i[1].endswith('us') or i[1].endswith('es') or i[1].endswith('ez') or i[1].endswith('akis') or i[1].endswith('idis') or i[1].endswith('opoulos') or i[1].endswith('ni') or i[1].endswith('no') or i[1].endswith('zzi') or i[1].endswith('tti') or i[1].endswith('ero') or i[1].endswith('eri') or i[1].endswith('elli') or i[1].endswith('er') or i[1].endswith('ossi') or i[1].endswith('aldi'):

        test_set.loc[i[0],'Ethnicity'] = 'SouthernEuropean'

    # CentralEuropean

    if i[1].startswith('van ') or i[1].startswith('von ') or i[1].endswith('che') or i[1].endswith('elle') or i[1].endswith('er') or i[1].endswith('stein') or i[1].endswith('baum') or i[1].endswith('berg'):

        test_set.loc[i[0],'Ethnicity'] = 'CentralEuropean'
train_set.isnull().sum(axis=0)['Ethnicity']
# Plotting Survival and title

Ethn_data = train_set.groupby(['Ethnicity', 'Survived'])['Ethnicity'].count().unstack('Survived')

plot1 = Ethn_data.plot(kind='bar', stacked=True, color=['r','b'], legend=['Dead', 'Alive'])

plot1.legend(["Dead","Survived"])
train_set['FamilySize'] = train_set['SibSp'] + train_set['Parch'] + 1

test_set['FamilySize'] = test_set['SibSp'] + test_set['Parch'] + 1
fig, axis1 = plt.subplots(1,1,figsize=(30,10))

average_fam = train_set[["FamilySize", "Survived"]].groupby(['FamilySize'],as_index=False).mean()

sns.barplot(x='FamilySize', y='Survived', data=average_fam)
# Scaling Sex

train_set['SexScaled'] = train_set['Sex'].map( {'female': 0, 'male': 1} )

test_set['SexScaled'] = test_set['Sex'].map( {'female': 0, 'male': 1} )

    

# Scaling Titles

title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train_set['TitleScaled'] = train_set['Title'].map(title_map).astype(int)

train_set['TitleScaled'] = train_set['TitleScaled'].fillna(0)

test_set['TitleScaled'] = test_set['Title'].map(title_map).astype(int)

test_set['TitleScaled'] = test_set['TitleScaled'].fillna(0)

    

# Mapping Embarked

train_set['EmbarkedScaled'] = train_set['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_set['EmbarkedSclaed'] = test_set['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



# Mapping Fare

train_set.loc[ train_set['Fare'] <= 7.91, 'Fare'] = 0

train_set.loc[(train_set['Fare'] > 7.91) & (train_set['Fare'] <= 14.454), 'Fare'] = 1

train_set.loc[(train_set['Fare'] > 14.454) & (train_set['Fare'] <= 31), 'Fare'] = 2

train_set.loc[ train_set['Fare'] > 31, 'Fare'] = 3

train_set['Fare'] = train_set['Fare'].astype(int)

test_set.loc[ test_set['Fare'] <= 7.91, 'Fare'] = 0

test_set.loc[(test_set['Fare'] > 7.91) & (test_set['Fare'] <= 14.454), 'Fare'] = 1

test_set.loc[(test_set['Fare'] > 14.454) & (test_set['Fare'] <= 31), 'Fare'] = 2

test_set.loc[ test_set['Fare'] > 31, 'Fare'] = 3

test_set['Fare'] = test_set['Fare'].astype(int)



# Scaling Ethnicity

ethnic_map = {"English": 1, "Irish": 2, "Scottish": 3,

             "Scandinavian": 4, "EasternEuropean": 5,

             "SouthernEuropean": 6, "CentralEuropean": 7}

train_set['EthnicityScaled'] = train_set['Ethnicity'].map(ethnic_map)

train_set['EthnicityScaled'] = train_set['EthnicityScaled'].fillna(0)

train_set.EthnicityScaled = train_set.EthnicityScaled.astype(int)

test_set['EthnicityScaled'] = test_set['Ethnicity'].map(ethnic_map)

test_set['EthnicityScaled'] = test_set['EthnicityScaled'].fillna(0)

test_set.EthnicityScaled = test_set.EthnicityScaled.astype(int)

  

# Mapping AgeBand

age_map = {"Underage": 1, "Young": 2, "MidAge": 3,

             "Old": 4, "VeryOld": 5}

train_set['AgeScaled'] = train_set['AgeBand'].map(age_map)

train_set['AgeScaled'] = train_set['AgeScaled'].fillna(0)

train_set.AgeScaled = train_set.AgeScaled.astype(int)

test_set['AgeScaled'] = test_set['AgeBand'].map(age_map)

test_set['AgeScaled'] = test_set['AgeScaled'].fillna(0)

test_set.AgeScaled = test_set.AgeScaled.astype(int)
# Drop unused parameters (wip)

drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Sex', 'Embarked', 'Surname', 'Title', 'Ethnicity', 'AgeBand']

train_set = train_set.drop(drop, axis = 1)

test_set  = test_set.drop(drop, axis = 1)
train_set.head(3)
test_set.head(3)
# Make Train and Test Data

X_train = train_set.drop("Survived",axis=1)

Y_train = train_set["Survived"]

X_test  = test_set.copy()



# Quickcheck with a very small Validation set. 

# Very dirty, using once the first ten % and once the last

tenPercent = int((len(X_train)/100)*10)



#Using the last 10% as Validation set

X_train_90_first = X_train.head((len(X_train)-tenPercent))

X_val_10_first = X_train.tail(tenPercent)

Y_train_90_first = Y_train.head((len(X_train)-tenPercent))

Y_val_10_first = Y_train.tail(tenPercent)



#Using the first 10% as Validation set

X_train_90_last = X_train.tail((len(X_train)-tenPercent))

X_val_10_last = X_train.head(tenPercent)

Y_train_90_last = Y_train.tail((len(X_train)-tenPercent))

Y_val_10_last = Y_train.head(tenPercent)
# Logistic Regression



logreg = LogisticRegression()



print("_Logistic Regression_")

print()



logreg.fit(X_train_90_first, Y_train_90_first)

print("Train Score 'first': ", logreg.score(X_train_90_first, Y_train_90_first))

print("Validation Score 'first': ",logreg.score(X_val_10_first, Y_val_10_first))

print()

logreg.fit(X_train_90_last, Y_train_90_last)

print("Train Score 'last': ", logreg.score(X_train_90_last, Y_train_90_last))

print("Validation Score 'last': ",logreg.score(X_val_10_last, Y_val_10_last))



Score_val_LR = (logreg.score(X_val_10_first, Y_val_10_first) + logreg.score(X_val_10_last, Y_val_10_last))/2



# Train full training set

logreg.fit(X_train, Y_train)

Y_pred_LR = logreg.predict(X_test)
# Support Vector Machines



svc = SVC()



print("_Support Vector Machines_")

print()



svc.fit(X_train_90_first, Y_train_90_first)

print("Train Score 'first': ", svc.score(X_train_90_first, Y_train_90_first))

print("Validation Score 'first': ", svc.score(X_val_10_first, Y_val_10_first))

print()

svc.fit(X_train_90_last, Y_train_90_last)

print("Train Score 'last': ", svc.score(X_train_90_last, Y_train_90_last))

print("Validation Score 'last': ",svc.score(X_val_10_last, Y_val_10_last))



Score_val_SVC = (svc.score(X_val_10_first, Y_val_10_first) + svc.score(X_val_10_last, Y_val_10_last))/2



#Train full training set

svc.fit(X_train, Y_train)

Y_pred_SVC = svc.predict(X_test)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



print("_Random Forests_")

print()



random_forest.fit(X_train_90_first, Y_train_90_first)

print("Train Score 'first': ", random_forest.score(X_train_90_first, Y_train_90_first))

print("Validation Score 'first': ", random_forest.score(X_val_10_first, Y_val_10_first))

print()

random_forest.fit(X_train_90_last, Y_train_90_last)

print("Train Score 'last': ", random_forest.score(X_train_90_last, Y_train_90_last))

print("Validation Score 'last': ",random_forest.score(X_val_10_last, Y_val_10_last))



Score_val_RF = (random_forest.score(X_val_10_first, Y_val_10_first) + random_forest.score(X_val_10_last, Y_val_10_last))/2



#Train full training set

random_forest.fit(X_train, Y_train)

Y_pred_RF = random_forest.predict(X_test)
# K nearest Neighbors



knn = KNeighborsClassifier(n_neighbors = 3)



print("_K nearest Neighbors_")

print()



knn.fit(X_train_90_first, Y_train_90_first)

print("Train Score 'first': ", knn.score(X_train_90_first, Y_train_90_first))

print("Validation Score 'first': ", knn.score(X_val_10_first, Y_val_10_first))

print()

knn.fit(X_train_90_last, Y_train_90_last)

print("Train Score 'last': ", knn.score(X_train_90_last, Y_train_90_last))

print("Validation Score 'last': ",knn.score(X_val_10_last, Y_val_10_last))



Score_val_KNN = (knn.score(X_val_10_first, Y_val_10_first) + knn.score(X_val_10_last, Y_val_10_last))/2



#Train full training set

knn.fit(X_train, Y_train)

Y_pred_KNN = knn.predict(X_test)
# Gaussian Naive Bayes



gaussian = GaussianNB()



print("_Gaussian Naive Bayes_")

print()



gaussian.fit(X_train_90_first, Y_train_90_first)

print("Train Score 'first': ", gaussian.score(X_train_90_first, Y_train_90_first))

print("Validation Score 'first': ", gaussian.score(X_val_10_first, Y_val_10_first))

print()

gaussian.fit(X_train_90_last, Y_train_90_last)

print("Train Score 'last': ", gaussian.score(X_train_90_last, Y_train_90_last))

print("Validation Score 'last': ",gaussian.score(X_val_10_last, Y_val_10_last))



Score_val_Gauss = (gaussian.score(X_val_10_first, Y_val_10_first) + gaussian.score(X_val_10_last, Y_val_10_last))/2



#Train full training set

gaussian.fit(X_train, Y_train)

Y_pred_Gauss = gaussian.predict(X_test)
Scores = pd.DataFrame({'Log Reg': Score_val_LR,

                        'SVM': Score_val_SVC,

                        'Random F': Score_val_RF,

                        'KNN': Score_val_KNN,

                        'Gauss': Score_val_Gauss}, index=[0])



Scores
# Compare Y predicitons for the various classifiers



Results = pd.DataFrame({'Log Reg': Y_pred_LR,

                        'SVM': Y_pred_SVC,

                        'Random F': Y_pred_RF,

                        'KNN': Y_pred_KNN,

                        'Gauss': Y_pred_Gauss})

Results.head(10)
Results.loc[:,'Log Reg'] *= Score_val_LR

Results.loc[:,'SVM'] *= Score_val_SVC

Results.loc[:,'Random F'] *= Score_val_RF

Results.loc[:,'KNN'] *= Score_val_KNN

Results.loc[:,'Gauss'] *= Score_val_Gauss
Y_pred = Results['Log Reg'].add(Results["SVM"]).add(Results["Random F"]).add(Results["KNN"]).add(Results["Gauss"])

Y_pred = Y_pred/(Score_val_LR+Score_val_SVC+Score_val_RF+Score_val_KNN+Score_val_Gauss)

Y_pred = Y_pred.values
Y_pred[Y_pred >= 0.5] = int(1)

Y_pred[Y_pred < 0.5] = int(0)
# Generate Submission File 

submission = pd.DataFrame({

        "PassengerId": test_set_orig["PassengerId"],

        "Survived": Y_pred.astype(int)

    })

submission.to_csv('titanic.csv', index=False)



submission.head(10)