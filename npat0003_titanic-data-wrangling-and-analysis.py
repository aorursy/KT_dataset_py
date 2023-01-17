import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
titanic1 = pd.read_csv('../input/train.csv')

titanic2 = pd.read_csv('../input/test.csv')
titanic = pd.merge(titanic1, titanic2, how='outer')


titanic.head()
titanic.info()
# The name column can be splitted into more meaningful columns for better analysis 

titanic.Name.unique()
# Lets seperate the titles from the name 

coltitle = titanic['Name'].apply(lambda s: pd.Series({'Title': s.split(',')[1].split('.')[0].strip(),

                                                   'LastName':s.split(',')[0].strip(), 'FirstName':s.split(',')[1].split('.')[1].strip()}))

print (coltitle)
# Add the columns to the titanic dataframe

titanic = pd.concat([titanic, coltitle], axis=1) 

# Drop the Name column

titanic.drop('Name', axis=1, inplace=True)

print (titanic.head())
# Lets check the number of male and female

titanic.Sex.value_counts()
# Lets set a style for all the plots

print (style.available)

style.use('classic')
# Lets plot the number of male and females on the ship

titanic.Sex.value_counts().plot(kind='bar')

plt.show()
# Lets check the number of casualties on the ship

titanic.Survived.value_counts()
# Lets plot the casualties

titanic.Survived.value_counts().plot(kind='bar', title='Number of people survived [0 - Not Surv, 1 - Surv]\n')

plt.show()
# We can use the title column to get an inside

titanic.Title.unique()
# Also reassign mlle, ms, and mme accordingly

titanic.loc[titanic['Title']=='Mlle', 'Title']='Miss'.strip()

titanic.loc[titanic['Title']=='Ms', 'Title']='Miss'.strip()

titanic.loc[titanic['Title']=='Mme', 'Title']='Mrs'.strip()
# Get the count of female and male passengers based on titles

tab = titanic.groupby(['Sex', 'Title']).size()

print (tab)
# Now lets get the count of unique surnames 

print (titanic.LastName.unique().shape[0])
titanic['total_members'] = titanic.SibSp + titanic.Parch + 1

survivor = titanic[['Survived', 'total_members']].groupby('total_members').mean()

survivor.plot(kind='bar')

plt.show()
titanic.isnull().sum()
# Drop the Ticket and Cabin column 

titanic.drop('Cabin', axis=1, inplace=True)

titanic.drop('Ticket', axis=1, inplace=True)
# There is one missing value in Fare

titanic[titanic.Fare.isnull()==True]
titanic[['Pclass', 'Fare']].groupby('Pclass').mean()
titanic.loc[titanic.PassengerId==1044.0, 'Fare']=13.30
# Check the null values in Embarked column

titanic.Embarked.isnull().sum()
titanic[titanic['Embarked'].isnull() == True]
# Lets try to find the embark based on survived

titanic[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean()
# Also lets try to find the fare based on Embarked 

titanic[['Embarked', 'Fare']].groupby('Embarked').mean()
# Imputting the missing value

titanic.loc[titanic['Embarked'].isnull() == True, 'Embarked']='C'.strip()
titanic.Age.isnull().sum()
titanic.Age.plot(kind='hist')

plt.show()
pd.pivot_table(titanic, index=['Sex', 'Title', 'Pclass'], values=['Age'], aggfunc='median')
# a function that fills the missing values of the Age variable

    

def fillAges(row):

    

    if row['Sex']=='female' and row['Pclass'] == 1:

        if row['Title'] == 'Miss':

            return 29.5

        elif row['Title'] == 'Mrs':

            return 38.0

        elif row['Title'] == 'Dr':

            return 49.0

        elif row['Title'] == 'Lady':

            return 48.0

        elif row['Title'] == 'the Countess':

            return 33.0



    elif row['Sex']=='female' and row['Pclass'] == 2:

        if row['Title'] == 'Miss':

            return 24.0

        elif row['Title'] == 'Mrs':

            return 32.0



    elif row['Sex']=='female' and row['Pclass'] == 3:

        

        if row['Title'] == 'Miss':

            return 9.0

        elif row['Title'] == 'Mrs':

            return 29.0



    elif row['Sex']=='male' and row['Pclass'] == 1:

        if row['Title'] == 'Master':

            return 4.0

        elif row['Title'] == 'Mr':

            return 36.0

        elif row['Title'] == 'Sir':

            return 49.0

        elif row['Title'] == 'Capt':

            return 70.0

        elif row['Title'] == 'Col':

            return 58.0

        elif row['Title'] == 'Don':

            return 40.0

        elif row['Title'] == 'Dr':

            return 38.0

        elif row['Title'] == 'Major':

            return 48.5



    elif row['Sex']=='male' and row['Pclass'] == 2:

        if row['Title'] == 'Master':

            return 1.0

        elif row['Title'] == 'Mr':

            return 30.0

        elif row['Title'] == 'Dr':

            return 38.5



    elif row['Sex']=='male' and row['Pclass'] == 3:

        if row['Title'] == 'Master':

            return 4.0

        elif row['Title'] == 'Mr':

            return 22.0





titanic['Age'] = titanic.apply(lambda s: fillAges(s) if np.isnan(s['Age']) else s['Age'], axis=1)

titanic.Age.plot(kind='hist')

plt.show()
titanic.info()
# Convert sex to 0 and 1 (Female and Male)

def trans_sex(x):

    if x == 'female':

        return 0

    else:

        return 1

titanic['Sex'] = titanic['Sex'].apply(trans_sex)



# Convert Embarked to 1, 2, 3 (S, C, Q)

def trans_embark(x):

    if x == 'S':

        return 3

    if x == 'C':

        return 2

    if x == 'Q':

        return 1

titanic['Embarked'] = titanic['Embarked'].apply(trans_embark)    

    
# Add a child and mother column for predicting survivals

titanic['Child'] = 0

titanic.loc[titanic['Age']<18.0, 'Child'] = 1

titanic['Mother'] = 0

titanic.loc[(titanic['Age']>18.0) & (titanic['Parch'] > 0.0) & (titanic['Sex']==0) & (titanic['Title']!='Miss'), 'Mother'] =1
titanic.isnull().sum()
# Feature selection for doing the predictions

features_label = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'total_members', 'Child', 'Mother']

target_label= ['Survived']

train = titanic[titanic['Survived'].isnull()!= True]

test = titanic[titanic['Survived'].isnull()== True]

print (train.shape)

print (test.shape)
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X=train[features_label], y=train[target_label])



Y_pred = random_forest.predict(X=test[features_label])



random_forest.score(X=train[features_label], y=train[target_label])
# Logistic Regression

regr = LogisticRegression()

regr.fit(X=train[features_label], y=train[target_label])

regr.score(X=train[features_label], y=train[target_label])
# Predicted Values for Survived

predict_t = regr.predict(X=test[features_label])

print (predict_t)
# Insert the predicted values for the missing rows for Survived column

titanic.loc[titanic['Survived'].isnull()== True, 'Survived']= predict_t
# fit an Extra Trees model to the data

model = ExtraTreesClassifier()

model.fit(X=train[features_label], y=train[target_label])

# display the relative importance of each attribute

importance = model.feature_importances_

print (importance)
# model is of type array, convert to type dataframe



imp = pd.DataFrame({'feature':features_label,'importance':np.round(model.feature_importances_,3)})

imp = imp.sort_values('importance',ascending=False).set_index('feature')

print (imp)

imp.plot.bar()

plt.show()
print ("\nThe number of passengers based on Sex\n")

print (titanic['Sex'].value_counts()) 



print ("\nThe number of survivors based on Sex\n")

print(titanic[['Survived', 'Sex']].groupby('Sex').sum()) 



print ("\nThe number of passengers based on Pclass\n")

print (titanic['Pclass'].value_counts())

       

print("\nThe number of survivors based on Pclass\n")

print(titanic[['Survived', 'Pclass']].groupby('Pclass').sum()) 



print ("\nThe number of passengers who are Mother\n")

print (titanic['Mother'].value_counts())

       

print ("\nThe number of survivors based on Mother\n")

print (titanic[['Survived', 'Mother']].groupby('Mother').sum())
# Convert sex to 0 and 1 (Female and Male)

def trans_sex(x):

    if x == 0:

        return 'female'

    else:

        return 'male'

titanic['Sex'] = titanic['Sex'].apply(trans_sex)



# Convert Embarked to 1, 2, 3 (S, C, Q)

def trans_embark(x):

    if x == 3:

        return 'S'

    if x == 2:

        return 'C'

    if x == 1:

        return 'Q'

titanic['Embarked'] = titanic['Embarked'].apply(trans_embark) 
titanic.to_csv('titanic.csv')