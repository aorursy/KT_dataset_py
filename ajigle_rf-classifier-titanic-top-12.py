#Titanic ML Kaggle Competition
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



from collections import Counter
#Upload data

test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")



#Used for final results

Pass_ID = test_df['PassengerId']
#Check out if any outliers using Quartile Ranges

def check_outliers(df, n, features):

    outliers_lst = []

    

    for feat in features:

        #Q1 25% 

        Q1 = np.percentile(df[feat], 25)

        #Q3 75%

        Q3 = np.percentile(df[feat], 75)

        #Inner Quartile Range

        IQR = Q3 - Q1

        

        #Determining outlier

        outlier = 1.5 * IQR

        

        #Determine which cols have outliers

        outlier_lst_col = df[(df[feat] < Q1 - outlier) | (df[feat] > Q3 + outlier)].index

        

        #Append the columns

        outliers_lst.extend(outlier_lst_col)

        

    #Drop outliers in more than n present in data

    outliers_tot = Counter(outliers_lst)

    outliers = list(j for j, o in outliers_tot.items() if o > n)

    

    return outliers

        
confirm_outliers = check_outliers(train_df, 2, ['Age', 'Parch', 'SibSp', 'Fare'])

confirm_outliers

#Locate the columns to drop

train_df.loc[confirm_outliers]
#Drop the confirmed outliers

train_df = train_df.drop(confirm_outliers, axis = 0).reset_index(drop=True)

train_df
#Get length for splitting once we are ready for modeling

train_len = len(train_df)



data = pd.concat(objs = [train_df, test_df], axis = 0, sort = False).reset_index(drop=True)

data.shape
#Create new feature Family Size

data['Family_Size'] = data['SibSp'] + data['Parch'] + 1

data.head()
#Decipher between family sizes on board

data['Single'] = data['Family_Size'].map(lambda f: 1 if f == 1 else 0)



#Small Families

data['Small_Fam'] = data['Family_Size'].map(lambda f: 1 if f == 2 else 0)



#Medium Families

data['Medium_Fam'] = data['Family_Size'].map(lambda f: 1 if 3 <= f <= 4 else 0)



#Large Families

data['Large_Fam'] = data['Family_Size'].map(lambda f: 1 if f >= 5 else 0)



data.head()
data['Cabin'].value_counts()
#Extract the first letter of the cabin ID in order to get what cabin passenger's are in

data['Cabin'] = data['Cabin'].astype(str)

data['Cabin'] = data['Cabin'].str.replace(r'[0-9]', '', regex = True)

data.head()
data['Cabin'].value_counts()
#Replace remaining Cabin values with their Prefix Letter

data['Cabin'] = data['Cabin'].replace({'B B': 'B', 'C C': 'C', 'B B B B': 'B', 'F G': 'F', 'B B B': 'B', 'F E': 'F', 

                                  'C C C': 'C', 'D D': 'D', 'E E': 'E'}, regex = True)

data['Cabin'] = data['Cabin'].replace({'B B': 'B', 'C C': 'C'}, regex = True)

data['Cabin'].value_counts()
#Now we need to take care of the NaN's I will create a new cabin class for them named 'Z'

data['Cabin'] = data['Cabin'].replace('nan', 'Z')

data['Cabin'].value_counts()
data = pd.get_dummies(data, columns = ['Cabin'], prefix = 'Cabin')
#Let's make a new column based on the titles of the people on board

data['Title'] = [title.split(',')[1].split('.')[0] for title in data['Name']]

data.head()
#Let's visualize the types of titles that are included in the dataset

titles = data['Title'].value_counts()

titles

#We can use count plot to visualize this and clearly see that several titles are rarely seen

fig, ax = plt.subplots(figsize = (15,5))

title_plot = sns.countplot(x = 'Title', data = data, ax = ax)
#Categorical to Numerical

data['Title'] = data['Title'].replace([' Major', ' the Countess', ' Sir', ' Lady', ' Don', ' Rev',

                                       ' Dr', ' Col', ' Capt', ' Jonkheer', 

                                       ' Dona', ' Countess'], 'Uncommon')

#Bug in code here

mapping = {' Master': 0, ' Miss': 1, ' Mrs': 1, ' Ms': 1, ' Mlle': 1, ' Mme': 1, 'Uncommon': 2, 'Uncommona': 2, ' Mr': 3}



data = data.applymap(lambda x: mapping.get(x) if x in mapping else x)

data['Title'] = data['Title'].astype(int)

data['Title'].value_counts()
#Convert them to numerical dummies

data = pd.get_dummies(data, columns = ['Title'], prefix = 'Title')
#Let's compare people with siblings or spouses and see the distribution.

#Clearly a lot more people are on board with either none or 1 spouse or sibling with them than multiple 

fig, ax = plt.subplots(figsize = (5,5))

fam_graph = sns.countplot(x = 'SibSp', data = data, ax = ax)
#And now parents and children on board as well

fig, ax = plt.subplots(figsize = (5,5))

par_child_graph = sns.countplot(x = 'Parch', data = data, ax = ax)
data.isnull().sum()
#Try median after model is built to see which obtains higher accuracy

data['Age'] = data['Age'].fillna(data['Age'].mean())



#Confirm the code above will indeed compute and replace ages with average age on board

print(data['Age'].mean())



#Confirm nan values for age are replaced

data.isnull().sum()
#Finally for embarked which designates which port they embarked from it can only three options

# S = Southampton, Q = Queenstown, C = Cherbourg

#Let's visualize that

port_graph = sns.countplot(x = 'Embarked', data = data)
#For purposes to confirm the replacing went to right value

data['Embarked'].value_counts()
#Replacing the 2 null values with 'Q'

data['Embarked'] = data['Embarked'].fillna('Q')
#We should see that the Queenstown port ('Q') has two additional members

data['Embarked'].value_counts()
#Convert to numerical 

data = pd.get_dummies(data, columns = ['Embarked'], prefix = 'Emb')
#First let's see the descrempancy of males and females on board

gender_plot = sns.countplot(x = 'Sex', data = data)
#Lets make males = 0 and females = 1

data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
#Drop unnecessary columns

data = data.drop(columns = ['Name', 'PassengerId'])

data.head()
data = pd.get_dummies(data, columns = ['Pclass'], prefix = 'Pcl')

data.head()
#Settle the fare Nan value

data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

print(data['Fare'].mean())
data['Ticket'].head()
#Take ticket prefix which makes give way to the cabin or fare paid for specific locations on the ship



tickets = []

for t in data['Ticket']:

    if not t.isdigit():

        #Obtaining prefix

        tickets.append(t.replace('/','').replace('.','').strip().split(' ')[0])

    else:

        #Appending a Z because I used Z for cabins that were not known

        tickets.append('Z')



data['Ticket'] = pd.Series(tickets)

data.head()
data = pd.get_dummies(data, columns = ['Ticket'], prefix = 'Tic_')

data.head()
#Split train and test from data

train = data[:train_len]

test = data[train_len:]



train.head()
#Make sure we have no null values for prediction

train.isnull().sum()
train_X = train.drop('Survived', axis = 1)

target = train['Survived']



test = test.drop('Survived', axis = 1)

train_X.head()
#Model with grid search

rf = RandomForestClassifier()

rf_params = {'n_estimators' : [100, 200, 300, 400],

            'criterion' : ['gini', 'entropy'],

            'max_depth' : [3,4,5,6]}



rf_gs = GridSearchCV(rf, param_grid = rf_params, scoring = 'accuracy')



rf_gs.fit(train_X, target)



rf_gs.best_score_
#Now predict using our test set and save the results

#Use the best estimator from above for prediction

rf_best = rf_gs.best_estimator_



survived_pred = rf_best.predict(test).astype(int)



results = pd.DataFrame(Pass_ID)



results['Survived'] = survived_pred



results.columns = ['PassengerId', 'Survived']



results.to_csv('Titanic_Survival_Results.csv', index = False)



results