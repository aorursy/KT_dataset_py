import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



# Example of meassuring time

# I always like to have an example of how to measure time at the top

import time

# start = time.time()

# ... code ...

# end = time.time()

# print('Execution Time: ' + str(round(end - start, 2)) + 's')



# Load the titanic dataset. Each row represents 1 passanger.

titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
# Dataframe Information:

# - attribute data types

# - attribute non-null values (non-missing values)

# - number of rows/indexes

# - memory usage (ram)

titanic_df.info()
# Missing values from each column

titanic_df.isnull().sum()
# Summary stats

# - PassengerId (is an id so most of the stats are useless)

# - Survived (is the class, which is either 0 or 1, so most of the stats are also useless)

# - Pclass (is the ticket class which is either 1,2 or 3. Again most stats are not applicable)

# - Age (is the age of the passanger)

# - SibSp (is the number of siblings/Spouses aboard)

# - Parch (is the number of Parents/Children aboard)

# - Fare (amount paid for the fare)

titanic_df.describe()
# sns visualization library build on top of matplotlib

import seaborn as sns

# visualization library

import matplotlib.pyplot as plt

%matplotlib inline
dead_color = '#0b4b51'

alive_color = '#22d5e5'

survival_palette=[dead_color, alive_color]



# Plot how many passanger who embarked at each of the docks, have survived and how many died

# Suprisingly most of th epassanger who embarked at C, survived.

chart_holder_embarked = sns.countplot(x='Embarked', hue='Survived', data=titanic_df, palette=survival_palette)
# Plot the ticket class the survival of passangers

# We can see that passangers with a higher class ticket had more chance to survive

chart_holder_ticket_class = sns.countplot(x='Pclass', hue='Survived', data=titanic_df, palette=survival_palette)
rich_color = '#efda5f'

middle_color = '#d1c168'

poor_color = '#a8a17c'

class_palette=[rich_color, middle_color, poor_color]



# Because in the 1st plot we can see that more people actually survived at embarked C = Cherbourg

# And because we see that a higher class ticket the passanger had more chance to survive.

# Here I plot the distribution of ticket classes accross the embarking points.

# We can see that C = Cherbourg, was the only embarking point with a majority of passanger with the 1st class ticket.

chart_holder_embarked_class = sns.countplot(x="Embarked", hue="Pclass", data=titanic_df[['Embarked', 'Pclass']], palette=class_palette)
age_med = titanic_df.Age.median() # set median value



# fill NAN data

titanic_df.Age.fillna(age_med, inplace=True)

titanic_df.Embarked.fillna(method='ffill', inplace=True) # Replaces NaN with previous value

titanic_df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
# Extracting title from name



mapped_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}



titanic_df['Title'] = titanic_df.Name.map(lambda x: mapped_titles[x.split(',')[1].split('.')[0].strip()])

titanic_df.drop(['Name'], axis=1, inplace=True)
# Creating Family Size



titanic_df['Family_size'] = titanic_df['SibSp'] + titanic_df['Parch']

titanic_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
# Create Dummy variables for Cabin



titanic_df.Cabin.fillna('U', inplace=True)

titanic_df.Cabin = titanic_df.Cabin.map(lambda x: x[0])



cabin_dummies = pd.get_dummies(titanic_df.Cabin, prefix="Cabin")

titanic_df.drop(['Cabin'], axis=1, inplace=True)



titanic_df = pd.concat([titanic_df, cabin_dummies], axis=1)
# Create Dummy variables for Pclass, Embarked, Title



pclass_dummies = pd.get_dummies(titanic_df.Pclass, prefix="Ticket_class")

titanic_df.drop(['Pclass'], axis=1, inplace=True)



embarked_dummies = pd.get_dummies(titanic_df.Embarked, prefix="Embarked")

titanic_df.drop(['Embarked'], axis=1, inplace=True)



title_dummies = pd.get_dummies(titanic_df.Title, prefix="Title")

titanic_df.drop(['Title'], axis=1, inplace=True)



titanic_df = pd.concat([titanic_df, pclass_dummies, embarked_dummies, title_dummies], axis=1)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



##

# Scales and Normalizes all the attributes except the class attribute (Survived)

##

def preprocess_attributes(dataframe):

    for col in dataframe.columns:

        if col == 'Survived':

            continue;

        dataframe[col] = le.fit_transform(dataframe[col]) # Scales and normalizes

    

    return dataframe

    

titanic_df = preprocess_attributes(titanic_df)
# A quick look at how the data looks before the modelling

titanic_df.head()
from sklearn.model_selection import train_test_split



X = titanic_df.drop(['Survived'], 1).values

Y = titanic_df['Survived'].values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier 



# Creating a grid of all hyper parameter combinations



parameters = {

    'n_estimators': [500,1000,1500],

    'max_depth': [10,16,20],

    'min_samples_split': [4,8,12],

    'min_samples_leaf': [3,4,5],

}



combinations = np.array([len(parameters[x]) for x in parameters])

combinations = np.multiply.reduce(combinations)

print(f'Number of all grid combinations: {combinations}')
start = time.time()



rf = RandomForestClassifier(random_state=0)



# Using a grid search with a 5-fold cross validation to find the best model

rf_clf = GridSearchCV(rf, parameters, scoring='accuracy', cv=5)

rf_clf.fit(X_train, Y_train)



print('Random Forrest')

print(rf_clf.best_params_)

print(f'Accuracy: {round(rf_clf.best_score_*100, 2)}%')

#svc_filled_clf.best_index_



end = time.time()

print('Execution Time: ' + str(round(end - start, 2)) + 's')
rf_clf_final = rf_clf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report





cm = confusion_matrix(Y_test, rf_clf_final)

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax)



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Survived', 'Died']); ax.yaxis.set_ticklabels(['Survived', 'Died']);
print(accuracy_score(Y_test, rf_clf_final))
print(classification_report(Y_test, rf_clf_final))
titanic_test_df = pd.read_csv('../input/test.csv')

titanic_test_df.isnull().sum()
age_med = titanic_test_df.Age.median() # set median value



# fill NAN data

titanic_test_df.Age.fillna(age_med, inplace=True)

titanic_test_df.Fare.fillna(method='ffill', inplace=True) # Replaces NaN with previous value

ids = titanic_test_df['PassengerId']

titanic_test_df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)



titanic_test_df['Title'] = titanic_test_df.Name.map(lambda x: mapped_titles[x.split(',')[1].split('.')[0].strip()])

titanic_test_df.drop(['Name'], axis=1, inplace=True)



titanic_test_df['Family_size'] = titanic_test_df['SibSp'] + titanic_test_df['Parch']

titanic_test_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)



titanic_test_df.Cabin.fillna('U', inplace=True)

titanic_test_df.Cabin = titanic_test_df.Cabin.map(lambda x: x[0])



cabin_dummies = pd.get_dummies(titanic_test_df.Cabin, prefix="Cabin")

titanic_test_df.drop(['Cabin'], axis=1, inplace=True)



titanic_test_df = pd.concat([titanic_test_df, cabin_dummies], axis=1)



pclass_dummies = pd.get_dummies(titanic_test_df.Pclass, prefix="Ticket_class")

titanic_test_df.drop(['Pclass'], axis=1, inplace=True)



embarked_dummies = pd.get_dummies(titanic_test_df.Embarked, prefix="Embarked")

titanic_test_df.drop(['Embarked'], axis=1, inplace=True)



title_dummies = pd.get_dummies(titanic_test_df.Title, prefix="Title")

titanic_test_df.drop(['Title'], axis=1, inplace=True)



titanic_test_df = pd.concat([titanic_test_df, pclass_dummies, embarked_dummies, title_dummies], axis=1)



titanic_test_df = preprocess_attributes(titanic_test_df)
titanic_df.head()
titanic_test_df['Cabin_T'] = 0

titanic_test_df = titanic_test_df[['Sex','Age','Fare','Family_size','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_T','Cabin_U','Ticket_class_1','Ticket_class_2','Ticket_class_3','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Officer','Title_Royalty']]

titanic_test_df.head()
predictions = rf_clf.predict(titanic_test_df.values)
#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)