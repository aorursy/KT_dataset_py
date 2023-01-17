# import libraries



import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score 
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



# Plot data that shows difference between survuved male and female

# we see that we can use this feature for our model

sex_pivot = train_data.pivot_table(index='Sex', values='Survived')

sex_pivot.plot.bar()

plt.show()

# Plot by class

pclass_pivot = train_data.pivot_table(index='Pclass', values='Survived')

pclass_pivot.plot.bar()

plt.show()
# Show survivors by age

survived = train_data[train_data['Survived'] == 1]

died = train_data[train_data['Survived'] == 0]

survived['Age'].plot.hist(alpha=0.5, color='red', bins=50)

died['Age'].plot.hist(alpha=0.5, color='blue', bins=50)

plt.legend(['Survived', 'Died'])

plt.show()
# Create dummy data

def create_dummies(df, column_name):

    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    df = pd.concat([df, dummies], axis=1)

    return df 

train = create_dummies(train_data, 'Pclass')

test = create_dummies(test_data, 'Pclass')

train.head(10)
# Group all ages so later we can use it as features in model

def process_age(df, cut_points, label_names):

    df['Age'] = df['Age'].fillna(-0.5)

    df['Age_categories'] = pd.cut(df['Age'], cut_points, labels=label_names)

    return df 



cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]

label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']



train_buff = process_age(train, cut_points, label_names)



age_cat_pivot = train_buff.pivot_table(index="Age_categories",values="Survived")

age_cat_pivot.plot.bar()

plt.show()
# Apply new fields into data and split labels with data 

numeric_fields = ['PassengerId','Pclass','Age', 'SibSp', 'Parch','Fare']

x = train_data[numeric_fields]

y = train_data['Survived']
#Remove nan from data

x['Age'] = x['Age'].fillna(x['Age']).median()

x['Age'].isnull().sum()
# Split training and testing

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2, random_state=0)
random_forest = RandomForestClassifier()

parameters = {

    "n_estimators": [4,5,6,15],

    "criterion": ["gini", "entropy"],

    "max_features": ["auto", "sqrt", "log2"], 

    "max_depth": [2, 3, 5, 10], 

    "min_samples_split": [2, 3, 5, 10],

    "min_samples_leaf": [1, 5, 8, 10]

}



grid_cv = GridSearchCV(random_forest, parameters, scoring = make_scorer(accuracy_score))

grid_cv = grid_cv.fit(trainX, trainY)



print("Our optimized Random Forest model is:")

grid_cv.best_estimator_
# Fit model

clf = RandomForestClassifier(

    n_estimators= 5,

    criterion= "gini",

    max_features= "sqrt", 

    max_depth= 3, 

    min_samples_split= 3,

    min_samples_leaf= 5)

clf.fit(trainX, trainY)
# Predictions on testing data

predictions = clf.predict(testX[numeric_fields])

accuracy = accuracy_score(testY, predictions)
accuracy