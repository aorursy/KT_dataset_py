import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  # as LR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head(10)
#New Characteristics of Single and Group Travelers.
train['Family_group'] = train.SibSp + train.Parch
test['Family_group'] = test.SibSp + test.Parch

#New family size characteristics.
train['Family'] = train.SibSp + train.Parch + 1
test['Family'] = test.SibSp + test.Parch + 1
#Family_group features converted to dummy variables
train['Family_group'] = np.where(train['Family_group'] >= 1, 1, 0)
test['Family_group'] = np.where(test['Family_group'] >= 1, 1, 0)

train.head(10)
# of single travelers, the probability of being able to get on the ship is set at 1 because it is a sure thing.

#Probability of being able to get on an escape boat for a group traveler with no honorific title.
escape_boarding_probability_average_train = 2 / ((sum(train['Family']) - sum(train['Family_group'] == 0))/sum(train['Family_group'] == 1))
escape_boarding_probability_average_test = 2 / ((sum(test['Family']) - sum(test['Family_group'] == 0))/sum(test['Family_group'] == 1))

escape_boarding_probability_average_train
train['escape_boarding_probability_train'] = 1
test['escape_boarding_probability_test'] = 1

train['escape_boarding_probability_train'] = train['escape_boarding_probability_train'].replace(1, np.nan)
test['escape_boarding_probability_test'] = test['escape_boarding_probability_test'].replace(1, np.nan)

for i in range(1, 891):
    if (train['Family_group'][i] == 0):
        train['escape_boarding_probability_train'][i] = 1

for i in range(1, 418):
    if (test['Family_group'][i] == 0):
        test['escape_boarding_probability_test'][i] = 1        
train.head(100)
train_mr_index = train['Name'].str.contains(' Mr. ')
train_miss_index = train['Name'].str.contains(' Miss. ')
train_mrs_index = train['Name'].str.contains(' Mrs. ')
train_master_index = train['Name'].str.contains(' Master. ')
test_mr_index = test['Name'].str.contains(' Mr. ')
test_miss_index = test['Name'].str.contains(' Miss. ')
test_mrs_index = test['Name'].str.contains(' Mrs. ')
test_master_index = test['Name'].str.contains(' Master. ')

train['escape_boarding_probability_train'][train_mr_index] = 1 / train['Family'][train_mr_index] 
train['escape_boarding_probability_train'][train_miss_index] = train['Family'][train_miss_index] - 1 / train['Family'][train_miss_index]
train['escape_boarding_probability_train'][train_mrs_index] = 1 / train['Family'][train_mrs_index]
train['escape_boarding_probability_train'][train_master_index] = train['Family'][train_master_index] - 1 / train['Family'][train_master_index]   
train['escape_boarding_probability_train']=train['escape_boarding_probability_train'].fillna(escape_boarding_probability_average_train)

test['escape_boarding_probability_test'][test_mr_index] = 1 / test['Family'][test_mr_index] 
test['escape_boarding_probability_test'][test_miss_index] = test['Family'][test_miss_index] - 1 / test['Family'][test_miss_index]
test['escape_boarding_probability_test'][test_mrs_index] = 1 / test['Family'][test_mrs_index]
test['escape_boarding_probability_test'][test_master_index] = test['Family'][test_master_index] - 1 / test['Family'][test_master_index]    
test['escape_boarding_probability_test']=test['escape_boarding_probability_test'].fillna(escape_boarding_probability_average_test)
train_mr = train[train['Name'].str.contains(' Mr. ')]
train_miss = train[train['Name'].str.contains(' Miss. ')]
train_mrs = train[train['Name'].str.contains(' Mrs. ')]
train_master = train[train['Name'].str.contains(' Master. ')]
test_mr = test[test['Name'].str.contains(' Mr. ')]
test_miss = test[test['Name'].str.contains(' Miss. ')]
test_mrs = test[test['Name'].str.contains(' Mrs. ')]
test_master = test[test['Name'].str.contains(' Master. ')]

train_mr_num = train_mr['Age'].dropna().mean()
train_miss_num = train_miss['Age'].dropna().mean()
train_mrs_num = train_mrs['Age'].dropna().mean()
train_master_num = train_master['Age'].dropna().mean()
train_all_num = train['Age'].dropna().median()

test_mr_num = test_mr['Age'].dropna().mean()
test_miss_num = test_miss['Age'].dropna().mean()
test_mrs_num = test_mrs['Age'].dropna().mean()
test_master_num = test_master['Age'].dropna().mean()
test_all_num = test['Age'].dropna().median()
train['Age'][train_mr_index] = train_mr['Age'].fillna(32)
train['Age'][train_miss_index] = train_master['Age'].fillna(22)
train['Age'][train_mrs_index] = train_mrs['Age'].fillna(36)
train['Age'][train_master_index] = train_master['Age'].fillna(5)
train['Age'] = train['Age'].fillna(28)

test['Age'][test_mr_index] = test_mr['Age'].fillna(32)
test['Age'][test_miss_index] = test_miss['Age'].fillna(22)
test['Age'][test_mrs_index] = test_mrs['Age'].fillna(39)
test['Age'][test_master_index] = test_master['Age'].fillna(7)
test['Age'] = test['Age'].fillna(27)

train.isnull().sum()
# Missing value completion
train['Embarked'] = train['Embarked'].fillna('S')
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
# Sex and Embarked conversion to dummy variables
dummy_train = pd.get_dummies(train[['Sex', 'Embarked']])
dummy_test = pd.get_dummies(test[['Sex', 'Embarked']])

train_two = pd.concat([train.drop(["Sex", "Embarked"], axis = 1),dummy_train], axis = 1)
test_two = pd.concat([test.drop(["Sex", "Embarked"], axis = 1),dummy_test], axis = 1)

train_two.isnull().sum()
# Removing Unnecessary Features
train_three = train_two.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Parch', 'SibSp'], axis = 1)
x_test = test_two.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Parch', 'SibSp'], axis = 1)

train_three.isnull().sum()
#conversion to data frame type.
x_train_df = train_three.drop(['Survived'], axis = 1)
x_train = x_train_df

#Storing objective variables
y_train = train_three.Survived

#Study the decision tree.
depth = 4
clf = tree.DecisionTreeClassifier(max_depth = depth)
clf.fit(x_train_df, y_train)
#apply class to return the leaf number for each leaf.
x_train_leaf_no = clf.apply(x_train_df)
x_test_leaf_no = clf.apply(x_test)


#Logistic regression analysis for each leaf.

# Prepare an array with all indexes set to 0.
x_train_proba = np.zeros(x_train.shape[0])
x_test_proba = np.zeros(x_test.shape[0])

# Store non-duplicate leaf numbers in a list.
unique_leaf_no = list(set(x_train_leaf_no))

#tuning the hyperparameters of logistic regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Retrieving stored leaf numbers.
for i in unique_leaf_no :
    #Confirm the leaf number to take out.
    print('leaf no:', i)
    
    #Stores the data frame retrieved by specifying the leaf number of the train data into a variable.
    leaf_data_train_x = x_train[x_train_leaf_no == i]
    leaf_data_train_y = y_train[x_train_leaf_no == i]
    #test data leaf number and store the retrieved data frame in a variable
    leaf_data_test_x = x_test[x_test_leaf_no == i]


    # once, exclude the data in the dummy variable.
    leaf_data_train_x_drop = leaf_data_train_x.drop(['Family_group', 'Pclass', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_S', 'Embarked_Q', 'escape_boarding_probability_train'], axis = 1)
    leaf_data_test_x = leaf_data_test_x.drop(['Family_group', 'Pclass', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_S', 'Embarked_Q', 'escape_boarding_probability_test'], axis = 1)
    
    #survived if there are both survivors and deaths in the value of
    if len(set(leaf_data_train_y)) > 1:

        #GridSearch is performed
        try:
            grid_search = GridSearchCV(LogisticRegression(), param_grid, cv = 5, scoring = 'roc_auc')   
            grid_search.fit(leaf_data_train_x_drop, leaf_data_train_y)
            clf = LogisticRegression(C=grid_search.best_params_['C'],class_weight="balanced")
        except (ValueError, TypeError, NameError, SyntaxError):
            clf = LogisticRegression()

        #Logistic Regression Analysis.
        clf.fit(leaf_data_train_x_drop, leaf_data_train_y)

        #Return the probability of the class you belong to.
        a = clf.predict_proba(leaf_data_train_x_drop)

        # Store only the probability of survival
        x_train_proba[x_train_leaf_no == i] = a[::,1]

        if len(leaf_data_test_x) > 0:
            b = clf.predict_proba(leaf_data_test_x)    
            x_test_proba[x_test_leaf_no == i] = b[::,1]


    #survived if the value of survived is either a survivor or a dead person.    
    else:
        x_train_proba[x_train_leaf_no == i] = leaf_data_train_y.head(1)
        if len(leaf_data_test_x) > 0:
            x_test_proba[x_test_leaf_no == i] =leaf_data_train_y.head(1)



#Confirming the end of the loop.
print("for loop end")

#combined probability of survival and death data frames
train_data = pd.concat([x_train, pd.DataFrame(x_train_proba)], axis =1)
test_data = pd.concat([x_test, pd.DataFrame(x_test_proba)], axis =1)

#tuning the hyperparameters of logistic regression
param_grid = {'max_depth': [3,5,8,13,21,34]}

#GridSearch.
grid_search = GridSearchCV(GradientBoostingClassifier(n_estimators=100), param_grid, cv = 5, scoring = 'roc_auc')   
grid_search.fit(train_data, y_train)

#Gradient Boosting for Learning and Prediction
model = GradientBoostingClassifier(max_depth=grid_search.best_params_['max_depth'], n_estimators=100)
model.fit(train_data, y_train)
output = model.predict(test_data).astype(int)


# Convert results to CSV
leaf_data_test = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": output
})
leaf_data_test

leaf_data_test.to_csv('submit.csv', index = False)
