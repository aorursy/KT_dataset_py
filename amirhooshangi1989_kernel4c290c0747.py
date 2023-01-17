import pandas as pd

from collections import Counter

import numpy as np

## Exploratory phase of our project. 

# Lets just load the data

training_dataframe = pd.read_csv('/kaggle/input/titanic/train.csv')

testing_dataframe = pd.read_csv('/kaggle/input/titanic/test.csv')

training_dataframe

# For checking which columns have missing values i used the following code

#training_dataframe.isna().any()

# testing_dataframe.isna().any()

#Just found that Age, Cabin and Embarked columns have missing values in training set and Fare has also missing value on tet set


#By checking all values in the Name column, we found that it has the following pattern: "Surname, Title. Name"

# so we transform our long text to categorical values, like: Mr,Ms,Miss etc.



def get_title_values(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



# all the different titles

titles = sorted(set([x for x in training_dataframe.Name.map(lambda x: get_title_values(x))]))



# Normalize the titles, returning 'Mr', 'Master', 'Miss' or 'Mrs'

def change_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



# Lets create a new column for the titles

training_dataframe['Title'] = training_dataframe['Name'].map(lambda x: get_title_values(x))

testing_dataframe['Title'] = testing_dataframe['Name'].map(lambda x: get_title_values(x))



# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'

training_dataframe['Title'] = training_dataframe.apply(change_titles, axis=1)

testing_dataframe['Title'] = testing_dataframe.apply(change_titles, axis=1)



training_dataframe
#Feature Engineering based on sibling, spouse and parent/chile relatives. in this manner we create a new feature in which

#if passenger as any relatives is not alone in the ship otherwise he/she is alone. considering that being alone

#has higher impact on dying!



data = [training_dataframe, testing_dataframe]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'alone'] = 'No'

    dataset.loc[dataset['relatives'] == 0, 'alone'] = 'Yes'



    

training_dataframe['Age'].fillna(training_dataframe['Age'].median(),inplace=True)

testing_dataframe['Age'].fillna(testing_dataframe['Age'].median(),inplace=True)



#training_dataframe["Age"] = training_dataframe["Age"].fillna(0, inplace=True)

#testing_dataframe["Age"] = testing_dataframe["Age"].fillna(0,inplace=True)

bins = [-1, 0, 14, 25, 35, 60, np.inf]

labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

training_dataframe['AgeGroup'] = pd.cut(training_dataframe["Age"], bins, labels = labels)

testing_dataframe['AgeGroup'] = pd.cut(testing_dataframe["Age"], bins, labels = labels)    

    

# age_mapping = {None: 'Unknown', 1: 'Child',  2: 'Teenager', 3: 'Young Adult', 4:'Adult', 5.5: 'Senior'}

# training_dataframe['AgeGroup'] = training_dataframe['AgeGroup'].map(age_mapping)

# testing_dataframe['AgeGroup'] = testing_dataframe['AgeGroup'].map(age_mapping)  

  



#Since there are missing values in the following columnds we impute them based on their medians 

#training_dataframe['Age'].fillna(training_dataframe['Age'].median(),inplace=True) # Imputing Missing Age Values

training_dataframe['Embarked'].fillna(training_dataframe['Embarked'].value_counts().index[0], inplace=True) # Imputing Missing Embarked Values



#for being able to convert the numerical column Pclass to categorical one, we map its value to 1st, 2nd and 3d

d = {1:'1st',2:'2nd',3:'3rd'} 

training_dataframe['Pclass'] = training_dataframe['Pclass'].map(d) 

training_dataframe.drop(['PassengerId','Name','Ticket'], 1, inplace=True) 

categorical_vars = training_dataframe[['Pclass','Sex','Embarked']] 

dummies = pd.get_dummies(categorical_vars,drop_first=False)



#we drop the columns that are not helpful for our model, like Cabin and transformed features like 'SibSp', 'Parch', 'relatives',

#and those that have converted to one_hot encoding already

training_dataframe = training_dataframe.drop(['Pclass','Sex','Embarked', 'SibSp', 'Parch', 'relatives', 'Cabin', 'Age'],axis=1) #Dropping the Original Categorical Variables to avoid duplicates

training_dataframe = pd.concat([training_dataframe,dummies],axis=1) #Now, concat the new dummy variables



#We do the same thing for the test dataset, and we know that the Fare column has missing value on testset.

#testing_dataframe['Age'].fillna(testing_dataframe['Age'].median(),inplace=True) # Age

testing_dataframe['Fare'].fillna(testing_dataframe['Fare'].median(),inplace=True) # Fare

d = {1:'1st',2:'2nd',3:'3rd'} #Pclass

testing_dataframe['Pclass'] = testing_dataframe['Pclass'].map(d)

testing_dataframe['Embarked'].fillna(testing_dataframe['Embarked'].value_counts().index[0], inplace=True)

ids = testing_dataframe[['PassengerId']]# Passenger Ids

testing_dataframe.drop(['PassengerId','Name','Ticket'],1,inplace=True)

categorical_vars = testing_dataframe[['Pclass','Sex','Embarked']]

dummies = pd.get_dummies(categorical_vars,drop_first=False)

testing_dataframe = testing_dataframe.drop(['Pclass','Sex','Embarked', 'SibSp', 'Parch', 'relatives', 'Cabin', 'Age'],axis=1)#Drop the Original Categorical Variables

testing_dataframe = pd.concat([testing_dataframe,dummies],axis=1)



pd.get_dummies(training_dataframe)
#Just for my readability



testing_dataframe
#Since there is just one Dona in the title of test set and it would break our one_hot encoding we replace it with Mrs.

testing_dataframe.loc[testing_dataframe['Title'] == 'Dona', 'Title'] = "Mrs"

testing_dataframe
pd.get_dummies(testing_dataframe)

#Here we Normalize the continues typed columns like Age and Fare using StandardScaler

from sklearn.preprocessing import StandardScaler



train_numerical_features = list(training_dataframe.select_dtypes(include=['float64']).columns)

ss_scaler = StandardScaler()

train_df_ss = pd.DataFrame(data = training_dataframe)

train_df_ss[train_numerical_features] = ss_scaler.fit_transform(train_df_ss[train_numerical_features])



train_numerical_features = list(testing_dataframe.select_dtypes(include=['float64']).columns)

ss_scaler = StandardScaler()

train_df_ss = pd.DataFrame(data = testing_dataframe)

train_df_ss[train_numerical_features] = ss_scaler.fit_transform(train_df_ss[train_numerical_features])





pd.get_dummies(testing_dataframe)
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split



y = training_dataframe['Survived']

X = pd.get_dummies(training_dataframe.drop(['Survived'],1)) 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



mlp = MLPClassifier(hidden_layer_sizes=(50),max_iter=300,activation='relu', batch_size=40)



mlp.fit(X_train, y_train)

print(mlp.score(X_test, y_test))



mlp.fit(X, y)

predictions = mlp.predict(pd.get_dummies(testing_dataframe))

results = ids.assign(Survived=predictions)



results.to_csv('mlp_submission.csv',index=False)

print("successfully saved the predictions.")
#RandomForestClassifier classifier with cross validation and GridSearch(commented)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



# param_grid = {

#     'bootstrap': [True],

#     'max_depth': [2, 3, 4, 5, 6, 7,8],

#     'max_features': [2, 3],

#     'min_samples_leaf': [1, 2, 3],

#     'min_samples_split': [8, 10, 12],

#     'n_estimators': [40, 60, 80, 100]

# }# Create a based model

# rf = RandomForestClassifier()# Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 2)



y = training_dataframe['Survived']

X = pd.get_dummies(training_dataframe.drop(['Survived'],1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



# grid_search.fit(X, y)

# print(grid_search.best_params_)



rf_model = RandomForestClassifier(n_estimators = 100)

rf_model.fit(X_train, y_train)



print(rf_model.score(X_test, y_test))



rf_model.fit(X, y)

predictions = rf_model.predict(pd.get_dummies(testing_dataframe))



results = ids.assign(Survived=predictions)



results.to_csv('random_forest_submission.csv',index=False)

print("successfully saved the predictions.")
#LogisticRegression classifier with cross validation

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



y = training_dataframe['Survived']

X = pd.get_dummies(training_dataframe.drop(['Survived'],1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



logmodel = LogisticRegression(solver='lbfgs')

# Fit our model to the training data

logmodel.fit(X_train, y_train)

print(logmodel.score(X_test, y_test))



logmodel.fit(X,y)

predictions = logmodel.predict(pd.get_dummies(testing_dataframe))

results = ids.assign(Survived=predictions)



results.to_csv('logistic_submission.csv',index=False)

print("successfully saved the predictions.")
#GradientBoostingClassifier classifier with cross validation



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split



y = training_dataframe['Survived']

X = pd.get_dummies(training_dataframe.drop(['Survived'],1)) 



#print(pd.get_dummies(testing_dataframe))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



gd_model = GradientBoostingClassifier(learning_rate=0.1,max_depth=3)

gd_model.fit(X_train, y_train)

print(gd_model.score(X_test, y_test))



gd_model.fit(X,y)

predictions = gd_model.predict(pd.get_dummies(testing_dataframe))

results = ids.assign(Survived=predictions)



results.to_csv('gradientboost_submission.csv',index=False)

print("successfully saved the predictions.")
#xgboost classifier with cross validation



import xgboost as xgb

from sklearn.model_selection import train_test_split



y = training_dataframe['Survived']

X = pd.get_dummies(training_dataframe.drop(['Survived'],1)) 



#print(pd.get_dummies(testing_dataframe))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train, y_train)

print(xgb_model.score(X_test, y_test))



xgb_model.fit(X, y)

predictions = xgb_model.predict(pd.get_dummies(testing_dataframe))

results = ids.assign(Survived=predictions)



results.to_csv('xgb_submission.csv',index=False)

print("successfully saved the predictions.")
#SVM classifier with cross validation

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



y = training_dataframe['Survived']

X = pd.get_dummies(training_dataframe.drop(['Survived'],1)) 



#print(pd.get_dummies(testing_dataframe))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)





svclassifier = SVC(kernel='rbf',gamma=0.1, C=10)

svclassifier.fit(X_train, y_train)

print(svclassifier.score(X_test, y_test))



svclassifier.fit(X,y)

predictions = svclassifier.predict(pd.get_dummies(testing_dataframe))

results = ids.assign(Survived=predictions)



results.to_csv('svm_submission.csv',index=False)

print("successfully saved the predictions.")
#A simple ensemble using our previously trained classifiers => random forest, mlp, xgb, etc...

from sklearn.ensemble import  VotingClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



ensemble_majority_voting = VotingClassifier(estimators=[

        ('lr', logmodel), ('rf', rf_model), ('gd', gd_model)

    ,('xgb', xgb_model), ('mlp', mlp)], voting='soft', weights=[1,1,1,2,3],

       flatten_transform=True)



ensemble_majority_voting = ensemble_majority_voting.fit(X_train, y_train)

print(ensemble_majority_voting.score(X_test, y_test))



ensemble_majority_voting.fit(X,y)

predictions = ensemble_majority_voting.predict(pd.get_dummies(testing_dataframe))

results = ids.assign(Survived=predictions)



results.to_csv('majority_voting_submission.csv',index=False)

print("successfully saved the predictions.")
#Ensemble classifier using bagging

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)





cart = DecisionTreeClassifier()

num_trees = 200

bagging_model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)





bagging_model = bagging_model.fit(X_train, y_train)

print(bagging_model.score(X_test, y_test))
