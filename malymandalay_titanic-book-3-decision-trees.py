# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn



from sklearn import tree

from sklearn import preprocessing



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = [train , test]



# Missing values 



for dataset in full_data: 

    mean_age = dataset['Age'].mean()

    std_age = dataset['Age'].std() 

    missing_age = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(mean_age - std_age, mean_age + std_age, size=missing_age)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())  

    

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S') #most common value 



# Create bands  

    

for dataset in full_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

     

bins = [0, 18, 34, 50, 200] 

group_names = [1, 2, 3 , 4]



for dataset in full_data:

    categories = pd.cut(dataset['Age'], bins, labels=group_names)

    dataset['Age_cat'] = pd.cut(dataset['Age'], bins, labels=group_names)



# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = 0

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset.loc[dataset['FamilySize'] > 8,"FamilySize"]=8 #meaning 8+ 

    

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



# Create Title variable 

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    dataset['Mother'] = 0

    dataset.loc[(dataset['Age'] > 18) & (dataset['Parch'] > 1) & (dataset['Title'] != 'Miss'),"Mother"]=1

    #dataset['Mother'][(dataset['Age'] > 18) & (dataset['Parch'] > 1) & (dataset['Title'] != 'Miss')] = 1



# So, we can classify passengers as males, females, and child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



for dataset in full_data:

    dataset['Person'] = dataset[['Age','Sex']].apply(get_person,axis=1)
test.head(3)
# Initialize label encoder

label_encoder = preprocessing.LabelEncoder()



# Convert Sex variable to numeric

encoded_Person = label_encoder.fit_transform(train["Person"])



# Initialize model

tree_model = tree.DecisionTreeClassifier()



# Train the model

tree_model.fit(X = pd.DataFrame(encoded_Person), 

               y = train["Survived"])
## I wanted to see an image of the decision tree but this doesn't seem to be working on Kaggle :) 



#import pydotplus

#from pydotplus import graphviz

#from IPython.display import Image

       

#dot_data = tree.export_graphviz(tree_model, feature_names=["Person"])

#graph = pydotplus.graphviz.graph_from_dot_file(dot_data)



#Image(graph.create_png())             # Display image*
# Get survival probability

preds = tree_model.predict_proba(X = pd.DataFrame(encoded_Person))



pd.crosstab(preds[:,0], train["Person"])
tree_model.score(X =  pd.DataFrame(encoded_Person), 

                 y = train["Survived"])
train.head(3)
# Convert categorical variables variable to numeric



encoded_Person = label_encoder.fit_transform(train["Person"])

encoded_Embarked = label_encoder.fit_transform(train["Embarked"])

encoded_Age = label_encoder.fit_transform(train["Age_cat"])

encoded_Title = label_encoder.fit_transform(train["Title"])



# define predictors 

#variables = [encoded_Person, encoded_Embarked, encoded_Age, encoded_Title, train["Fare"] , train["FamilySize"] ]

variables = [encoded_Person,  train["Age"] ,  train["Pclass"] , encoded_Title,  train["Fare"] , train["FamilySize"] ]

# Make data frame of predictors

predictors = pd.DataFrame(variables).T



max_depth = 8

min_samples_split = 10



# Initialize model with options 

tree_model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)



# Train the model

tree_model.fit(X = predictors, 

               y = train["Survived"])



# Get survival probability

preds = tree_model.predict_proba(X = predictors)



# Get cross table (only with 2 variables )

#pd.crosstab(preds[:,0], columns = [train["Age_cat"], 

 #                                  train["Sex"]])



tree_model.score(X =  predictors, 

                y = train["Survived"])
from sklearn.cross_validation import KFold



cv = KFold(n=len(train),  # Number of elements

           n_folds=10,            # Desired number of cv folds

           random_state=12)       # Set a random seed



#After creating a cross validation object, you can loop over each fold and train and evaluate a your model on each one:



fold_accuracy = []



# Convert Sex variable to numeric

encoded_sex = label_encoder.fit_transform(train["Sex"])

train["Sex"] = encoded_sex





for train_fold, valid_fold in cv:

    train_k = train.loc[train_fold] # Extract train data with cv indices

    valid_k = train.loc[valid_fold] # Extract valid data with cv indices

    

    model = tree_model.fit(X = train_k[["Sex","Pclass","Age","Fare"]], 

                           y = train_k["Survived"])

    valid_acc = model.score(X = valid_k[["Sex","Pclass","Age","Fare"]], 

                            y = valid_k["Survived"])

    fold_accuracy.append(valid_acc)    



print("Accuracy per fold: ", fold_accuracy, "\n")

print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))
from sklearn.cross_validation import cross_val_score



scores = cross_val_score(estimator= tree_model,     # Model to test

                X= train[["Sex","Pclass",   # Train Data

                                  "Age","Fare"]],  

                y = train["Survived"],      # Target variable

                scoring = "accuracy",               # Scoring metric    

                cv=10)                              # Cross validation folds



print("Accuracy per fold: ")

print(scores)

print("Average accuracy: ", scores.mean())
from sklearn.cross_validation import train_test_split



v_train, v_test = train_test_split(train,     # Data set to split

                                   test_size = 0.25,  # Split ratio

                                   random_state=1,    # Set random seed

                                   stratify = train["Survived"]) #*



# Training set size for validation

print(v_train.shape)

# Test set size for validation

print(v_test.shape)
full_data = [v_train , v_test]



max_depth = 8

min_samples_split = 10

tree_model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)



def model_validation(dataset):

    # Convert categorical variables variable to numeric

    encoded_Person = label_encoder.fit_transform(dataset["Person"])

    encoded_Embarked = label_encoder.fit_transform(dataset["Embarked"])

    encoded_Age = label_encoder.fit_transform(dataset["Age_cat"])

    encoded_Title = label_encoder.fit_transform(dataset["Title"])



    variables = [encoded_Person,  dataset["Age"] ,  dataset["Pclass"] , encoded_Title,  dataset["Fare"] , dataset["FamilySize"] ]

    # Make data frame of predictors

    predictors = pd.DataFrame(variables).T



    tree_model.fit(X = predictors, 

                   y = dataset["Survived"])



    # Get survival probability

    preds = tree_model.predict_proba(X = predictors)



    return tree_model.score(X =  predictors, 

                        y = dataset["Survived"])

    

x = model_validation(v_train)

y = model_validation(v_test)



print ("Train Accuracy  = ", x) 

print ("Test Accuracy  = ", y) 
# Convert categorical variables variable to numeric

encoded_Person = label_encoder.fit_transform(test["Person"])

encoded_Embarked = label_encoder.fit_transform(test["Embarked"])

encoded_Age = label_encoder.fit_transform(test["Age_cat"])

encoded_Title = label_encoder.fit_transform(test["Title"])



variables_test = [encoded_Person,  test["Age"] ,  test["Pclass"] , encoded_Title,  test["Fare"] , test["FamilySize"] ]

# Make data frame of predictors

test_features = pd.DataFrame(variables_test).T



# Make test set predictions

test_preds = tree_model.predict(X=test_features)



# Create a submission for Kaggle

submission = pd.DataFrame({"PassengerId":test["PassengerId"],

                           "Survived":test_preds})



# Save submission to CSV

submission.to_csv("tutorial_dectree_submission.csv", 

                  index=False)        # Do not save index values