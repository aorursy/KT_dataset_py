import numpy as np

import pandas as pd
# Load and prepare Titanic data

titanic_train = pd.read_csv("../input/train.csv")    # Read the data



# Impute median Age for NA Age values

new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check

                       28,                       # Value if check is true

                       titanic_train["Age"])     # Value if check is false



titanic_train["Age"] = new_age_var 
from sklearn import tree

from sklearn import preprocessing
# Initialize label encoder

label_encoder = preprocessing.LabelEncoder()



# Convert Sex variable to numeric

encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])



# Initialize model

tree_model = tree.DecisionTreeClassifier()



# Train the model

tree_model.fit(X = pd.DataFrame(encoded_sex), 

               y = titanic_train["Survived"])
import graphviz



# Save tree as dot file

dot_data = tree.export_graphviz(tree_model, out_file=None) 

graph = graphviz.Source(dot_data)  

graph 
# Get survival probability

preds = tree_model.predict_proba(X = pd.DataFrame(encoded_sex))



pd.crosstab(preds[:,0], titanic_train["Sex"])
# Make data frame of predictors

predictors = pd.DataFrame([encoded_sex, titanic_train["Pclass"]]).T



# Train the model

tree_model.fit(X = predictors, 

               y = titanic_train["Survived"])
# Save tree as dot file

dot_data = tree.export_graphviz(tree_model, out_file=None) 

graph = graphviz.Source(dot_data)  

graph 
# Get survival probability

preds = tree_model.predict_proba(X = predictors)



# Create a table of predictions by sex and class

pd.crosstab(preds[:,0], columns = [titanic_train["Pclass"], 

                                   titanic_train["Sex"]])
predictors = pd.DataFrame([encoded_sex,

                           titanic_train["Pclass"],

                           titanic_train["Age"],

                           titanic_train["Fare"]]).T



# Initialize model with maximum tree depth set to 8

tree_model = tree.DecisionTreeClassifier(max_depth = 8)



tree_model.fit(X = predictors, 

               y = titanic_train["Survived"])
# Save tree as dot file

dot_data = tree.export_graphviz(tree_model, out_file=None) 

graph = graphviz.Source(dot_data)  

graph 
tree_model.score(X = predictors, 

                 y = titanic_train["Survived"])
# Read and prepare test data

titanic_test = pd.read_csv("../input/test.csv")    # Read the data



# Impute median Age for NA Age values

new_age_var = np.where(titanic_test["Age"].isnull(), # Logical check

                       28,                       # Value if check is true

                       titanic_test["Age"])      # Value if check is false



new_fare_var = np.where(titanic_test["Fare"].isnull(), # Logical check

                       50,                       # Value if check is true

                       titanic_test["Fare"])      # Value if check is false



titanic_test["Age"] = new_age_var 

titanic_test["Fare"] = new_fare_var
# Convert test variables to match model features

encoded_sex_test = label_encoder.fit_transform(titanic_test["Sex"])



test_features = pd.DataFrame([encoded_sex_test,

                              titanic_test["Pclass"],

                              titanic_test["Age"],

                              titanic_test["Fare"]]).T
# Make test set predictions

test_preds = tree_model.predict(X=test_features)



# Create a submission for Kaggle

submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],

                           "Survived":test_preds})



# Save submission to CSV

submission.to_csv("tutorial_dectree_submission.csv", 

                  index=False)        # Do not save index values
from sklearn.model_selection import train_test_split
v_train, v_test = train_test_split(titanic_train,     # Data set to split

                                   test_size = 0.25,  # Split ratio

                                   random_state=1,    # Set random seed

                                   stratify = titanic_train["Survived"]) #*



# Training set size for validation

print(v_train.shape)

# Test set size for validation

print(v_test.shape)
from sklearn.model_selection import KFold



kf = KFold(n_splits=10, random_state=12)

kf.get_n_splits(titanic_train)
fold_accuracy = []



titanic_train["Sex"] = encoded_sex



for train_fold, valid_fold in kf.split(titanic_train):

    train = titanic_train.loc[train_fold] # Extract train data with cv indices

    valid = titanic_train.loc[valid_fold] # Extract valid data with cv indices

    

    model = tree_model.fit(X = train[["Sex","Pclass","Age","Fare"]], 

                           y = train["Survived"])

    valid_acc = model.score(X = valid[["Sex","Pclass","Age","Fare"]], 

                            y = valid["Survived"])

    fold_accuracy.append(valid_acc)    



print("Accuracy per fold: ", fold_accuracy, "\n")

print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator= tree_model,     # Model to test

                X= titanic_train[["Sex","Pclass",   # Train Data

                                  "Age","Fare"]],  

                y = titanic_train["Survived"],      # Target variable

                scoring = "accuracy",               # Scoring metric    

                cv=10)                              # Cross validation folds



print("Accuracy per fold: ")

print(scores)

print("Average accuracy: ", scores.mean())