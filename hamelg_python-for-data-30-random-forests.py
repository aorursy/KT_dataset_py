import numpy as np

import pandas as pd

import os
# Load and prepare Titanic data

titanic_train = pd.read_csv("../input/train.csv")    # Read the data



# Impute median Age for NA Age values

new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check

                       28,                       # Value if check is true

                       titanic_train["Age"])     # Value if check is false



titanic_train["Age"] = new_age_var 
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
# Set the seed

np.random.seed(12)



# Initialize label encoder

label_encoder = preprocessing.LabelEncoder()



# Convert some variables to numeric

titanic_train["Sex"] = label_encoder.fit_transform(titanic_train["Sex"])



# Initialize the model

rf_model = RandomForestClassifier(n_estimators=1000, # Number of trees

                                  max_features=2,    # Num features considered

                                  oob_score=True)    # Use OOB scoring*



features = ["Sex","Pclass","SibSp","Age","Fare"]



# Train the model

rf_model.fit(X=titanic_train[features],

             y=titanic_train["Survived"])



print("OOB accuracy: ")

print(rf_model.oob_score_)
for feature, imp in zip(features, rf_model.feature_importances_):

    print(feature, imp)
# Read and prepare test data

titanic_test = pd.read_csv("../input/test.csv")    # Read the data



# Impute median Age for NA Age values

new_age_var = np.where(titanic_test["Age"].isnull(),

                       28,                      

                       titanic_test["Age"])      



titanic_test["Age"] = new_age_var 



# Fill missing Fare with 50

new_fare_var = np.where(titanic_test["Fare"].isnull(),

                       50,                      

                       titanic_test["Fare"])      



titanic_test["Fare"] = new_fare_var 



# Convert some variables to numeric

titanic_test["Sex"] = label_encoder.fit_transform(titanic_test["Sex"])
# Make test set predictions

test_preds = rf_model.predict(X = titanic_test[features])



# Create a submission for Kaggle

submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],

                           "Survived":test_preds})



# Save submission to CSV

submission.to_csv("tutorial_randomForest_submission.csv", 

                  index=False)        # Do not save index values