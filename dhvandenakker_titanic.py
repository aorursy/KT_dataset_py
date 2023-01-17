import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# read data and inspect it
Train_Data = pd.read_csv("../input/train.csv")
Test_Data = pd.read_csv("../input/test.csv")
Train_Data.head()
# Check if Train data contains missing values
print("Train_Data\n", Train_Data.isna().sum())
print("Test_Data\n", Test_Data.isna().sum())
# Check the distribution of the Age
Train_Data.Age.hist()
Train_Data.Fare.hist()
#replace missing age values with median
median_age = Train_Data.Age.median()
median_fare = Train_Data.Fare.median()

Train_Data.Age.fillna(median_age, inplace=True)

Test_Data.Age.fillna(median_age, inplace=True)
Test_Data.Fare.fillna(median_fare, inplace=True)
# encode pclass and sex as one hot and add it to the Data
Train_Data = pd.get_dummies(Train_Data, columns=['Sex', 'Pclass' ])

Test_Data = pd.get_dummies(Test_Data, columns=['Sex', 'Pclass'])

# Make a boolean feature "Known Cabin" if the cabin was documentated
Train_Data["Known_Cabin"] = ~Train_Data.Cabin.isna()
Train_Data["Known_Cabin"] = Train_Data.Known_Cabin.astype(int).values


Test_Data["Known_Cabin"] = ~Test_Data.Cabin.isna()
Test_Data["Known_Cabin"] = Test_Data.Known_Cabin.astype(int).values
# add the mean age of a family (quick and dirty determined by grouping on the family name) as feature

def search_re_pattern(pattern, string, default_return_value=None):
    """ returns the string of the result directly if present, else returns the default return value"""
    
    result = re.search(pattern, string)
    if result:
        return result.group()
    return default_return_value

def mean_family_age(Data):
    """ Returns Data frame with additional column that contains the mean age of the family"""
    
    # group families based on the familiy name
    name_pattern = "^.*(?=,)"
    Data["Family_Name"] = Data.Name.apply(lambda x: search_re_pattern(name_pattern, x))
    Family_Dict = Data.groupby("Family_Name").groups
    
    mean_family_age = []
    
    for index, row in Data.iterrows():
        
        current_family = Data.loc[Family_Dict[row["Family_Name"]]]
        mean_family_age.append(current_family["Age"].mean())

    Data["Fam_Age"] = mean_family_age
    return Data

# use function to add the information to the dataframes
Train_Data, Test_Data = mean_family_age(Train_Data), mean_family_age(Test_Data)
# split the train set into two subsets
train_set, test_set = train_test_split(Train_Data)

# features to exclude from dataset
to_drop = ['PassengerId', "Name", "Ticket", "Cabin", "Embarked", "Survived", "Family_Name"]

# split the feature matrix and outcome for both train and test set
X_train, y_train = train_set.drop(columns=to_drop), train_set[['Survived']].values.ravel()
X_test, y_test = test_set.drop(columns=to_drop), test_set[["Survived"]].values.ravel()

# train rf using parameter grid search
rfc_parameter_grid = {'n_estimators':[10, 15, 20, 25, 30], 
                      'criterion':['gini'],
                      'max_depth':[3, 5, 7, 10, 15],
                      'min_samples_leaf':[1, 2, 5, 10],
                        }

rf = RandomForestClassifier()
rf_search = GridSearchCV(rf, rfc_parameter_grid, cv=10)
rf_search.fit(X_train, y_train)
# select best estimator
best_rf = rf_search.best_estimator_

for parameter, value in best_rf.get_params().items():
    print(parameter, " ", value)
print("Score on train\n", best_rf.score(X_train, y_train), "\nScore on test\n", best_rf.score(X_test, y_test))
# still overfitting... use it for the final predictions anyway

# is not in testdata so remove from to drop list
to_drop.remove("Survived")

# add predictions
X = Test_Data.drop(columns=to_drop)
Test_Data["Survived"] = best_rf.predict(X)

# save outputfile
Test_Data[["PassengerId", "Survived"]].to_csv("Predictions.csv", index=False)