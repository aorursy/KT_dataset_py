# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
example_submission = pd.read_csv('../input/gender_submission.csv')
training_data = pd.read_csv('../input/train.csv')
my_imputer = SimpleImputer()
prediction_fields=["Pclass", "Age", "SibSp", "Parch", "Fare"]
X_with_missing_values=training_data[prediction_fields]
X = my_imputer.fit_transform(X_with_missing_values)
y=training_data.Survived

#Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1)

#mean_absolute_error function
def the_mean_absolute_error(max_leaf_nodes):
    decision_tree_model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    decision_tree_model.fit(X_train,y_train)
    y_pred=decision_tree_model.predict(X_val)
    the_mean_absolute_error = mean_absolute_error(y_pred, y_val)
    return the_mean_absolute_error
#determine a maximum number of leaf nodes
possible_max_leaf_nodes = [520,40,60,80,100,120,140,160,180,200,220,240,260,280,300]

for number_of_leaves in possible_max_leaf_nodes:
    try:
        mae
        best_number_of_leaves
    except:
        mae=the_mean_absolute_error(number_of_leaves)
        best_number_of_leaves=number_of_leaves
    else:
        if the_mean_absolute_error(number_of_leaves)<mae:
            mae = the_mean_absolute_error(number_of_leaves)
            best_number_of_leaves=number_of_leaves

print("The best number of leaves is %d which gives a mean absolute error of %f when using a decision tree model." %(best_number_of_leaves, mae))
#Training with Random Forest
random_forest_model=RandomForestRegressor(random_state=1)
random_forest_model.fit(X_train,y_train)
RF_y_pred=random_forest_model.predict(X_val)
RF_mae=mean_absolute_error(RF_y_pred, y_val)

print("The mae when using a random forest model is %f" %(RF_mae))
#Train using the whole dataset 
final_model = DecisionTreeRegressor(max_leaf_nodes=best_number_of_leaves, random_state=1)
final_model.fit(X,y)
testing_data = pd.read_csv('../input/test.csv')

testing_columns_with_missing_values = testing_data[prediction_fields]
testing_columns = my_imputer.fit_transform(testing_columns_with_missing_values)

test_predictions=final_model.predict(testing_columns).astype(int)

my_very_first_submission=pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': test_predictions})
my_very_first_submission.to_csv('submission.csv', index=False)




