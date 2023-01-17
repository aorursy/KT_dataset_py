import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
# Uploading the dataset

graduate_admisssion_data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
# Test to see if the data can open

graduate_admisssion_data.head()
# Quick analzation of the dataset

graduate_admisssion_data.describe()
# Check if the columns name are correctly named

graduate_admisssion_data.columns

# Clean the titles of the columns, because there are spaces

graduate_admisssion_data.columns = graduate_admisssion_data.columns.str.replace(' ', '_')

#Check to see if the spaces have been replaced

graduate_admisssion_data.columns

y = graduate_admisssion_data.Chance_of_Admit_

admission_features = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR_', 'CGPA', 'Research']

X = graduate_admisssion_data[admission_features]
admission_model = DecisionTreeRegressor(random_state=1)

admission_model.fit(X, y)
print("Making predictions for the following 5 students:")

print(X.head())

print("The predictions are")

print(admission_model.predict(X.head()))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

        model.fit(train_X, train_y)

        preds_val = model.predict(val_X)

        mae = mean_absolute_error(val_y, preds_val)

        return(mae)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

admission_model = DecisionTreeRegressor()

admission_model.fit(train_X, train_y)

val_predictions = admission_model.predict(val_X)

print (mean_absolute_error(val_y, val_predictions))