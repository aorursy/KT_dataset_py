import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedShuffleSplit
# Loading the data

data = pd.read_csv("../input/covtype.csv")
# Viewing Distribution of Data

data.head()

# Viewing Data Types

data.dtypes
# Converting Response Variable to Category

data['Cover_Type'] =  data['Cover_Type'].astype("category")
# Selecting the important variables for further analysis

sel_var = ['Elevation','Aspect','Slope' , 'Horizontal_Distance_To_Hydrology' ,'Vertical_Distance_To_Hydrology' ,

'Horizontal_Distance_To_Roadways' , 'Hillshade_Noon' , 'Hillshade_3pm' ,  'Horizontal_Distance_To_Fire_Points' , 'Wilderness_Area2' ,

    'Wilderness_Area3', 'Wilderness_Area4' , 'Soil_Type10', 'Soil_Type12', 'Soil_Type22' , 'Soil_Type23' , 'Soil_Type29' ,

                 'Soil_Type30' , 'Soil_Type32' , 'Soil_Type33' ,'Cover_Type']
X_data = data[sel_var[:-1]]

Y_data = data[['Cover_Type']]

Y_data.columns
# Stratified Split of train and test data

from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(data['Cover_Type'],  test_size=0.3)
for train_index, test_index in sss:

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]

    Y_train, Y_test = Y_data.iloc[train_index], Y_data.iloc[test_index]
# Shape of train and test data

X_train.shape,  X_test.shape
# Shape of response variable in train

Y_train.shape
# Model Development

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini')

rf.fit(X_train, Y_train)
# Prediction On test data

pred = rf.predict(X_test)
##  Accuracy of model on test data

sum(pred == Y_test['Cover_Type'])/float(len(Y_test))