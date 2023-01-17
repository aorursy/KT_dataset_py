# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import pandas as pd

##exploring train data

train_data = pd.read_csv(r'/kaggle/input/learn-together/train.csv')

train_data.describe()

train_data.head()

train_data.columns
##target variable

Train_y = train_data.Cover_Type



##features

train_features = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']



Train_X = train_data[train_features]
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(Train_X, Train_y, random_state = 0)
##decisiontree



from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

model = DecisionTreeClassifier()



model.fit(train_X,train_y)



predictions = model.predict(val_X)



##predictions = DecisionTreeClassifier()



##train_model.fit(train_X,train_y)



##predictions = model.predict(val_X)train_model = DecisionTreeRegressor(random_state=1)

##train_model.fit(train_X, train_y)
from sklearn.metrics import accuracy_score

accuracy_score(predictions,val_y)
##predictions for first 5 houses, before checking with test data



print("Making predictions for the following 5 houses:")

print(Train_X.head())

print("The predictions are")

print(model.predict(Train_X.head()))




###measuring for test data

test_data = pd.read_csv('/kaggle/input/learn-together/test.csv')



###append covertype column to test data

import numpy as np



test_data['Cover_Type']=np.nan



test_data.head()
##target variable for test

Test_y = test_data.Cover_Type



##features for test

test_features = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']



Test_X = test_data[test_features]
##run train model on test



test_predictions = model.predict(Test_X)



my_submission = pd.DataFrame({'Id': pd.read_csv('../input/learn-together/test.csv')['Id'], 'Cover_Type': test_predictions})

my_submission.to_csv('submission_data_cleaned.csv', index=False)

##solutions = test_data.Id,test_predictions





##Data_Crunchers = pd.DataFrame(solutions) 

##Data_Crunchers_transposed = Data_Crunchers.T
##Data_Crunchers_transposed.head()
##Data_Crunchers_transposed.columns = [

 ## 'Id',

 ## 'Cover_Type'

##]

##Data_Crunchers_transposed

##save file

##write_csv(Data_Crunchers_transposed, "submission.csv")
##export_csv = Data_Crunchers.to_csv (r'G:\Kaggle\DC_Solutions.csv', index = None, header=False) #Don't forget to add '.csv' at the end of the path

##Data_Crunchers_transposed.to_csv(r'C:\Users\VMRA\Desktop\kaggle practisse\DC_Solutions.csv')
##Data_Crunchers_transposed.to_excel(r"G:\Kaggle\DC_Solutions.xlsx") 
###Data_Crunchers_transposed.info()
###kaggle competitions submit -c learn-together -f Data_Crunchers_transposed.csv -m "Message"

##Data_Crunchers_transposed.to_csv('submission_data_cleaned.csv', index=False)
##type(Data_Crunchers_transposed)