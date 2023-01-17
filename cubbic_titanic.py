# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



print("setup done")
titanic_data=pd.read_csv("../input/titanic/train.csv")

print("All columns present in the csv:")

print(titanic_data.columns)



# can't use strings in RandomForestRegressor

titanic_data.Sex.replace({"male": 1, "female": 0}, inplace=True)


# Not all columns are important, for example the number of siblings outside the ship probably affects nothing

# https://www.kaggle.com/c/titanic/datacheck definitions of what each column means

features_set = ["Pclass", "Sex", "Age", "Fare"]







features_set_data = titanic_data[features_set]





X = features_set_data[features_set]



y = titanic_data.Survived



X
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

inputted_train_X =  my_imputer.fit_transform(train_X)

inputted_val_X = my_imputer.fit_transform(val_X)

    

   

model = RandomForestRegressor(random_state=1)

model.fit(inputted_train_X, train_y)



predictions = model.predict(inputted_val_X)



mean_absolute_error(val_y, predictions)



# full model



model = RandomForestRegressor(random_state=1)

inputted_X =  my_imputer.fit_transform(X)

model.fit(inputted_X, y)



test_data=pd.read_csv("../input/titanic/test.csv")

# can't use strings in RandomForestRegressor

test_data.Sex.replace({"male": 1, "female": 0}, inplace=True)

featured_test_data = test_data[features_set]

inputted_featured_test_data = my_imputer.fit_transform(featured_test_data)

predictions = model.predict(inputted_featured_test_data)



predictions_whole_number = []



for prediction in predictions:

    if(prediction > 0.5):

        predictions_whole_number.append(1)

    else: 

        predictions_whole_number.append(0)



output_data = pd.DataFrame(index= test_data.PassengerId)



output_data["Survived"] = predictions_whole_number



output_data.to_csv(path_or_buf="gender_submission.csv")



output_data
