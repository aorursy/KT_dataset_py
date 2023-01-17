# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



!ls /kaggle

!ls /kaggle/input

!ls /kaggle/input/learn-together



main_file_path = '../input/learn-together/train.csv'

main_data = pd.read_csv(main_file_path)

main_data.head()



#print(main_data.columns)



y = main_data.Cover_Type # Target variable for scoring



main_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']



X = main_data[main_features]



train_X, train_y, val_X, val_y = train_test_split(X, y, random_state = 0)



forest_model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 1)

forest_model.fit(X, y)



val_predictions = forest_model.predict(X)

print(mean_absolute_error(y, val_predictions))





test_file_path = '../input/learn-together/test.csv'

test_data = pd.read_csv(test_file_path)



test_X = test_data[main_features]

final_predictions = forest_model.predict(test_X)



output = pd.DataFrame({'Id': test_data.Id, 'Cover_Type': final_predictions})

output.to_csv('submission.csv', index=False)
