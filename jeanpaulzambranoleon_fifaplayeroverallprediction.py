# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fifa_19_path = '../input/fifa-20-complete-player-dataset/players_19.csv'

fifa_20_path = '../input/fifa-20-complete-player-dataset/players_20.csv'



fifa_19_df = pd.read_csv(fifa_19_path)

fifa_20_df = pd.read_csv(fifa_20_path)



fifa_19_df.columns
features = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling',

            'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning'] #choose the player features that we want to train



labels_df = fifa_19_df.overall #this is the labels that we will train with the player features



fifa_19_features = fifa_19_df[features] #fifa 19 player features



fifa_20_features = fifa_20_df[features] #fifa 20 player features
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer() #this function will replace the Nan values in the training and testing data

fifa_19_new_df = pd.DataFrame(my_imputer.fit_transform(fifa_19_features))

fifa_19_new_df.columns = fifa_19_features.columns #needs to add the columns back to the new training data



fifa_20_new_df = pd.DataFrame(my_imputer.fit_transform(fifa_20_features))

fifa_20_new_df.columns = fifa_20_features.columns #needs to add the columns back
X_train, X_test, y_train, y_test = train_test_split(fifa_19_new_df, labels_df)

print("Length of training data: " + str(len(X_train)))

print("Length of testing data: " + str(len(X_test)))

print("Length of total data: " + str(len(fifa_19_new_df)))
from xgboost import XGBRegressor

my_model = XGBRegressor()

my_model.fit(X_train, y_train)

my_model.predict(X_test.head()) #this will predict the overall of the top 5 playesr in the testing data
y_test.head() #this is the actual overall of the fifa 19 test data
my_model.predict(fifa_20_new_df.head()) #this is the prediction based on the fifa 20 player features
fifa_20_df.overall.head() #this is the actual overall of the top 5 players in fifa 20