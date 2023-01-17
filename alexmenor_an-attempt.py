# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('../input/BlackFriday.csv')

raw_data.head()
raw_data.info()

prediction_data = raw_data.drop(["User_ID", "Product_ID"],axis=1)
prediction_data = prediction_data.drop("Product_Category_2", axis = 1)

prediction_data = prediction_data.drop("Product_Category_3", axis = 1)

prediction_data.info()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

toEncode = prediction_data["Gender"]

genderEncoded = encoder.fit_transform(toEncode)

prediction_data["Gender"] = genderEncoded

prediction_data.head()
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

age_1hot = encoder.fit_transform(prediction_data["Age"])

print(encoder.classes_)

print(age_1hot)



# Probably there's a much better way to do this

age_1hot = age_1hot.transpose()

for (cat, ar) in zip(encoder.classes_, age_1hot):

    prediction_data[cat] = ar

print(prediction_data.head())

prediction_data = prediction_data.drop("Age", axis=1)

prediction_data.head()
encoder = LabelEncoder()

toEncode = prediction_data["City_Category"]

cityEncoded = encoder.fit_transform(toEncode)

prediction_data["City_Category"] = cityEncoded

print(encoder.classes_)

prediction_data.head()
encoder = LabelEncoder()

toEncode = prediction_data["Stay_In_Current_City_Years"]

yearsEncoded = encoder.fit_transform(toEncode)

prediction_data["Stay_In_Current_City_Years"] = yearsEncoded

print(prediction_data.info())

prediction_data.head()
prediction_data["Gender"].value_counts() / len(raw_data)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(prediction_data, prediction_data["Gender"]):

    train_set = prediction_data.loc[train_index]

    test_set = prediction_data.loc[test_index]
train_set["Gender"].value_counts()/len(train_set)

test_set["Gender"].value_counts()/len(test_set)
corr_matrix = prediction_data.corr()

corr_matrix["Purchase"].sort_values(ascending=False)
prediction_data.hist(bins=50,figsize=(20,15))
blackFriday = train_set.drop("Purchase", axis=1)

blackFridayLabels = train_set["Purchase"].copy()
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(blackFriday, blackFridayLabels)
some_data = blackFriday.iloc[:10]

some_labels = blackFridayLabels.iloc[:10]

print("Predictions:\t",lin_reg.predict(some_data))

print("Labels:\t", list(some_labels))
from sklearn.metrics import mean_squared_error

predictions = lin_reg.predict(some_data)

lin_mse = mean_squared_error(predictions, some_labels)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, blackFriday, blackFridayLabels, 

                         scoring = "neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print("\nScores: ", rmse_scores, "\nMean: ", rmse_scores.mean()

     ,"\nStd Deviation: " , rmse_scores.std())
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10)

scores_forest = cross_val_score (forest_reg, blackFriday, 

                                 blackFridayLabels,

                                 scoring = "neg_mean_squared_error",

                                 cv=10)

rmse_scores_forest = np.sqrt(-scores_forest)
print("\nScores: ", rmse_scores_forest, "\nMean: ", rmse_scores_forest.mean()

     ,"\nStd Deviation: " , rmse_scores_forest.std())
forest_reg.fit(blackFriday, blackFridayLabels)
test_x = test_set.drop("Purchase", axis=1)

test_labels = test_set["Purchase"].copy()

final_predictions = forest_reg.predict(test_x)

final_mse = mean_squared_error(test_labels, final_predictions)

final_rmse = np.sqrt(final_mse)

print("Final RMSE: ", final_rmse)