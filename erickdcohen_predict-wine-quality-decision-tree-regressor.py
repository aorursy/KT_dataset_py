import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, mean_absolute_error
# load in the data

wineFilePath = "/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv"



wineDf = pd.read_csv(wineFilePath)
wineDf.head()
wineDf.describe()
y = wineDf.quality



predictorCols = wineDf.columns.drop("quality")

x = wineDf[predictorCols]
print(wineDf.head())



print("\nPredictions: ")

print(wine_model.predict(x.head()))
# Split the data into training and testing 

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 100)
wine_model = DecisionTreeRegressor(random_state = 2)



wine_model.fit(train_x, train_y)
val_predictions = wine_model.predict(val_x)



predictions = wine_model.predict(x)
mae_val = mean_absolute_error(val_y, val_predictions)

mae_real = mean_absolute_error(y, predictions)



print("The Validation MAE is: " + str(mae_val))

print("The model MAE is : " + str(mae_real))
# confusion matrix 

confusion_matrix(wineDf.quality, predictions)