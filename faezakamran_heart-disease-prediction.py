# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
# read the data

heart_data = pd.read_csv('../input/heart.csv', sep=',')
# view all column names and few rows of data

heart_data.describe()
sns.set()

# people with heart attack based on sex,age,cp

sns.catplot(x="sex", y="age",col="target",hue="cp", kind="swarm",data=heart_data)
# comparison of people with heart attack on basis of sex

sns.countplot(x="target", hue="sex",data=heart_data)

plt.show()
# comparison of fbs on basis of sex

sns.countplot(x="fbs", hue="sex",data=heart_data)

plt.show()
# comparison of fbs on basis of people who had a heart attack

sns.countplot(x="fbs", hue="target",data=heart_data)

plt.show()
# comparison of restecg on basis of sex

sns.countplot(x="restecg", hue="sex",data=heart_data)

plt.show()
# comparison of restecg on basis of people who had a heart attack

sns.countplot(x="restecg", hue="target",data=heart_data)

plt.show()
# comparison of thal on basis of sex

sns.countplot(x="thal", hue="sex",data=heart_data)

plt.show()
# comparison of thal on basis of people who had a heart attack

sns.countplot(x="thal", hue="target",data=heart_data)

plt.show()
import numpy as np

from sklearn.model_selection import train_test_split



y = heart_data.target.values

X = heart_data.drop(['target'], axis = 1)



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



# Define the model. Set random_state to 1

heart_model = RandomForestClassifier(n_estimators = 1000, random_state=1)





# fit your model

heart_model.fit(train_X,train_y)
val_predictions = heart_model.predict(val_X)

# Calculate the mean absolute error of your Random Forest model on the validation data

heart_val_mae = mean_absolute_error(val_predictions, val_y)

acc = accuracy_score(val_y, val_predictions)



print("Validation MAE for Random Forest Model: {}".format(heart_val_mae))

print("Validation Accuracy for Random Forest Model: {}".format(acc))



print("\nPredicted Values:")

print(val_predictions)

print("\nActual Values:")

print(val_y)