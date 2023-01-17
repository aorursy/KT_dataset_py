# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

melb_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

melb_data.describe()



print("The shape of the data is:" + str(melb_data.shape))
result = melb_data.isna().sum();

nonZeroColumns = []

for (columnName, count) in result.iteritems():

    if count != 0:

        nonZeroColumns.append(columnName)

print('There are '+str(len(nonZeroColumns))+' columns with non-zero values')

print(nonZeroColumns)

print('These values will be dropped!');
melb_data = melb_data.dropna(axis=0)
featureNames = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'];

X = melb_data[featureNames];

y = melb_data.Price;
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

# define model

model1 = DecisionTreeRegressor(random_state = 1)



# fit model

result = model1.fit(X, y);

print(result)
print("Predicting the 1st 5 values:");

print(X.head());

print("Predictions");

print(model1.predict(X.head()))
print("Let us look at the actual values:");

print(y.head());