# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/input/learn-together/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the files train.csv and test.csv into variables df_train_data, and df_test_data.

train_data = pd.read_csv('../input/learn-together/train.csv')

test_data = pd.read_csv('../input/learn-together/test.csv')
train_data.head()
test_data.head()
print("train dataset shape "+ str(train_data.shape))

print("test dataset shape "+ str(test_data.shape))
train_data.info()
test_data.info()
#train_data.describe()
#test_data.describe()
X = train_data.copy()

X = X.drop(columns=['Cover_Type'])

y = train_data[['Cover_Type']]
X.columns
colorado_features = ['Id', 'Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon']

X = X[colorado_features]
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

colorado_model = DecisionTreeRegressor(random_state=1)



# Fit model

colorado_model.fit(X, y)
print("Making 5 predictions")

print(X.head())

print("The predictions are")

print(colorado_model.predict(X.head()))
print("OK")

print(X,"The predictions are    ", colorado_model.predict(X))

print()
def evaluate_metric_score(y_true, y_pred):

    if y_true.shape[0] != y_pred.shape[0]:

        raise Exception("Sizes do not match")

        return 0

    else:

        size = y_true.shape[0]

        matches = 0

        y_true_array = np.array(list(y_true))

        y_pred_array = np.array(list(y_pred))

        for i in range(0, size):

            if y_true_array[i]==y_pred_array[i]:

                matches = matches + 1

        return mathces/size
output = pd.DataFrame({'Id': X.index,

                       'Cover_Type': test_preds})

output.to_csv('submission1.csv', index=False)

output.head()