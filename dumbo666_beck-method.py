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
df = pd.read_csv('../input/train.csv')

df.head(n=10)
df['ug'] = df['u'] - df['g']

df['gr'] = df['g'] - df['r']

df['ri'] = df['r'] - df['i']

df['iz'] = df['i'] - df['z']
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
df.head(n=10)
X_train, X_test, y_train, y_test = train_test_split(

    normalize(df[['r', 'ug', 'ri', 'iz', 'gr']]),

    df['redshift'],

    test_size=0.5,

    random_state=42

)
"""

    Previous task, implementing the method of Robert

    Beck.

"""

from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LinearRegression



# Linear fitter for the provided training set

def linfit(x_train, y_train):

    reg = LinearRegression()

    fit = reg.fit(x_train, y_train)

    return reg



# Predict redshift based on a linear model

def predict_redshift(x_train, y_train, y_test):

    reg = linfit(x_train, y_train)

    return reg.predict(y_test.reshape(1,-1))



# Remove outliers

def calc_excluded_indices(x_reg, y_reg, k):

    deltas = []

    linreg = linfit(x_reg, y_reg)

    y_pred = linreg.predict(x_reg)

    delta_y_pred = np.sqrt(np.sum(np.square(y_reg-y_pred))/k)

    for ind in range(y_reg.size):

        err = np.linalg.norm(y_reg[ind] - y_pred[ind])

        if 3*delta_y_pred < err:

            deltas.append(ind)

    return deltas



# Subtract mean and divide by standard deviation to get 0 mean and 1 standard deviation

def normalize_dataframe(data, labels):

    df = data[labels].subtract(data[labels].mean(axis=1), axis=0)

    return df.divide(df.std(axis=1), axis=0)



def beck_method(test, train, labels, k=100):

    prediction = []

    # Normalize dataframes, test and train for appropiate labels

    df = normalize_dataframe(train, labels)

    test_df = normalize_dataframe(test, labels)

    # Fit the training data

    neigh = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')

    neigh.fit(df.values)

    # Do the prediction

    for point in test_df.values:

        indices = neigh.kneighbors([point], return_distance=False)[0,:]

        redshift = predict_redshift(df.values[indices], train['z'].values[indices], point)

        # Calculate excluded indices

        deltas = calc_excluded_indices(df.values[indices], train['z'].values[indices], k)

        #Redo fit

        prev_size = indices.size

        indices = np.delete(indices, deltas)

        if indices.size < prev_size:

            redshift = predict_redshift(df.values[indices], train['z'].values[indices], point)

        prediction.append(redshift)

    return np.array(prediction)
beck_columns = ['r', 'ug', 'gr', 'ri', 'iz'] # 5 dimensions



# Creating training and test sets from data

mask = np.random.rand(len(df)) < 0.6 # make the test set more then 20% of the dataset

# Convert masked data to np array

train = df[~mask]

test = df[mask]
pred = beck_method(test, train, beck_columns, k=166)
y_test = test['z'].values

y_test = y_test.reshape(y_test.size, 1)



from sklearn.metrics import mean_squared_error



# Calculating the MSE

MSE = mean_squared_error(pred, y_test)
MSE