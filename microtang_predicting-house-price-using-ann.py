# Importing the libraries

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
# Importing the dataset

df = pd.read_csv('../input/kc_house_data.csv')

df.head()
# Getting the dependent variables and independent variables

df['date'] = df['date'].apply(lambda x: int(x[:8]))

X = df[['date', 'bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']].values

y = df['price'].values
# Splitting the training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# define base model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(19, input_dim=19, kernel_initializer='normal', activation='relu'))

    model.add(Dense(output_dim = 19, init = 'uniform', activation = 'relu'))

    model.add(Dense(output_dim = 19, init = 'uniform', activation = 'relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

# Fitting to the training set

estimator = KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=10, verbose=False)

estimator.fit(X_train, y_train)

# Predicting the results

prediction = estimator.predict(X_test)
# Visualization the results and evaluation

n = 12

length = len(prediction)

sns.set_style('darkgrid', {'axis.facecolor':'black'})

f, axes = plt.subplots(n, 1, figsize=(25,30))

times = 0

for i in range(n):

    if i == 0:

        plt.sca(axes[0])

        plt.plot(y_test[:round(length/n)], color = 'red', label = 'Real Price')

        plt.plot(prediction[:round(length/n)], color = 'blue', label = 'Predicted Price')

        plt.title('House Price Predicitionin King County', fontsize=30)

        plt.ylabel('Price', fontsize=20)

        plt.legend(loc=1, prop={'size': 10})

    else:

        if i == n-1:

            plt.sca(axes[n-1])

            plt.plot(y_test[round(length/n*(n-1)):], color = 'red', label = 'Real Price')

            plt.plot(prediction[round(length/n*(n-1)):], color = 'blue', label = 'Predicted Price')

            plt.ylabel('Price', fontsize=20)

            plt.legend(loc=1, prop={'size': 10})

        else:

            plt.sca(axes[i])

            plt.plot(y_test[round(length/n*i):round(length/n*(i+1))], color = 'red', label = 'Real Price')

            plt.plot(prediction[round(length/n*i):round(length/n*(i+1))], color = 'blue', label = 'Predicted Price')

            plt.ylabel('Price', fontsize=20)

            plt.legend(loc=1, prop={'size': 10})

plt.show()