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

df = pd.read_csv('../input/nyc-rolling-sales.csv')

df.head()
# Data processing

df = df.drop(df[df['SALE PRICE']==' -  '].index)

df= df.reset_index()

df['SALE PRICE'] = df['SALE PRICE'].apply(lambda x: int(x))

for i in range(len(df)):

    if df.at[i,'ADDRESS'].find(',')!=-1:

        df.at[i,'ADDRESS'] = df.at[i,'ADDRESS'][:min(df.at[i,'ADDRESS'].find(','),len(df.at[i,'ADDRESS']))]

    df.at[i,'ADDRESS'] = df.at[i,'ADDRESS'][df.at[i,'ADDRESS'].find(' '):]

for i in range(len(df)):

    if df.at[i,'LAND SQUARE FEET'] == ' -  ':

        df.at[i,'LAND SQUARE FEET'] = int(0)

    else:

        df.at[i,'LAND SQUARE FEET'] = int(df.at[i,'LAND SQUARE FEET'])

for i in range(len(df)):

    if df.at[i,'GROSS SQUARE FEET'] == ' -  ':

        df.at[i,'GROSS SQUARE FEET'] = int(0)

    else:

        df.at[i,'GROSS SQUARE FEET'] = int(df.at[i,'GROSS SQUARE FEET'])

df['SALE DATE'] = df['SALE DATE'].apply(lambda x: int(x[:4]+x[5:7]+x[8:10]))

df['SALE DATE'] = df['SALE DATE'].astype(int)

df = df[df['SALE PRICE'] != 0]
# Getting the dependent variables and independent variables

X = df[['BOROUGH','NEIGHBORHOOD','BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','BLOCK','LOT','BUILDING CLASS AT PRESENT','ADDRESS','ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','TAX CLASS AT TIME OF SALE','BUILDING CLASS AT TIME OF SALE','SALE DATE']].values

y = df['SALE PRICE'].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

labelencoder_X_3 = LabelEncoder()

X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

labelencoder_X_6 = LabelEncoder()

X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])

labelencoder_X_7 = LabelEncoder()

X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])

labelencoder_X_16 = LabelEncoder()

X[:, 16] = labelencoder_X_16.fit_transform(X[:, 16])
# Splitting the training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = 0)

# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# define base model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(18, input_dim=18, kernel_initializer='normal', activation='relu'))

    model.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu'))

    model.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu'))

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

f, axes = plt.subplots(n, 1, figsize=(20,100))

times = 0

for i in range(n):

    if i == 0:

        plt.sca(axes[0])

        plt.plot(y_test[:round(length/n)], color = 'red', label = 'Real Price')

        plt.plot(prediction[:round(length/n)], color = 'blue', label = 'Predicted Price')

        plt.title('NYC Property Price Prediction', fontsize=30)

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