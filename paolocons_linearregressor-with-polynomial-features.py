# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import combinations

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/another-fiat-500-dataset-1538-rows/automobile_dot_it_used_fiat_500_in_Italy_dataset_filtered.csv')

df.head()
#from https://it.wikipedia.org/wiki/Fiat_500_(2007)#Versioni_e_allestimenti

# we can check the power values for 500's engines

# we have 4 possible values in kw 62,5 - 77,2 - 51 - 73,5



def replace_power(power):

    if power == 73 or power == 74:

        return 73

    elif power == 62 or power == 63:

        return 62

    else:

        return power

    

# fix power values

df['engine_power'] = df['engine_power'].apply(lambda power: replace_power(power))

#since 58 Kw does not correspond at any sold engine and we have only one element: remove those rows

df = df[df['engine_power'] != 58]
# encode and standardize



def replace_model(model):

    if model == 'sport':

        return 2

    if model == 'lounge':

        return 1

    if model == 'pop':

        return 0

    

# fix power values

df['model'] = df['model'].apply(lambda model: replace_model(model))
df.head()
sns.pairplot(df)
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)

figure.set_size_inches(16,28)

_ = sns.regplot(df['engine_power'], df['price'], ax=ax1)

_ = sns.regplot(df['age_in_days'], df['price'], ax=ax2)

_ = sns.regplot(df['km'], df['price'], ax=ax3)

_ = sns.regplot(df['previous_owners'], df['price'], ax=ax4)

_ = sns.regplot(df['lat'], df['price'], ax=ax5)

_ = sns.regplot(df['lon'], df['price'], ax=ax6)

# correlation map

df.corr()
features = ['model', 'engine_power', 'age_in_days', 'km', 'previous_owners', 'lat','lon', 'price']



selected_features = ['model', 'engine_power', 'age_in_days', 'km', 'previous_owners', 'lat','lon']



#df.drop(drop_features, axis=1, inplace=True)
# try all combinations to get the best results



for i in range(1, len(selected_features)+1):

    combs = combinations(selected_features,i)

    for comb in combs:

        df2 = df[list(comb)]

        X = df2.values

        Y = df.price.values



        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0, shuffle=True)



        print(list(comb))

        for deg in range(1,5):

            poly = PolynomialFeatures(degree=deg)

            X_train_poly = poly.fit_transform(X_train)

            X_test_poly = poly.transform(X_test)

            regressor = LinearRegression()

            regressor.fit(X_train_poly,y_train)

            y_test_pred = regressor.predict(X_test_poly)

            print('deg=%d, train score= %.4f, test score= %.4f' %(deg,regressor.score(X_train_poly, y_train),regressor.score(X_test_poly, y_test)))

    
# Use best feature list and train the model again



best_feature_list = ['model', 'age_in_days', 'km', 'previous_owners', 'lat', 'lon']



X = df[best_feature_list].values

Y = df.price.values



X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0, shuffle=True)



poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

regressor = LinearRegression()

regressor.fit(X_train_poly,y_train)

y_train_pred = regressor.predict(X_train_poly)

y_test_pred = regressor.predict(X_test_poly)

print('deg=%d, train score= %.4f, test score= %.4f' %(deg,regressor.score(X_train_poly, y_train),regressor.score(X_test_poly, y_test)))

    
# predict the first 20 values of test set and compare with the real value

for i in range(0, len(y_test[:20])):

    print(str(y_test[i]) + ",  " +  str(y_test_pred[i]))
# plot difference between y_train and y_train_pred

plt.scatter(y_train, y_train_pred)

plt.scatter(y_train, y_train);
# plot difference between y_test and y_test_pred

plt.scatter(y_test, y_test_pred)

plt.scatter(y_test, y_test);