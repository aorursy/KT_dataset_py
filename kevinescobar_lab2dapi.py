import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor as mlp_r

from sklearn.neural_network import MLPClassifier as mlp_c

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn import datasets
Heart_data = pd.read_csv("../input/heart.csv")

Heart_data.head()
X = (Heart_data.iloc[:,:-1]).as_matrix()

y = (Heart_data.iloc[:,2]).as_matrix()
X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

y = y.reshape((-1,1))/3
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)
regressor = mlp_r(

    hidden_layer_sizes=(100,100),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',

    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
regressor.fit(X_train, y_train)
y_hats = regressor.predict(X_test)
plt.scatter(y_test, y_hats, c='k')



plt.plot([0.1, 0.9], [0.1, 0.9], 'r')

plt.xlabel('Real')

plt.ylabel('Estimada')
from sklearn.metrics import mean_squared_error



capa_1 = [5, 7, 9, 11, 13, 17,  19, 23, 29, 31]

capa_2 = [1, 5, 7, 9, 11, 13, 17,  19, 23, 29]



mse_m = np.zeros((len(capa_1),len(capa_1)))

mse_std = np.zeros((len(capa_1),len(capa_1)))



for j, n_1 in enumerate(capa_1):

    for k, n_2 in enumerate(capa_2):

        mse_temp = []

    

        for i in range(10):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            regressor = mlp_r(hidden_layer_sizes=(n_1,n_2),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',

            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            regressor.fit(X_train, y_train)

            y_hats = regressor.predict(X_test)

            mse_temp.append(mean_squared_error(y_test, y_hats))

            

        mse_m[j, k] = np.mean(mse_temp)

        mse_std[j, k] = np.std(mse_temp)

plt.imshow(mse_m)

plt.colorbar()
plt.imshow(mse_std)

plt.colorbar()
mmntm=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

mse_m = np.zeros(len(mmntm))

for m,n_3 in enumerate(mmntm):

    for i in range(21):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        regressor = mlp_r(hidden_layer_sizes=(29,31),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',

        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

        random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=n_3, nesterovs_momentum=True,

        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        regressor.fit(X_train, y_train)

        y_hats = regressor.predict(X_test)

        mse_temp.append(mean_squared_error(y_test, y_hats))

    mse_m[m] = np.mean(mse_temp)
plt.plot(mmntm, mse_m, '.')

plt.xlabel('Momentum')

plt.ylabel('Error cuadratico medio')
X = (Heart_data.iloc[:,:-1]).as_matrix()

y = (Heart_data.iloc[:,2]).as_matrix()
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)
clf = mlp_c(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(1000, 500), random_state=1)



clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)
cm1 = confusion_matrix(y_test, y_hat)

cm2 = confusion_matrix(y_test, y_hat)

cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]

cm1
cm2
capa_1=[100, 400, 800, 900, 1000]

capa_2=[20, 100, 500, 700, 800]

cero=np.zeros((len(capa_1),len(capa_2)))

uno=np.zeros((len(capa_1),len(capa_2)))

dos=np.zeros((len(capa_1),len(capa_2)))

tres=np.zeros((len(capa_1),len(capa_2)))

suma=np.zeros((len(capa_1),len(capa_2)))

for j, n_1 in enumerate(capa_1):

    for k, n_2 in enumerate(capa_2):

        X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.33, random_state=42)



        clf = mlp_c(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(n_1, n_2), random_state=1)



        clf.fit(X_train, y_train)

        y_hat = clf.predict(X_test)

        cm2 = confusion_matrix(y_test, y_hat)

        cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]

        suma[j,k]=cm2[0,0]+cm2[1,1]+cm2[2,2]+cm2[3,3]

        cero[j,k]=cm2[0,0]

        uno[j,k]=cm2[1,1]

        dos[j,k]=cm2[2,2]

        tres[j,k]=cm2[3,3]

            

plt.imshow(cero)

plt.colorbar()
plt.imshow(uno)

plt.colorbar()
plt.imshow(dos)

plt.colorbar()
plt.imshow(tres)

plt.colorbar()
plt.imshow(suma/4)

plt.colorbar()