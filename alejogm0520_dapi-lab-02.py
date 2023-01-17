import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor as mlp_r

from sklearn.neural_network import MLPClassifier as mlp_c

from sklearn.metrics import confusion_matrix



import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn import datasets

wine_data = pd.read_csv("../input/winequality-red.csv")

wine_data.head()
X = (wine_data.iloc[:,:-1]).as_matrix()

y = (wine_data.iloc[:,-1]).as_matrix()

X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

y = y.reshape((-1,1))/10

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)
regressor = mlp_r(

    hidden_layer_sizes=(100,),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',

    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
regressor.fit(X_train, y_train)
y_hats = regressor.predict(X_test)

plt.scatter(y_test, y_hats, c='k')



plt.plot([0.2, 0.9], [0.2, 0.9], 'r')

plt.xlabel('Real')

plt.ylabel('Estimada')
pca = PCA(n_components=3)



X_pca = pca.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(

    X_pca, np.ravel(y), test_size=0.33, random_state=42)



regressor2 = mlp_r(

    hidden_layer_sizes=(100,),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',

    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



regressor2.fit(X_train, y_train)



y_hat2 = regressor2.predict(X_test)
plt.scatter(y_test, y_hat2, c='k')



plt.plot([0.2, 0.8], [0.2, 0.8], 'r')

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

            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33)

            

            regresor_temp = mlp_r(hidden_layer_sizes=(n_1, n_2, ),  activation='tanh')

            regresor_temp.fit(X_train, y_train)

            y_pred = regressor2.predict(X_test)

            mse_temp.append(mean_squared_error(y_test, y_pred))

        mse_m[j, k] = np.mean(mse_temp)

        mse_std[j, k] = np.std(mse_temp)

plt.imshow(mse_m)

plt.colorbar()
plt.imshow(mse_std)

plt.colorbar()
iris = datasets.load_iris()

X = iris.data 

y = iris.target
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)

clf = mlp_c(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(100, 2), random_state=1)



clf.fit(X_train, y_train)     
y_hat = clf.predict(X_test)
cm1 = confusion_matrix(y_test, y_hat)

cm2 = confusion_matrix(y_test, y_hat)

cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]



cm1
cm2