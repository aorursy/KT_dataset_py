import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



import tensorflow as tf

import keras



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras import regularizers

from keras.layers import Activation

from keras.callbacks import EarlyStopping



from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import time

from scipy import stats
vol_matrix = [[0.3024, 0.1354, 0.0722, 0.1367, 0.1641],

              [0.1354, 0.2270, 0.0613, 0.1264, 0.1610],

              [0.0722, 0.0613, 0.0717, 0.0884, 0.0699],

              [0.1367, 0.1264, 0.0884, 0.2937, 0.1394],

              [0.1641, 0.1610, 0.0699, 0.1394, 0.2535]]

S0 = 100

r = 0.05

N = 48

T = 1

m = 48

n = 2000

t = np.linspace(0, 1, m + 1)
def standardBrownianMotion(n, m): 

    '''

    Simulate a 1D standard Brownian motion on [0, ‘T‘] with ‘n‘ Brownian increments .

    '''

   

    dxs = np.random.normal(

                loc=0, scale=np.sqrt(1 / n), size=(m, n+1))

    dxs[:,0] = 0.0

    xs = np.cumsum(dxs,  axis = 1) 

    return xs



def payoff_function(X_t):

    if X_t <= 75:

        return 15

    elif X_t <= 90 and X_t > 75:

        return 90 - X_t

    elif X_t <= 110 and X_t > 90:

        return 0

    elif X_t <= 125 and X_t > 110:

        return X_t - 110

    else:

        return 15



def simulate_basket_average(n_stocks, vol_matrix, n_timesteps, n_paths):

    stocks = np.zeros((n_paths, n_timesteps + 1))

    for k in range(n_paths):

      S = np.zeros((n_stocks,n_timesteps + 1))

      W = standardBrownianMotion(n_timesteps, n_stocks)

      for i in range(n_stocks):

        for t_ind in range(m+1):

          sigma = vol_matrix[i]

          S[i,t_ind] = S0 * np.exp(r * t[t_ind]) * np.exp( np.sum( sigma * W[:,t_ind]  - 0.5 * np.array(sigma)**2 * t[t_ind]))

      stocks[k,:] = np.mean(S,axis=0)

    return stocks





def simulate_basket(n_stocks, vol_matrix, n_timesteps, n_paths):

    stocks = np.zeros((n_paths, n_timesteps + 1, n_stocks))

    for k in range(n_paths):

      W = standardBrownianMotion(n_timesteps, n_stocks)

      for i in range(n_stocks):

        for t_ind in range(m+1):

          sigma = vol_matrix[i]

          stocks[k,t_ind,i] = S0 * np.exp(r * t[t_ind]) * np.exp( np.sum( sigma * W[:,t_ind]  - 0.5 * np.array(sigma)**2 * t[t_ind]))

    return stocks
n_simulations = 3

option_prices = np.zeros((n_simulations))

n_stocks = 5

k_nodes = 2 ** np.array(range(6))

n_evaluations = 4000





for sim in range(n_simulations):

    q_hat = {}

    q_hat[m] = lambda x: 0

    for t_ind in range(m-1, -1, -1):

      print("Timestep: " + str(t_ind))

      start = time.time()

      stocks = simulate_basket(n_stocks, vol_matrix, n_timesteps = m, n_paths = n)

      Y = np.zeros((n))

      for i in range(n):

        Y[i] = max( np.exp(- r * t[t_ind + 1]) * payoff_function( np.mean(stocks[i,t_ind + 1,:])), q_hat[t_ind + 1] (stocks[i,t_ind+1,:].reshape((1,5)))) 

      X = stocks[:, t_ind,:]

      X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.5)

      best_loss = 1e6

      best_k = 1

      hist = {}

      for k in k_nodes:

        print("Training with nodes: "+str(k) )

        model = Sequential()

        model.add(Dense(k, activation = 'sigmoid',

                            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None), input_dim = 5))

        model.add(Dense(1, activation = 'linear',

                        kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=(1/k)**0.5, seed=None)))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

        hist[k] = model.fit(X_train, y_train, epochs=100, validation_split = 0, verbose = 0, callbacks = [EarlyStopping(min_delta = 0.25, patience = 5)],

                        validation_data = (X_test, y_test))

        mse, _ = model.evaluate(X_test,y_test, verbose = 0) 

        if mse < best_loss:

          best_loss = mse

          q_hat[t_ind] = model.predict

          best_k = k

     

      print("Train:" + str(hist[best_k].history["loss"][-1]))

      print("Validation:" + str(best_loss))      

      plt.figure(figsize = (15,7))

      plt.plot(hist[best_k].history["loss"], color = "blue", label = "Training")

      plt.plot(hist[best_k].history["val_loss"], color = "red", label = "Validation")

      plt.show()

      print("Time ellapsed: " + str(round(time.time() - start)))



    #Zweiter Schritt der Preisberechnung: Simuliere 4000 Pfade, werte die optimale Stoppregel q aus und nimm den Mittelwert

    stocks = simulate_basket_average(n_stocks, vol_matrix, n_timesteps = m, n_paths = n_evaluations)

    realized_prices = np.zeros((n_evaluations))

    for j in range(n_evaluations):

      for t_ind in range(m+1):

        disc_payoff = np.exp(- r * t[t_ind]) * payoff_function(np.mean(stocks[j,t_ind,:]))

        q_t = q_hat[t_ind](stocks[j,t_ind,:].reshape((1,5)))

        if disc_payoff >= q_t:

          realized_prices[j] = disc_payoff

          break



    plt.hist(realized_prices)

    print("\n\nOPTION PRICES:" + str(np.mean(realized_prices)) + "," + str(q_hat[0](np.array(S0))))

    option_prices[sim] = np.mean(realized_prices[:,0])

    plt.boxplot(option_prices[:sim])

    plt.show()




