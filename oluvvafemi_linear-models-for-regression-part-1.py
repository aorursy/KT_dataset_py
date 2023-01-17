#Simulation in Python



import numpy as np



np.random.seed(1)



def simulateLinearRegressionData(m, n):

    """

  m - number of samples

  n - number of features

  

  returns:

    matrix X (m*n-1)

    vector y(m)

    vector beta(n)

    

  note: for convenience 1 has been added to the last column of X

    """

    X = np.zeros((m,n-1))

    beta = np.zeros(n)

    for i in range(n-1):

      mu = np.random.randint(0, 50)

      sigma = np.random.normal(3, 0.1)

      X[:,i] = np.random.normal(mu, sigma, size=m)

      beta[i] =  np.random.normal(mu, sigma)

      

    X = np.hstack((X, np.ones((m,1))))

    beta[-1] = np.random.normal()

    beta =  np.reshape(beta, (n, -1))

    y = np.dot(X, beta)

    

    return X, y, beta

    

X, y, true_beta = simulateLinearRegressionData(100000, 10)

print(true_beta)

from numpy.linalg import inv



def estimateBetaLR(X, y):

  return np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))

estimated_beta = estimateBetaLR(X, y)

print("estimated beta,   true beta\n", np.hstack((estimated_beta, true_beta)))

np.random.seed(2)



def calc_rss(beta, x, y):

  return float((y-x.dot(beta)).T.dot(y-x.dot(beta)))

  

  

def oracle(beta, x, y):

  return -2 * np.dot(x.T, (y-np.dot(x,beta)))

  # return (-2*(x.T)).dot(y-x.dot(_beta))

  

  

def gd(x, y, maxit, step_size):

  beta = np.random.normal(size=(x.shape[1], 1))

  rss_history = []

  for i in range(maxit):

    rss_history.append(calc_rss(beta, x, y))

    step_direction = -oracle(beta, x, y)

    beta = beta + step_size*step_direction

    if i==maxit-1:

      print(beta)

    

  return beta, rss_history

    

gd_beta, rss_history = gd(X, y,50, 1e-10)



print("gd-estimated beta,   true beta\n", np.hstack((gd_beta, true_beta)))



# library(reticulate)

# py_install("matplotlib")

# py_install("--user loess") 

import matplotlib.pyplot as plt

plt.style.use("seaborn")

plt.plot(rss_history)

def calc_tss(y):

  return np.dot((y-np.mean(y)).T, ((y-np.mean(y))))

  

def rse_r2(_beta,x, y):

  rss = calc_rss(_beta,x, y )

  rse = np.sqrt(rss/(len(y)-2))

  tss = calc_tss(y)

  r2 = 1-(rss/tss)

  return rse, r2

  

rse, r2 = rse_r2(gd_beta, X, y)

print(f"RSE = {rse}")

print(f"R2 = {float(r2)}")

def plot_residuals(x, y, _beta):

  y_hat = x.dot(_beta)

  e = y - y_hat

  if x.shape[1] > 1:

    plt.scatter(y_hat, e)

    plt.show()

  else:

    plt.scatter(x[:,0], e)

    

plt.figure(figsize=(20,10))

plot_residuals(X, y, gd_beta)
