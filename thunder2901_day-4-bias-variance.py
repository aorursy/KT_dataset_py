import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

curve = pd.read_csv("../input/curve.csv")

curve.head()
def fit_poly(degree):

    p = np.polyfit(curve.x, curve.y, deg = degree)

    curve['fit'] = np.polyval(p, curve.x)

    sns.regplot(curve.x, curve.y, fit_reg = False)

    return plt.plot(curve.x, curve.fit, label='Fitting')
fit_poly(5)

plt.xlabel("x values")

plt.ylabel("y values")
curve.head()
from sklearn.model_selection import train_test_split

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(curve.x, curve.y,test_size=0.2,random_state=10)



rmse = {

    'degree' : [],

    'rmse_train' : [],

    'rmse_test' : []

}





for degree in range(1,15):

    p = np.polyfit(X_train, y_train, deg=degree)

    rmse['degree'].append(degree)

    rmse['rmse_train'].append(metrics.mean_squared_error(y_train, np.polyval(p, X_train)))

    rmse['rmse_test'].append(metrics.mean_squared_error(y_test, np.polyval(p, X_test)))
rmseDf = pd.DataFrame(rmse)
rmseDf
plt.plot(rmseDf.degree,rmseDf.rmse_train,label='RMSE_TRAIN',c='red')

plt.plot(rmseDf.degree,rmseDf.rmse_test,label='RMSE_TEST',c='green')

plt.xlabel("Degree")

plt.ylabel("RMSE")

plt.legend()