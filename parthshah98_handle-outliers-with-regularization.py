# you should not import any other packages

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import numpy as np

from sklearn.linear_model import SGDRegressor
import numpy as np

import scipy as sp

import scipy.optimize



def angles_in_ellipse(num,a,b):

    assert(num > 0)

    assert(a < b)

    angles = 2 * np.pi * np.arange(num) / num

    if a != b:

        e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5

        tot_size = sp.special.ellipeinc(2.0 * np.pi, e)

        arc_size = tot_size / num

        arcs = np.arange(num) * arc_size

        res = sp.optimize.root(

            lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)

        angles = res.x

    return angles
a = 2

b = 9

n = 50



phi = angles_in_ellipse(n, a, b)

e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5

arcs = sp.special.ellipeinc(phi, e)



fig = plt.figure()

ax = fig.gca()

ax.axes.set_aspect('equal')

ax.scatter(b * np.sin(phi), a * np.cos(phi))

plt.show()
X= b * np.sin(phi)

Y= a * np.cos(phi)

target1 = [1] * len(X)

target0 = [0] * len(X)


alphas=[0.0001, 1, 100]

outlier = [(0,2),(21, 13), (-23, -15), (22,14), (23, 14)]



plt.figure(figsize=(10,10))





for alpha in alphas:

    

    # Reassign X and Y for each alpha

    X= b * np.sin(phi)

    Y= a * np.cos(phi)

    plt.figure(figsize=(20,5))

    

    for j,olt in enumerate(outlier):

        plt.subplot(1, 5, j+1)

        # Add outlier

        X = (np.append(X, olt[0]))

        Y = (np.append(Y, olt[1]))

        

        X1 = np.array([[i] for i in X])

        Y1 = np.array([[i] for i in Y])

        

        model = SGDRegressor(alpha=alpha, eta0=0.001, learning_rate='constant',random_state=0)

        model.fit(X1,Y1)

        pred = model.predict(X1)

        

        #plot

        plt.scatter(X,Y)

        plt.plot(X1, pred)

    plt.show()