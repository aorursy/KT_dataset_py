import random

import matplotlib.pyplot as plt



X = [0]

Y = [0]



for n in range(100000):

    r = random.uniform(0, 100)

    if r < 1.0:

        x = 0

        y = 0.16*Y[n-1]

    elif r < 86.0:

        x = 0.85*X[n-1] + 0.04*Y[n-1]

        y = -0.04*X[n-1] + 0.85*Y[n-1]+1.6

    elif r < 93.0:

        x = 0.2*X[n-1] - 0.26*Y[n-1]

        y = 0.23*X[n-1] + 0.22*Y[n-1] + 1.6

    else:

        x = -0.15*X[n-1] + 0.28*Y[n-1]

        y = 0.26*X[n-1] + 0.24*Y[n-1] + 0.44

    X.append(x);Y.append(y)



plt.figure(figsize = [15,15])

plt.scatter(X,Y,color = 'k',alpha=0.1, s=2)

plt.axes().set_aspect('equal', 'datalim')

plt.axis('off')