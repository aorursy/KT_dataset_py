import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

def gradient_descent(x,y):

    m_curr = b_curr = 0

    rate = 0.01

    n = len(x)

    plt.scatter(x,y,color='black',marker='*',linewidth='5')

    for i in range(10000):

        y_predicted = m_curr * x + b_curr

#         print (m_curr,b_curr, i)

        plt.plot(x,y_predicted,color='red')

        md = -(2/n)*sum(x*(y-y_predicted))

        yd = -(2/n)*sum(y-y_predicted)

        m_curr = m_curr - rate * md

        b_curr = b_curr - rate * yd
x = np.array([1,2,3,4,5])

y = np.array([5,7,9,11,13])
gradient_descent(x,y)