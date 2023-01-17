import numpy as np

from pprint import pprint
def gradE(w1,w2):

    g1 = -0.8182 + 1/2*(2*w1+2*0.8182*w2)

    g2 = -0.354 + 1/2*(2*0.8182*w1+2*w2)

    return [g1,g2]
def desc(rate,w1,w2):

    vector = gradE(w1,w2)

    return [component * -rate for component in vector]
w0 = [0,0]

w = [0,0]



rate = 0.3



w0_rate03_vect = [0]

w1_rate03_vect = [0]



for i in range(200):

    descent = desc(rate, w[0], w[1])

    w[0] = w[0] + descent[0]

    w[1] = w[1] + descent[1]

    w0_rate03_vect.append(w[0])

    w1_rate03_vect.append(w[1])    

    

pprint(w)
w0 = [0,0]

w = [0,0]



rate = 1



w0_rate1_vect = [0]

w1_rate1_vect = [0]



for i in range(80):

    descent = desc(rate, w[0], w[1])

    w[0] = w[0] + descent[0]

    w[1] = w[1] + descent[1]

    w0_rate1_vect.append(w[0])

    w1_rate1_vect.append(w[1])

    

pprint(w)
from matplotlib import pyplot as plt



plt.plot(w0_rate03_vect, list(range(201)), label='η=0.3', color='blue')

plt.plot(w1_rate03_vect, list(range(201)), color='blue')

plt.plot(w0_rate1_vect, list(range(81)), label='η=1.0', color='green')

plt.plot(w1_rate1_vect, list(range(81)), color='green')



plt.xlabel("Peso")

plt.ylabel("Num. Iteraciones")

plt.legend(loc='best')



plt.show()