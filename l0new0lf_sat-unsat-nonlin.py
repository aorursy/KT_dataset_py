import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



def sigmoid(x):

    return 1/(1 + np.exp(-(x)))



def relu(x):

    import copy

    

    x_deep = copy.deepcopy(x) 

    neg_ele_idxs = np.where(x<0)

    x_deep[neg_ele_idxs] = 0

    return x_deep



# gen data

xs = np.linspace(-50,50,100)



# plot

fig, axarr = plt.subplots(1, 2)

fig.set_size_inches(10,4)





axarr[0].plot(xs, sigmoid(xs))

axarr[0].set_title("Saturating sigmoid")



axarr[1].plot(xs, relu(xs))

axarr[1].set_title("Non-Saturating ReLU")



plt.show()
xs = np.random.normal(0, 0.01, 99999)



from datetime import datetime

np.random.seed(123)



beg = datetime.now()

sigmoid(xs)

end = datetime.now()

print('Time for sat func: ', end-beg)



beg = datetime.now()

sigmoid(xs)

end = datetime.now()

print('Time for non-sat func: ', end-beg)
