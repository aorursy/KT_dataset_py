# Used packages



import numpy as np

import scipy as sc

import matplotlib.pyplot as plt



from sklearn.datasets import make_circles



from IPython.display import display, Math, Latex



import time

from IPython.display import clear_output



from sklearn.metrics import accuracy_score
# Train set and test set are created by using make_circles

n = 500

p = 2



np.random.seed(42)

X, y = make_circles(n_samples=n, factor=0.5, noise=0.06)

y = y[:,np.newaxis]



X_test, y_test = make_circles(n_samples=n, factor=0.5, noise=0.06)

y_test = y_test[:,np.newaxis]



plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c='skyblue')

plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c='salmon')

plt.axis('equal')

plt.show()
class neural_layer():

    

    def __init__(self, n_conn, n_neur, act_f):

        

        self.act_f = act_f

        # Bias term

        self.b = np.random.randn(1, n_neur) * 2 - 1 # To normalize it from -1 to 1

        # Weights

        self.w = np.random.randn(n_conn, n_neur) * 2 - 1
sigm = (lambda x: 1/(1+ np.e**(-x)),

        lambda x: x * (1 - x))





_x = np.linspace(-5, 5, 100)

plt.plot(_x, sigm[0](_x))

plt.show()
# A frunction to create the layers



def create_nn(topology, act_f):

    

    nn =  []

    

    for l, layer in enumerate(topology[:-1]):

        

        nn.append(neural_layer(topology[l], topology[l + 1], act_f=act_f))

        

    return nn
topology = [p, 4, 8, 4, 1]

neural_net = create_nn(topology, sigm)



# Mean square error

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr)**2), 

           lambda Yp, Yr: (Yp - Yr))



def train(neural_net, X, y, l2_cost, learning_rate=0.5, train=True):

    

    out = [(None, X)]

    

    # Forward pass

    for l, layer in enumerate(neural_net):

        Z = out[-1][1] @ neural_net[l].w + neural_net[l].b

        a = neural_net[l].act_f[0](Z)

    

        out.append((Z, a))



    

    if train:

        

        # Backward pass

        deltas = []

        

        for l in reversed(range(0, len(neural_net))):

            

            Z = out[l + 1][0]

            a = out[l + 1][1]

            

            #print(a.shape)

            

            if l == len(neural_net) - 1:

                deltas.insert(0, l2_cost[1](a, y) * neural_net[l].act_f[1](a))

                

            else:      

                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

                

            _W = neural_net[l].w

            

            # Gradient descent

            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate

            neural_net[l].w = neural_net[l].w -  out[l][1].T @ deltas[0] * learning_rate

    

    return out[-1][1]
neural_n = create_nn(topology, sigm)



loss = []

epoch = []



for i in range(500):

    

    pY = train(neural_n, X, y, l2_cost, learning_rate=0.06)

    

    if i % 25 == 0:





        loss.append(l2_cost[0](pY, y))

        epoch.append(i)



        res = 50



        _x0 = np.linspace(-1.5, 1.5, res)

        _x1 = np.linspace(-1.5, 1.5, res)



        _Y = np.zeros((res, res))



        for i0, x0 in enumerate(_x0):

            for i1, x1 in enumerate(_x1):

                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), y, l2_cost, train=False)[0][0]    

            

        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")

        plt.axis("equal")



        plt.scatter(X[y[:,0] == 0, 0], X[y[:,0] == 0, 1], c="skyblue")

        plt.scatter(X[y[:,0] == 1, 0], X[y[:,0] == 1, 1], c="salmon")



        clear_output(wait=True)

        plt.show()

        plt.plot(epoch, loss)

        plt.xlabel('Epoch')

        plt.ylabel('mse')

        plt.show()

        time.sleep(0.5)  
logits = train(neural_n, X_test, y, l2_cost, train=False)

logits[logits >= 0.5] = 1

logits[logits < 0.5] = 0



accuracy_score(logits, y_test) #100% of accuracy
plt.scatter(X_test[logits[:, 0] == 0, 0], X_test[logits[:, 0] == 0, 1], c='skyblue')

plt.scatter(X_test[logits[:, 0] == 1, 0], X_test[logits[:, 0] == 1, 1], c='salmon')

plt.axis('equal')

plt.show()