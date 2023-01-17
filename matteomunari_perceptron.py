import numpy as np



# X is a m*n matrix containing m n-dimension example, one in each row

# y is the vector of the target values

# eta is the learning rate



def perceptron(X, y, eta):

    m,n = X.shape

    w = np.random.rand(n,1)

    pred_y = np.sign(np.dot(X,w))

    i = 0

    

    #the algorithm loops until the predicted values are equal to y

    while not np.array_equal(pred_y,y):

        x = X[i,:].reshape(-1,1)

        t = y[i]

        o = np.sign(np.dot(w.T,x))

        if o != t:

            w = w + eta*(t - o)*x

            pred_y = np.sign(np.dot(X,w))

        

        #the index i is increased so in the next round a different example x is selected, when it reaches m it restarts from 0

        i = (i + 1) % m

    return w
import matplotlib.pyplot as plt

n_samples = 50



# arbitrary hyperplane used to classify the examples

real_w = np.array([[-3],[10],[1]])

np.random.seed(42)

x = np.c_[np.ones(n_samples), (np.random.rand(n_samples,2) * 2 - 1)*10]



# in order to make easy the plotting the target value is the last column of x

x = np.c_[x, np.sign(np.dot(x,real_w))]



plt.plot(x[(x[:,3]==1)][:,1],x[(x[:,3]==1)][:,2],'.g',label='positive examples')

plt.plot(x[(x[:,3]==-1)][:,1],x[(x[:,3]==-1)][:,2],'.r',label='negative examples')

plt.plot(np.linspace(-10,10,100),-(np.linspace(-10,10,100)*real_w[1,0] + real_w[0,0])/real_w[2,0], 'y',label='original hyperplane')

plt.ylim(-20,20)

plt.legend()
eta = 0.5

pred_w = perceptron(x[:,:-1], x[:,-1].reshape(-1,1), eta)

pred_w
pred = np.c_[x[:,:-1], np.sign(np.dot(x[:,:-1],pred_w))]



plt.plot(pred[(pred[:,3]==1)][:,1],pred[(pred[:,3]==1)][:,2],'.g',label='positive examples')

plt.plot(pred[(pred[:,3]==-1)][:,1],pred[(pred[:,3]==-1)][:,2],'.r',label='negative examples')

plt.plot(np.linspace(-10,10,100),-(np.linspace(-10,10,100)*pred_w[1,0] + pred_w[0,0])/pred_w[2,0], '-b', label='predicted hyperplane')

plt.plot(np.linspace(-10,10,100),-(np.linspace(-10,10,100)*real_w[1,0] + real_w[0,0])/real_w[2,0], '-y', label='original hyperplane')

plt.ylim(-20,20)

plt.legend()