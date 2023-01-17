import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use(['ggplot'])
X = 2 * np.random.rand(100,1)

y = 4 +3 * X+np.random.randn(100,1)


plt.plot(X,y,'b.')

plt.xlabel("$x$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

_ =plt.axis([0,2,0,15])
X_b = np.c_[np.ones((100,1)),X]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)
X_new = np.array([[0],[2]])

X_new_b = np.c_[np.ones((2,1)),X_new]

y_predict = X_new_b.dot(theta_best)

y_predict
plt.plot(X_new,y_predict,'r-')

plt.plot(X,y,'b.')

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.axis([0,2,0,15])


def  cal_cost(theta,X,y):

    '''

    

    Calculates the cost for given X and Y. The following shows and example of a single dimensional X

    theta = Vector of thetas 

    X     = Row of X's np.zeros((2,j))

    y     = Actual y's np.zeros((2,1))

    

    where:

        j is the no of features

    '''

    

    m = len(y)

    

    predictions = X.dot(theta)

    cost = (1/2*m) * np.sum(np.square(predictions-y))

    return cost

def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):

    '''

    X    = Matrix of X with added bias units

    y    = Vector of Y

    theta=Vector of thetas np.random.randn(j,1)

    learning_rate 

    iterations = no of iterations

    

    Returns the final theta vector and array of cost history over no of iterations

    '''

    m = len(y)

    cost_history = np.zeros(iterations)

    theta_history = np.zeros((iterations,2))

    for it in range(iterations):

        

        prediction = np.dot(X,theta)

        

        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))

        theta_history[it,:] =theta.T

        cost_history[it]  = cal_cost(theta,X,y)

        

    return theta, cost_history, theta_history
lr =0.01

n_iter = 1000



theta = np.random.randn(2,1)



X_b = np.c_[np.ones((len(X),1)),X]

theta,cost_history,theta_history = gradient_descent(X_b,y,theta,lr,n_iter)





print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))

print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
fig,ax = plt.subplots(figsize=(12,8))



ax.set_ylabel('J(Theta)')

ax.set_xlabel('Iterations')

_=ax.plot(range(n_iter),cost_history,'b.')


fig,ax = plt.subplots(figsize=(10,8))

_=ax.plot(range(200),cost_history[:200],'b.')
def plot_GD(n_iter,lr,ax,ax1=None):

     """

     n_iter = no of iterations

     lr = Learning Rate

     ax = Axis to plot the Gradient Descent

     ax1 = Axis to plot cost_history vs Iterations plot



     """

     _ = ax.plot(X,y,'b.')

     theta = np.random.randn(2,1)



     tr =0.1

     cost_history = np.zeros(n_iter)

     for i in range(n_iter):

        pred_prev = X_b.dot(theta)

        theta,h,_ = gradient_descent(X_b,y,theta,lr,1)

        pred = X_b.dot(theta)



        cost_history[i] = h[0]



        if ((i % 25 == 0) ):

            _ = ax.plot(X,pred,'r-',alpha=tr)

            if tr < 0.8:

                tr = tr+0.2

     if not ax1== None:

        _ = ax1.plot(range(n_iter),cost_history,'b.')  
fig = plt.figure(figsize=(30,25))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



it_lr =[(2000,0.001),(500,0.01),(200,0.05),(100,0.1)]

count =0

for n_iter, lr in it_lr:

    count += 1

    

    ax = fig.add_subplot(4, 2, count)

    count += 1

   

    ax1 = fig.add_subplot(4,2,count)

    

    ax.set_title("lr:{}".format(lr))

    ax1.set_title("Iterations:{}".format(n_iter))

    plot_GD(n_iter,lr,ax,ax1)
_,ax = plt.subplots(figsize=(14,10))

plot_GD(100,0.1,ax)
def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):

    '''

    X    = Matrix of X with added bias units

    y    = Vector of Y

    theta=Vector of thetas np.random.randn(j,1)

    learning_rate 

    iterations = no of iterations

    

    Returns the final theta vector and array of cost history over no of iterations

    '''

    m = len(y)

    cost_history = np.zeros(iterations)

    

    

    for it in range(iterations):

        cost =0.0

        for i in range(m):

            rand_ind = np.random.randint(0,m)

            X_i = X[rand_ind,:].reshape(1,X.shape[1])

            y_i = y[rand_ind].reshape(1,1)

            prediction = np.dot(X_i,theta)



            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))

            cost += cal_cost(theta,X_i,y_i)

        cost_history[it]  = cost

        

    return theta, cost_history
lr =0.5

n_iter = 50



theta = np.random.randn(2,1)



X_b = np.c_[np.ones((len(X),1)),X]

theta,cost_history = stocashtic_gradient_descent(X_b,y,theta,lr,n_iter)





print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))

print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
fig,ax = plt.subplots(figsize=(10,8))



ax.set_ylabel('{J(Theta)}',rotation=0)

ax.set_xlabel('{Iterations}')

theta = np.random.randn(2,1)



_=ax.plot(range(n_iter),cost_history,'b.')
def minibatch_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10,batch_size =20):

    '''

    X    = Matrix of X without added bias units

    y    = Vector of Y

    theta=Vector of thetas np.random.randn(j,1)

    learning_rate 

    iterations = no of iterations

    

    Returns the final theta vector and array of cost history over no of iterations

    '''

    m = len(y)

    cost_history = np.zeros(iterations)

    n_batches = int(m/batch_size)

    

    for it in range(iterations):

        cost =0.0

        indices = np.random.permutation(m)

        X = X[indices]

        y = y[indices]

        for i in range(0,m,batch_size):

            X_i = X[i:i+batch_size]

            y_i = y[i:i+batch_size]

            

            X_i = np.c_[np.ones(len(X_i)),X_i]

           

            prediction = np.dot(X_i,theta)



            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))

            cost += cal_cost(theta,X_i,y_i)

        cost_history[it]  = cost

        

    return theta, cost_history
lr =0.1

n_iter = 200



theta = np.random.randn(2,1)





theta,cost_history = minibatch_gradient_descent(X,y,theta,lr,n_iter)





print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))

print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
fig,ax = plt.subplots(figsize=(10,8))



ax.set_ylabel('{J(Theta)}',rotation=0)

ax.set_xlabel('{Iterations}')

theta = np.random.randn(2,1)



_=ax.plot(range(n_iter),cost_history,'b.')