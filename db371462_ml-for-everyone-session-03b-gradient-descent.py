%matplotlib inline
import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns
# np.random.rand gives us uniformly distributed random numbers [0, 1], so multiplying by 10 gives us [0, 10]

X = 10*np.random.rand(50) 

# np.random.randn gives us standard normal random numbers, so multiplying by 2 gives us N(0, 2)

eps = 2*np.random.randn(50)

# our final simulated dataset

y = 8*X+eps
plt.scatter(X, y)

plt.xlabel("x")

plt.ylabel("y")

plt.show()
def loss_function(preds, y):

    sq_residuals = (y-preds)**2

    rss = np.sum(sq_residuals)/len(sq_residuals)

    return rss
def predict_linear_model(b0, b1, x_to_pred):

    preds = b0+b1*x_to_pred

    return preds
def plot_data_and_preds(b0, b1, x_to_pred):

    preds = predict_linear_model(b0, b1, x_to_pred)



    plt.scatter(X, y)

    plt.plot(X, preds, c="red")

    plt.xlabel("x")

    plt.ylabel("y")

    plt.legend(["Regression Line", "Raw Data"])

    plt.show()



    the_loss = loss_function(preds, y)

    print("loss=%s" % the_loss)

    return the_loss
plot_data_and_preds(0, 2, X)
def plot_loss_function(y, X, n_points=5):

    # equally spaced array of 5 values between -20 and 20, like the seq function in R

    beta1s = np.linspace(-20, 20, n_points)

    losses = []

    for beta1 in beta1s:

        print("beta1=%s " % beta1)

        loss = plot_data_and_preds(0, beta1, X)

        losses.append(loss)

    plt.scatter(beta1s, losses)

    plt.xlabel("beta1")

    plt.ylabel("J")

    plt.show()
#plot_loss_function(y, X)

plot_loss_function(y, X, 20)
def gradient_b1(b0, b1, y, X):

    grad = np.sum(-2.0*X*(y-b0-b1*X))/len(X)

    return grad
# the gradient at 20 is positive...

gradient_b1(0, 20, y, X)

# and at -10 is negative...

gradient_b1(0, -10, y, X)
def gradient_descent(b0_start, b1_start, y, X, learning_rate=0.01, n_steps=25):

    b1 = b1_start

    print("b1=%s" % b1)

    for i in range(n_steps):

        grad = gradient_b1(b0_start, b1, y, X)

        #print "gradient=%s" % grad

        b1 = b1-learning_rate*grad

        print("b1=%s" % b1)
# play with different learning rates: 0.00001, 0.1, 10

gradient_descent(0, 15, y, X, learning_rate=0.01)
# things can go terribly wrong and diverge if the learning rate is too high

gradient_descent(0, 20, y, X, learning_rate=10)
# a tiny learning rate with converge slowly

gradient_descent(0, 20, y, X, learning_rate=0.00001, n_steps=100)