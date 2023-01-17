import math
import numpy as np
import matplotlib.pyplot as plt

# We can change this any time we want, to change the number of points we generate.
N = 100

def generateSinPairs(N, valueRange):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    for i in range(N):
#         Set X to a random value between -4pi and 4pi.
        X[i][0] = (np.random.ranf() *(valueRange) - (0.5*valueRange)) * math.pi
        Y[i][0] = math.sin(X[i][0])
        pass
    return (X,Y)

(train_X, train_Y) = generateSinPairs(N, 2)

plt.subplot(111)
plt.scatter(train_X, train_Y, )
plt.show()
# Notice I've added all the complex stuff here,
# but I'm going to set them to zero or one to cancel them out properly
# I'm doing this so I can reuse my code later on.
# Don't worry, I'll be generating the exact function you see above for now.
def generateRadialPairs(N, mu, lambda_val, w):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    max_val = 8
    init_val = - max_val
    step = 2 * max_val / N
    for i in range(N):
        X[i][0] = init_val
        x = X[i][0] - mu
        x_squared = x * x
        lambda_x_squared = lambda_val * x_squared
        Y[i][0] = w * math.exp(-lambda_x_squared)
        init_val += step
        pass
    return (X,Y)

(exp_X, exp_Y) = generateRadialPairs(100, 0, 1, 1)


plt.subplot(111)
plt.plot(exp_X, exp_Y)
plt.show()
plt.subplot(111)
col = [0,0,0.5]
for i in range(5):
    (mu_X, mu_Y) = generateRadialPairs(100, i*2 - 5, 1, 1)
    plt.plot(mu_X, mu_Y, c=col.copy(), label="mu={}".format(i*2-5))
    col[0] += 0.1
    col[2] -= 0.1
    pass
plt.legend()
plt.show()
    
plt.subplot(111)
col = [0,0,0.5]
for i in range(5):
    (mu_X, mu_Y) = generateRadialPairs(100, 0, (i+1)*0.08, 1)
    plt.plot(mu_X, mu_Y, c=col.copy(), label="mu={}, l={}".format(0, (i+1)*.08))
    col[0] += 0.1
    col[2] -= 0.1
    pass
plt.legend(loc="best")
plt.show()
plt.subplot(111)
col = [0,0,0.5]
for i in range(5):
    (mu_X, mu_Y) = generateRadialPairs(100, 0, 0.3, i+0.3)
    plt.plot(mu_X, mu_Y, c=col.copy(), label="mu={}, l={}, w={}".format(0, 0.3, i+0.3))
    col[0] += 0.1
    col[2] -= 0.1
    pass
plt.legend(loc="best")
plt.show()
def generateRBFPairs(max_val, mu, lambda_val, w):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    max_val = 8
    init_val = - max_val
    step = 2 * max_val / N
    for i in range(N):
        sum = 0
        for j in range(mu.shape[0]):
            X[i][0] = init_val
            x = X[i][0] - mu[j]
            x_squared = x * x
            lambda_x_squared = lambda_val[j] * x_squared
            sum += w[j] * math.exp(-lambda_x_squared)
            pass
        Y[i][0] = float(sum)
        init_val += step
        pass
    return (X,Y)

mu = np.array([-6,-3,0,3,6])
lambda_val = np.array([.1,2,.4,.03,.8])
w = np.array([.3,-1,.5,.01,.7])

(rbf_X, rbf_Y) = generateRBFPairs(100, mu, lambda_val, w)

plt.subplot(111)
plt.plot(rbf_X, rbf_Y)
plt.show()

def _predict(x, mu, lambda_val, w):
    x = float(x)
    sum = 0
    for j in range(mu.shape[0]):
        dist = np.linalg.norm(x - mu[j])
        x_squared = dist * dist
        lambda_x_squared = lambda_val[j] * x_squared
        sum += w[j] * math.exp(-lambda_x_squared)
        pass
    return sum
def predict (x, mu, precision_rate, w):
    A = np.zeros((1, mu.shape[0]))
    for i in range(mu.shape[0]):
        #   distance between point x and mu
        dist = np.linalg.norm(x - mu[i])
        #   square distance, multiply by lambda
        exponent = (- precision_rate[i]) * dist * dist
        A[0][i] = math.exp(exponent)
        pass
    return np.dot(A, w)
def get_loss(X, Y, precision_rate, mu, w):
    mse = 0
    count = 0
    for i in range(X.shape[0]):
        predicted = predict(X[i], mu, precision_rate, w)
        err = Y[i] - predicted
        mse += err * err
        count += 1
        pass
    res = mse / count
    return res
#  Lloyd's algorithm: Implmentation.
# Assume feature engineering has been done.
class KMeans:

    def __init__ (self, k, X):
        self.pointMatrix = X
        self.k = int(k)
        pass

    def assign_means(self, num_trials, num_iterations, verbose= True):
        if verbose:
            print("evaluting means for k = {}...".format(self.k))
            plt.subplot(111)
            pass
        best_error = float("inf") # Best error is positive infinity
        best_mu = []
        best_meanSet = []
        pointMatrix = self.pointMatrix
        k = self.k
        col = [0,0,1]

        for trial in range(num_trials):
            self.mu = self.init_mu()
            errors = []

            for i in range(num_iterations):
                #iteratively update the clusters and cluster means nunm_iterations times
                self.meanSet = self.update_clusters()
                self.mu = self.update_cluster_points()
                errors.append(self.calculate_total_error())
                pass

            #calculate error of current mu model
            model_error = self.calculate_total_error()
            
            if verbose:
                print("MSE for trial {} : {}".format(trial, model_error))
    #             plot convergence over iterations
                plt.plot(np.arange(0,num_iterations, 1), errors.copy(), c=col.copy())
    #             reset errors, iterate color
                col[1] += (1/num_trials) * 0.9
                col[2] -= (1/num_trials) * 0.9
                pass

            #always take the best model w/ minimum error
            if(model_error < best_error):
                best_error = model_error
                best_mu = self.mu
                best_meanSet = self.meanSet
            pass
        pass
        if verbose:
            plt.show()
            pass
        print("Saved best mu with mean squared error: {}".format(best_error))
        return best_mu


    def init_mu(self):
        k = self.k
        X = self.pointMatrix
        mu = np.zeros((k, X.shape[1]))
#         print("generating mu of {} from x of {}".format(mu.shape, X.shape))
        for i in range(X.shape[1]):
            for j in range(k):
                index = int(np.random.ranf() * X.shape[0])
                mu[j] = X[index]
                pass
            pass
        return mu

    def update_clusters(self):
        pointMatrix = self.pointMatrix
        mu = self.mu
        k = self.k
        meanSet = [[] for i in range(k)]
        #   iterate over points
        for i in range(pointMatrix.shape[0]):
            minIndex = 0
            minDistance = np.linalg.norm(pointMatrix[i] - mu[minIndex])
        #     iterate over mu (mean points)
            for j in range(k):
                dist = np.linalg.norm(pointMatrix[i] - mu[j])
                #       pick j with the minimum distance from i
                if(dist < minDistance):
                    minDistance = dist
                    minIndex = j
                    pass
                pass
            #   Add point i to mu[j]'s cluster'
            meanSet[minIndex].append(pointMatrix[i])
            pass
        return meanSet

    def update_cluster_points(self):
        meanSet = self.meanSet
        mu = self.mu
        k = self.k
        #   iterate over mu
        for i in range(k):
            set_sum = np.zeros(mu[i].shape)
            #     iterate over mu[i]'s cluster'
            for j in range(len(meanSet[i])):
                # sum up all the positions of each point
                set_sum += meanSet[i][j]
                pass
            # update mu to the average of each point in mu's cluster
            if len(meanSet[i]) != 0:
                mu[i] = set_sum / len(meanSet[i])
                pass
            pass
        return mu

    def calculate_total_error(self):
        meanSet = self.meanSet
        mu = self.mu
        mserror = 0
        N = 0
        for i in range(self.k):
            for j in range(len(meanSet[i])):
                error = np.linalg.norm(meanSet[i][j] - mu[i])
                mserror += error * error
                N += 1
                pass
            pass
        return mserror / N

    pass


# Let's generate a large N

N = 1000

(train_X, train_Y) = generateSinPairs(N, 4)

K = int(2 * math.log(N))

Lloyd = KMeans(K, train_X)

mu = Lloyd.assign_means(10,30)
plt.scatter(mu, [1 for i in range(mu.shape[0])])
plt.show()
def regress_w(X, Y, mu, lambda_val):
    A = np.zeros((X.shape[0], mu.shape[0]))
    #   iterate over X
    for i in range(X.shape[0]):
        #     iterate over mu
        for j in range(mu.shape[0]):
            #     create vectors for x and mu
            _x = np.transpose([X[i]])
            _mu = np.transpose([mu[j]])
            #     Take the distance between point x and mu
            dist = np.linalg.norm(_x - _mu)
            #   square distance, multiply by lambda
            exponent = (- lambda_val[j]) * dist * dist
            A[i][j] = math.exp(exponent)
            pass
        pass

    #   Get ATA
    transpose = np.transpose(A)
    ATA = np.dot(transpose, A)

    #   Invert ATA
    pseudoInv = np.linalg.inv(ATA)

    #   Take(ATA)^-1 ATy
    res = np.dot(transpose, Y)
    res = np.dot(pseudoInv, res)
    return res
# Train on 1000 values
train_N = 1000
# Error calculated from 100 values
test_N = 200
# Set K proportional to log size of training set
K = int(2 * math.log(train_N))

# Generate relevant (x, y) pairs
(train_X, train_Y) = generateSinPairs(train_N, 8)
(test_X, test_Y) = generateSinPairs(test_N, 8)

# Run K-means on training set.
Lloyd = KMeans(K, train_X)
mu = Lloyd.assign_means(5,20)

# Set lambda to 1 for all values
lambda_val = np.ones(mu.shape)

print("Training for w...")
#   Train for given precision rate
w = regress_w(train_X, train_Y, mu, lambda_val)
#   Get the error within the training set
E_in = get_loss(train_X, train_Y, lambda_val, mu, w)
# Get the error outside the training set
E_out = get_loss(test_X, test_Y, lambda_val, mu, w)
print('MSE for lambda = {} was {}, In-sample error was {}'.format(1, E_out, E_in))

plot_X = np.arange(-4*math.pi, 4*math.pi, 0.1)
plot_Y = np.zeros(plot_X.shape)
plot_sin = np.zeros(plot_X.shape)
plt.subplot(111)
for i in range(plot_X.shape[0]):
    plot_Y[i] = predict(plot_X[i], mu, lambda_val, w)
    plot_sin[i] = math.sin(plot_X[i])
    pass

plt.plot(plot_X, plot_Y, c=[0,0,1], label="f(x)")
plt.plot(plot_X, plot_sin, c=[1,0,0], label="sin(x)")
plt.legend(loc="best")
plt.show()

plt.subplot(111)
x = np. arange(-3,7,0.1)
y = lambda x : x ** 4 - 7 *x ** 3 - 9*x
grad = lambda x : 4 * x ** 3 - 21 *x ** 2 - 9
tangent_line = lambda s : grad(s) * x + (y(s) - grad(s)*s)
plt.plot(x, y(x), c=[1,0,0], label="f(x)")
plt.plot(x,tangent_line(3), c=[0,1,0], label="tangent at x=3")
plt.legend(loc='best')
plt.show()
def descend_lambda (x, y, mu, lambda_val, w, eta):
#     Predict the value of f(x)
#     'eta' is the learning rate, by the way
    f = predict(x, mu, lambda_val, w)
    coefficient =  - eta * (y - f)
    learning_vector = np.ones(mu.shape)
    for i in range(mu.shape[0]):
        dist = (x - mu[i]) **2
        exponent = -lambda_val[i] * dist
        learning_vector[i] = dist * w[i] * math.exp(exponent)
        pass
    new_lambda = lambda_val.copy()
    new_lambda += coefficient * learning_vector
    return new_lambda        
# This block initializes our dataset and uses K-means to find the vector mu.

# Train on 1000 values
train_N = 1000
# Error calculated from 100 values
test_N = 200
# Set K proportional to log size of training set
K = int(2 * math.log(train_N))

# Generate relevant (x, y) pairs
(train_X, train_Y) = generateSinPairs(train_N, 8)
(test_X, test_Y) = generateSinPairs(test_N, 8)

# Run K-means on training set.
print("Running K-means...")
Lloyd = KMeans(K, train_X)
mu = Lloyd.assign_means(10,15, verbose=False)
# Start off lambda as 1 for all values
lambda_val = np.ones(mu.shape) * 1

# Set the learning rate and number of epochs
eta = 0.005
num_epochs = 30

# Set up
E_in_vector = []
E_out_vector = []
epoch_X = np.arange(0,num_epochs,1)

print("Training model...")
for epoch in range(num_epochs):
#     Regress to find w
    w = regress_w(train_X, train_Y, mu, lambda_val)
#     use gradient dexcent through one epoch to find lambda
    for i in range(train_X.shape[0]):
        lambda_val = descend_lambda(train_X[i], train_Y[i], mu, lambda_val, w, eta)
        pass
    
    #   Get the error within the training set
    E_in = get_loss(train_X, train_Y, lambda_val, mu, w)
    E_in_vector.append(float(E_in))
    
    # Get the error outside the training set
    E_out = get_loss(test_X, test_Y, lambda_val, mu, w)
    E_out_vector.append(float(E_out))
    
    print('MSE for epoch {} was {}, In-sample error was {}'.format(epoch, E_out, E_in))
    pass

plt.subplot(111)
plt.plot(epoch_X, E_in_vector, c=[0,0,1], label ="in-sample error")
plt.plot(epoch_X, E_out_vector, c=[0,1,0], label ="out-of-sample error")
plt.legend(loc='best')
plt.show()

plot_X = np.arange(-4*math.pi, 4*math.pi, 0.1)
plot_Y = np.zeros(plot_X.shape)
plot_sin = np.zeros(plot_X.shape)
plt.subplot(111)
for i in range(plot_X.shape[0]):
    plot_Y[i] = predict(plot_X[i], mu, lambda_val, w)
    plot_sin[i] = math.sin(plot_X[i])
    pass

plt.plot(plot_X, plot_Y, c=[0,0,1], label="f(x)")
plt.plot(plot_X, plot_sin, c=[1,0,0], label="sin(x)")
plt.legend(loc="best")
plt.show()
plot_X = np.arange(-10*math.pi, 10*math.pi, 0.1)
plot_Y = np.zeros(plot_X.shape)
plot_sin = np.zeros(plot_X.shape)
plt.subplot(111)
for i in range(plot_X.shape[0]):
    plot_Y[i] = predict(plot_X[i], mu, lambda_val , w)
    plot_sin[i] = math.sin(plot_X[i])
    pass

plt.plot(plot_X, plot_Y, c=[0,0,1], label="Predicted values")
plt.plot(plot_X, plot_sin, c=[1,0,0], label="Actual values")
plt.legend(loc="best")
plt.show()
def generateNoisySinPairs(N, valueRange):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    for i in range(N):
#         Set X to a random value between -4pi and 4pi.
        X[i][0] = (np.random.ranf() *(valueRange) - (0.5*valueRange)) * math.pi
        Y[i][0] = math.sin(X[i][0]) + np.random.ranf()*2 - 1
        pass
    return (X,Y)

(train_X, train_Y) = generateNoisySinPairs(1000, 8)

plt.subplot(111)
plt.scatter(train_X, train_Y)
plt.show()
# Train on 10000 values
train_N = 1000
# Error calculated from 100 values
test_N = 200
# Set K proportional to log size of training set
K = int(2 * math.log(train_N))

# Generate relevant (x, y) pairs
(train_X, train_Y) = generateNoisySinPairs(train_N, 8)
(test_X, test_Y) = generateNoisySinPairs(test_N, 8)

# Run K-means on training set.
print("Running K-means for K={}...".format(K))
Lloyd = KMeans(K, train_X)
mu = Lloyd.assign_means(10,15, verbose=False)
# Start off lambda as 1 for all values
lambda_val = np.ones(mu.shape) * 1

# Set the learning rate and number of epochs
eta = 0.005
num_epochs = 60

# Set up error vectors to plot data
E_in_vector = []
E_out_vector = []
epoch_X = np.arange(0,num_epochs,1)

print("Training model...")
for epoch in range(num_epochs):
#     Regress to find w
    w = regress_w(train_X, train_Y, mu, lambda_val)
#     use gradient dexcent through one epoch to find lambda
    for i in range(train_X.shape[0]):
        lambda_val = descend_lambda(train_X[i], train_Y[i], mu, lambda_val, w, eta)
        pass
    
    #   Get the error within the training set
    E_in = get_loss(train_X, train_Y, lambda_val, mu, w)
    E_in_vector.append(float(E_in))
    
    # Get the error outside the training set
    E_out = get_loss(test_X, test_Y, lambda_val, mu, w)
    E_out_vector.append(float(E_out))
    
    if epoch % 10 == 0:
        print('MSE for epoch {} was {}, In-sample error was {}'.format(epoch, E_out, E_in))
    pass

plt.subplot(111)
plt.plot(epoch_X, E_in_vector, c=[0,0,1], label ="in-sample error")
plt.plot(epoch_X, E_out_vector, c=[0,1,0], label ="out-of-sample error")
plt.legend(loc='best')
plt.show()
plot_X = np.arange(-7*math.pi, 7*math.pi, 0.1)
plot_Y = np.zeros(plot_X.shape)
plot_sin = np.zeros(plot_X.shape)
plt.subplot(111)
for i in range(plot_X.shape[0]):
    plot_Y[i] = predict(plot_X[i], mu, lambda_val, w)
    plot_sin[i] = math.sin(plot_X[i])
    pass

plt.plot(plot_X, plot_Y, c=[0,0,1], label="f(x)")
plt.plot(plot_X, plot_sin, c=[1,0,0], label="sin(x)")
plt.legend(loc="best")
plt.show()