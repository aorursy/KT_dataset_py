# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

from scipy.io import loadmat



mat = loadmat('../input/dataset/ex8data1.mat')

X = mat["X"]

Xval = mat["Xval"]

yval = mat["yval"]



plt.scatter(X[:,0],X[:,1],marker="x")

plt.xlim(0,30)

plt.ylim(0,30)

plt.xlabel("Latency (ms)")

plt.ylabel("Throughput (mb/s)")
def estimateGaussian(X):

    """

     This function estimates the parameters of a Gaussian distribution using the data in X

    """

    m = X.shape[0]

    

    #compute mean

    sum_ = np.sum(X,axis=0)

    mu = 1/m *sum_

    

    # compute variance

    var = 1/m * np.sum((X - mu)**2,axis=0)

    

    return mu,var

mu, sigma2 = estimateGaussian(X)
def multivariateGaussian(X, mu, sigma2):

    """

    Computes the probability density function of the multivariate gaussian distribution.

    """

    k = len(mu)

    

    sigma2=np.diag(sigma2)

    X = X - mu.T

    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(sigma2) * X,axis=1))

    return p

p = multivariateGaussian(X, mu, sigma2)
plt.figure(figsize=(8,6))

plt.scatter(X[:,0],X[:,1],marker="x")

X1,X2 = np.meshgrid(np.linspace(0,35,num=70),np.linspace(0,35,num=70))

p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)

contour_level = 10**np.array([np.arange(-20,0,3,dtype=np.float)]).T



print(contour_level)



for x in range(7):

    plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level[x])

plt.xlim(0,35)

plt.ylim(0,35)

plt.xlabel("Latency (ms)")

plt.ylabel("Throughput (mb/s)")
def selectThreshold(yval, pval):

    """

    Find the best threshold (epsilon) to use for selecting outliers

    """

    best_epi = 0

    best_F1 = 0

    

    stepsize = (max(pval) -min(pval))/1000

    epi_range = np.arange(pval.min(),pval.max(),stepsize)

    for epi in epi_range:

        predictions = (pval<epi)[:,np.newaxis]

        tp = np.sum(predictions[yval==1]==1)

        fp = np.sum(predictions[yval==0]==1)

        fn = np.sum(predictions[yval==1]==0)

        

        # compute precision, recall and F1

        prec = tp/(tp+fp)

        rec = tp/(tp+fn)

        

        F1 = (2*prec*rec)/(prec+rec)

        

        if F1 > best_F1:

            best_F1 =F1

            best_epi = epi

        

    return best_epi, best_F1

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)

print("Best epsilon found using cross-validation:",epsilon)

print("Best F1 on Cross Validation Set:",F1)
plt.figure(figsize=(8,6))

# plot the data

plt.scatter(X[:,0],X[:,1],marker="x")

# potting of contour

X1,X2 = np.meshgrid(np.linspace(0,35,num=70),np.linspace(0,35,num=70))

p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)

contour_level = 10**np.array([np.arange(-20,0,3,dtype=np.float)]).T

for x in range(7):

    plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level[x])

# Circling of anomalies

outliers = np.nonzero(p<epsilon)[0]

plt.scatter(X[outliers,0],X[outliers,1],marker ="o",facecolor="none",edgecolor="r",s=70)

plt.xlim(0,35)

plt.ylim(0,35)

plt.xlabel("Latency (ms)")

plt.ylabel("Throughput (mb/s)")
mat2 = loadmat("../input/dataset3/ex8data2.mat")

X2 = mat2["X"]

Xval2 = mat2["Xval"]

yval2 = mat2["yval"]

# compute the mean and variance

mu2, sigma2_2 = estimateGaussian(X2)

# Training set

p3 = multivariateGaussian(X2, mu2, sigma2_2)

# cross-validation set

pval2 = multivariateGaussian(Xval2, mu2, sigma2_2)

# Find the best threshold

epsilon2, F1_2 = selectThreshold(yval2, pval2)

print("Best epsilon found using cross-validation:",epsilon2)

print("Best F1 on Cross Validation Set:",F1_2)

print("# Outliers found:",np.sum(p3<epsilon2))
mat3 = loadmat("../input/setdata1/ex8_movies.mat")

mat4 = loadmat("../input/setdata2/ex8_movieParams.mat")

Y = mat3["Y"] # 1682 X 943 matrix, containing ratings (1-5) of 1682 movies on 943 user

R = mat3["R"] # 1682 X 943 matrix, where R(i,j) = 1 if and only if user j give rating to movie i

X = mat4["X"] # 1682 X 10 matrix , num_movies X num_features matrix of movie features

Theta = mat4["Theta"] # 943 X 10 matrix, num_users X num_features matrix of user features

# Compute average rating 

print("Average rating for movie 1 (Toy Story):",np.sum(Y[0,:]*R[0,:])/np.sum(R[0,:]),"/5")

plt.figure(figsize=(8,16))

plt.imshow(Y)

plt.xlabel("Users")

plt.ylabel("Movies")
def  cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):

    """

    Returns the cost and gradient for the collaborative filtering problem

    """

    

    # Unfold the params

    X = params[:num_movies*num_features].reshape(num_movies,num_features)

    Theta = params[num_movies*num_features:].reshape(num_users,num_features)

    

    predictions =  X @ Theta.T

    err = (predictions - Y)

    J = 1/2 * np.sum((err**2) * R)

    

    #compute regularized cost function

    reg_X =  Lambda/2 * np.sum(Theta**2)

    reg_Theta = Lambda/2 *np.sum(X**2)

    reg_J = J + reg_X + reg_Theta

    

    # Compute gradient

    X_grad = err*R @ Theta

    Theta_grad = (err*R).T @ X

    grad = np.append(X_grad.flatten(),Theta_grad.flatten())

    

    # Compute regularized gradient

    reg_X_grad = X_grad + Lambda*X

    reg_Theta_grad = Theta_grad + Lambda*Theta

    reg_grad = np.append(reg_X_grad.flatten(),reg_Theta_grad.flatten())

    

    return J, grad, reg_J, reg_grad
num_users, num_movies, num_features = 4,5,3

X_test = X[:num_movies,:num_features]

Theta_test= Theta[:num_users,:num_features]

Y_test = Y[:num_movies,:num_users]

R_test = R[:num_movies,:num_users]

params = np.append(X_test.flatten(),Theta_test.flatten())

# Evaluate cost function

J, grad = cofiCostFunc(params, Y_test, R_test, num_users, num_movies, num_features, 0)[:2]

print("Cost at loaded parameters:",J)

J2, grad2 = cofiCostFunc(params, Y_test, R_test, num_users, num_movies, num_features, 1.5)[2:]

print("Cost at loaded parameters (lambda = 1.5):",J2)
# load movie list

import pandas as pd

import numpy as np

import csv



# load movie list

movieList = open("../input/movies/movie_ids.txt","r", encoding="ISO-8859-1").read().split("\n")[:-1]

# see movie list

#np.set_printoptions(threshold=np.nan)

movieList
# Initialize my ratings

my_ratings = np.zeros((1682,1))

# Create own ratings

my_ratings[0] = 4 

my_ratings[97] = 2

my_ratings[6] = 3

my_ratings[11]= 5

my_ratings[53] = 4

my_ratings[63]= 5

my_ratings[65]= 3

my_ratings[68] = 5

my_ratings[82]= 4

my_ratings[225] = 5

my_ratings[354]= 5

print("New user ratings:\n")

for i in range(len(my_ratings)):

    if my_ratings[i]>0:

        print("Rated",int(my_ratings[i]),"for index",movieList[i])
def normalizeRatings(Y, R):

    """

    normalized Y so that each movie has a rating of 0 on average, and returns the mean rating in Ymean.

    """

    

    m,n = Y.shape[0], Y.shape[1]

    Ymean = np.zeros((m,1))

    Ynorm = np.zeros((m,n))

    

    for i in range(m):

        Ymean[i] = np.sum(Y[i,:])/np.count_nonzero(R[i,:])

        Ynorm[i,R[i,:]==1] = Y[i,R[i,:]==1] - Ymean[i]

        

    return Ynorm, Ymean

def gradientDescent(initial_parameters,Y,R,num_users,num_movies,num_features,alpha,num_iters,Lambda):

    """

    Optimize X and Theta

    """

    # unfold the parameters

    X = initial_parameters[:num_movies*num_features].reshape(num_movies,num_features)

    Theta = initial_parameters[num_movies*num_features:].reshape(num_users,num_features)

    

    J_history =[]

    

    for i in range(num_iters):

        params = np.append(X.flatten(),Theta.flatten())

        cost, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda)[2:]

        

        # unfold grad

        X_grad = grad[:num_movies*num_features].reshape(num_movies,num_features)

        Theta_grad = grad[num_movies*num_features:].reshape(num_users,num_features)

        X = X - (alpha * X_grad)

        Theta = Theta - (alpha * Theta_grad)

        J_history.append(cost)

    

    paramsFinal = np.append(X.flatten(),Theta.flatten())

    return paramsFinal , J_history
Y = np.hstack((my_ratings,Y))

R =np.hstack((my_ratings!=0,R))

# Normalize Ratings

Ynorm, Ymean = normalizeRatings(Y, R)

num_users = Y.shape[1]

num_movies = Y.shape[0]

num_features = 10

# Set initial Parameters (Theta,X)

X = np.random.randn(num_movies, num_features)

Theta = np.random.randn(num_users, num_features)

initial_parameters = np.append(X.flatten(),Theta.flatten())

Lambda = 10

# Optimize parameters using Gradient Descent

paramsFinal, J_history = gradientDescent(initial_parameters,Y,R,num_users,num_movies,num_features,0.001,400,Lambda)
plt.plot(J_history)

plt.xlabel("Iteration")

plt.ylabel("$J(\Theta)$")

plt.title("Cost function using Gradient Descent")
# unfold paramaters

X = paramsFinal[:num_movies*num_features].reshape(num_movies,num_features)

Theta = paramsFinal[num_movies*num_features:].reshape(num_users,num_features)

# Predict rating

p = X @ Theta.T

my_predictions = p[:,0][:,np.newaxis] + Ymean

import pandas as pd

df = pd.DataFrame(np.hstack((my_predictions,np.array(movieList)[:,np.newaxis])))

df.sort_values(by=[0],ascending=False,inplace=True)

df.reset_index(drop=True,inplace=True)

print("Top recommendations for you:\n")

for i in range(10):

    print("Predicting rating",round(float(df[0][i]),1)," for index",df[1][i])