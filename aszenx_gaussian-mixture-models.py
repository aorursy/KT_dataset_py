import numpy as np
import pandas as pd
import scipy.stats as scs
from scipy.stats import multivariate_normal as mvn # for computing mvn 
import matplotlib.pyplot as plt
import seaborn as sns
# randomly initialize gmm parameters -> means, covariances & weights (strength of each gaussian)
def initialize_gmm_parameters(v_noOfVariables, k_noOfGaussDist, cov_scale=1):
    # initialize k random nos ( range 0-1 ) as weights for each gaussian in gmm
    W_gaussWeights = np.random.random(k_noOfGaussDist)
    # since the weights must sum up to 1 (constraint) we divide each weight by total sum of weights
    W_gaussWeights /= W_gaussWeights.sum() 
    
    # randomly initialize v means for each gaussian in th range 0-100
    U_gaussMeans = np.random.randint(0, 100, (k_noOfGaussDist, v_noOfVariables))
    
    # initialize k covariance matrices each (v*v) for every gaussian, use identity matrices 
    S_covMatrices = np.array([np.eye(v_noOfVariables)] * k_noOfGaussDist ) * cov_scale # generate k matrices each v*v
    
    return [W_gaussWeights, U_gaussMeans, S_covMatrices]
# fit the data X to gmm and the return the optimal paramters
def gmm_fit(X, k_noOfGaussDist, tol=0.1, max_iter=100):
    """Uses Expectation Maximization (EM) algorithm"""
    # look at the data
    n_noOfSamples, v_noOfVariables = X.shape
    # initialize parameters 
    W_gaussWeights, U_gaussMeans, S_covMatrices = initialize_gmm_parameters(v_noOfVariables, k_noOfGaussDist, X.std())
    ll_history = []
    ll_old = 0 # log-likelihood
    # start training
    print('Iterations: ', end='')
    for i in range(max_iter):
        print(str(i) + '.', end='')
        
        # E-STEP ->
            # Compute the probability that each data point was generated by each of the k Gaussians.
            # In other words, compute a matrix where the rows are the data points & the cols are the Gaussians, 
                # an element at row i, column j is the probability that x{i} was generated by Gaussian j.
        
        # initialize matrix E with zeros
        E = np.zeros((k_noOfGaussDist, n_noOfSamples))
        
        # for each sample compute the probability that it belongs to each of the k Gaussians
        for j in range(k_noOfGaussDist):
            for i in range(n_noOfSamples):
                # first form a gaussian with the current parameters then calculate probability for sample
                E[j, i] = W_gaussWeights[j] * mvn(U_gaussMeans[j], S_covMatrices[j], allow_singular=True).pdf(X[i])
        
        # sum of probabilities for different gaussians on a sample must be one so we divide by the sum  
        E /= E.sum(0)
        
        # M-STEP ->
            # In this step, update weights, means, & covariances.
            # 1. For weights -> sum up the probability that each point was generated by Gaussian j and
                # divide by the total number of points.
            # 2. For means -> compute the mean of all points weighted by the probability of that point 
                # being generated by Gaussian j.
            # 3. For covariances -> compute the covariance of all points weighted by the probability of that point
                # being generated by Gaussian j.
            # Do each of these for each Gaussian j.
        
        # Update weights (strength of each gaussian)
        W_gaussWeights = np.zeros(k_noOfGaussDist)
        for j in range(k_noOfGaussDist):
            for i in range(n_noOfSamples):
                W_gaussWeights[j] += E[j, i]
        W_gaussWeights /= n_noOfSamples

        # Update means
        U_gaussMeans = np.zeros((k_noOfGaussDist, v_noOfVariables))
        for j in range(k_noOfGaussDist):
            for i in range(n_noOfSamples):
                U_gaussMeans[j] += E[j, i] * X[i]
            U_gaussMeans[j] /= E[j, :].sum()
 
        # Update covariances
        S_covMatrices = np.zeros((k_noOfGaussDist, v_noOfVariables, v_noOfVariables))
        for j in range(k_noOfGaussDist):
            for i in range(n_noOfSamples):
                ys = np.reshape(X[i] - U_gaussMeans[j], (v_noOfVariables, 1))
                S_covMatrices[j] += E[j, i] * np.dot(ys, ys.T)
            S_covMatrices[j] /= E[j, :].sum()

        # update complete log likelihoood
        # iterate till log-likelihood doesn't change upto a certain tolerance 
        ll_new = 0.0
        for i in range(n_noOfSamples):
            s = 0
            for j in range(k_noOfGaussDist):
                s += W_gaussWeights[j] * mvn(U_gaussMeans[j], S_covMatrices[j], allow_singular=True).pdf(X[i])
            ll_new += np.log(s)
#         print(f'log_likelihood: {ll_new:3.4f}; ')
        ll_history.append(ll_old)
        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new        
        
    return [ll_history,W_gaussWeights.round(2), U_gaussMeans.round(2), S_covMatrices.round(2)] # return the learned optimal parameters
def generate_random_samples(k, v):
    # generate data from k gaussians each having v features(variables)
    weights, means, covariances = initialize_gmm_parameters(v, k) # randomly initialize gmm parameters
    print('\n\nActual Weights: \n %s' % weights)
    print('Actual Means: \n %s' % means)
    print('Actual Covariances: \n %s' % covariances)
    n = 100 # no of samples
    # generate n random samples from these parameters 
    xs = np.concatenate([np.random.multivariate_normal(m, s, int(w*n))
                        for w, m, s in zip(weights, means, covariances)])
    return xs
no_of_gaussians = 3
variables = 3
X = generate_random_samples(no_of_gaussians, variables)
plt.plot(X)
_ = plt.title('Actual data samples')
# train these samples in gmm to get predicted parameters 
ll, weights, means, covariances = gmm_fit(X, no_of_gaussians)
print('\n\nOutput Weights: \n %s' % weights)
print('Output Means: \n %s' % means)
print('Output Covariances: \n %s' % covariances)
# plot log likelihood versus iterations plot
plt.plot(ll)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
_=plt.title('Log likelihood vs iterations')
n = 100
output_samples = np.concatenate([np.random.multivariate_normal(m, s, int(w*n))
                        for w, m, s in zip(weights, means, covariances)])
plt.plot(output_samples)
_=plt.title('Generated samples')