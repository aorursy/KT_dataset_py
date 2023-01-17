import pandas as pd



r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv('../input/data02/ml-100k/u.data', sep='\t', names=r_cols)



print('matrix size:', ratings.shape)

ratings.head(10)
# create a movie-user rating_matrix in DataFrame



import numpy as np

num_users = ratings.user_id.unique().shape[0]

num_movies = ratings.movie_id.unique().shape[0]



# movie_id --> index, user_id --> column

rating_matrix = pd.DataFrame(np.nan, index=range(1,num_movies+1), columns=range(1,num_users+1))



# assign ratings to appropriate elements of the matrix

for i in range(ratings.shape[0]):

    rating_matrix.iloc[ratings.movie_id[i]-1, ratings.user_id[i]-1] = ratings.rating[i]



# plot rating_matrix

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



fig, ax = plt.subplots(1,1,figsize=(10,6))

ax = sns.heatmap(rating_matrix, 0,5, cmap='rainbow')



ax.set_title('rating_matrix', fontsize=18)



ax.set_xlabel('user_id', fontsize=14)

ax.set_xticklabels(ax.get_xticklabels()[::100])

ax.set_xticks(ax.get_xticks()[::100])



ax.set_ylabel('movie_id', fontsize=14)

ax.set_yticklabels(ax.get_yticklabels()[::200])

ax.set_yticks(ax.get_yticks()[::200])



plt.show()

print('User ID: %d rated %d movies, the higest number of movies rated by one user.' %

      (rating_matrix.notnull().sum(axis=0).argmax(), rating_matrix.notnull().sum(axis=0).max()))

# It is too much of computation to make a model using all data



# just randomly pick some for testing

rand_user_size = 30

rand_movie_size = 100

rand_user_id = np.random.choice(50, rand_user_size)

rand_movie_id = np.random.choice(150, rand_movie_size)



# new randomly generated rating_matrix

rand_rating_matrix = pd.DataFrame(rating_matrix,rand_movie_id,rand_user_id)



fig, ax = plt.subplots(1,1,figsize=(10,6))

ax = sns.heatmap(rand_rating_matrix, 0,5, cmap='rainbow')



ax.set_title('Randomly generated rating_matrix', fontsize=18)

ax.set_xlabel('random user_id', fontsize=14)

ax.set_ylabel('random movie_id', fontsize=14)

ax.set_xticklabels([])

ax.set_yticklabels([])



plt.show()

# mean-normalization of the actual rating matrix

# make a 'deep' copy

model_ratings = np.copy(rand_rating_matrix.values)



# calculate mean, ignoring NaN

model_mean = np.nanmean(model_ratings, axis=1)



# calculate normalized ratings from model_ratings and actual ratings

model_norm = model_ratings - model_mean.reshape(rand_movie_size,1)

# build an initialized prediction matrix

# set a number of features to learn

n_features =15



# initialize parameters theta (user_prefs), X (movie_features)

# movie_features

X_init = np.random.randn(rand_movie_size, n_features)

# user_prefs

theta_init = np.random.randn(rand_user_size, n_features)



# create 1D array containing 'X' and 'theta'

initial_X_and_theta = np.r_[X_init.flatten(), theta_init.flatten()]



# compute prediction matrix

init_ratings = X_init@theta_init.T



# compare initialized predictions with normalized ratings

fig, ax = plt.subplots(1,2,figsize=(14,6),sharey=True)

cbar_ax = fig.add_axes([0.92, 0.2, .02, 0.6])



for ax, data, title in zip(ax, [init_ratings,model_norm],

                           ['Predicted rating_matrix (initialized)','Actual rating matrix (normalized)']):



    sns.heatmap(data, -3.1, 3.1, cbar=True, cbar_ax=cbar_ax, ax=ax)

    ax.set_title(title, fontsize=18)

    ax.set_xlabel('random user_id', fontsize=14)

    ax.set_ylabel('random movie_id', fontsize=14)

    ax.set_xticklabels([])

    ax.set_yticklabels([])



plt.show()

# a method to separate 'X' and 'theta'

def X_and_theta_sep(X_and_theta, n_movies, n_users, n_features):



    # extract 'X' from the first (n_movies*n_features) elements of 'X_and_theta'

    X = X_and_theta[:n_movies*n_features].reshape(n_movies, n_features)

    # extract 'theta' from the rest

    theta = X_and_theta[n_movies*n_features:].reshape(n_users, n_features)



    return X, theta

# a method to calculate cost function

def cost_cal(X_and_theta, ratings, n_movies, n_users, n_features, reg_param):



    # get 'X' and 'theta'

    X, theta = X_and_theta_sep(X_and_theta, n_movies, n_users, n_features)



    # calculate the cost function due to accuracy and regularization

    cost = np.nansum((X@theta.T - ratings)**2)/2

    regCost = (reg_param/2)*(np.sum(theta**2) + np.sum(X**2))



    return cost + regCost



# set a regularization parameter

reg_param = 1



# initial cost value

print('Initial cost value:', cost_cal(initial_X_and_theta, model_norm, rand_movie_size, rand_user_size, n_features, reg_param))

# a method to calculate gradients

def gradient_cal(X_and_theta, ratings, n_movies, n_users, n_features, reg_param):



    # get 'X' and 'theta'

    X, theta = X_and_theta_sep(X_and_theta, n_movies, n_users, n_features)



    # predicted rating

    pred = X@theta.T

    # calculate the difference between predicted and actual rating, replace NaN with 0

    diff = np.nan_to_num(pred-ratings)



    # set learning rate

    learning_rate = 2

    # calculate gradients

    X_grad = learning_rate*(diff@theta + reg_param*X)

    theta_grad = learning_rate*(diff.T@X + reg_param*theta)



    return np.r_[X_grad.flatten(), theta_grad.flatten()]

from scipy.optimize import minimize



# set a regularization parameter

reg_param = 1



# minimizing the cost function

min_res = minimize(cost_cal, initial_X_and_theta, args=(model_norm, rand_movie_size, rand_user_size, n_features, reg_param),

         method=None, jac=gradient_cal, options={'maxiter':3000, 'disp':True})

# initial cost value

print('Initial cost value:', cost_cal(initial_X_and_theta, model_norm, rand_movie_size, rand_user_size, n_features, reg_param))



# final gradient array

print('Optimized cost value:', min_res.fun)



# initial gradient array

n_display = 5

print('Examples of initial gradient values (first %d elements): ' % n_display)

print(gradient_cal(initial_X_and_theta, model_norm, rand_movie_size, rand_user_size, n_features, reg_param)[:n_display])



# final gradient array

print('Examples of optimized gradient values (first %d elements): ' % n_display)

print(min_res.jac[:n_display])

# get predicted 'X' and 'theta'

X, theta = X_and_theta_sep(min_res.x, rand_movie_size, rand_user_size, n_features)



# make rating predictions    

predicted_ratings = X@theta.T



# compare optimized predictions with normalized ratings

fig, ax = plt.subplots(1,2,figsize=(14,6),sharey=True)

cbar_ax = fig.add_axes([0.92, 0.2, .02, 0.6])



for ax, data, title in zip(ax, [predicted_ratings,model_norm],

                           ['Predicted rating_matrix (optimized)','Actual rating matrix (normalized)']):



    sns.heatmap(data, -3.1, 3.1, cbar=True, cbar_ax=cbar_ax, ax=ax)

    ax.set_title(title, fontsize=18)

    ax.set_xlabel('random user_id', fontsize=14)

    ax.set_ylabel('random movie_id', fontsize=14)

    ax.set_xticklabels([])

    ax.set_yticklabels([])



plt.show()
