import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from surprise import CoClustering

from surprise import SVD

from surprise import Dataset

from surprise import Trainset

from surprise import Reader

from surprise import accuracy

from surprise.model_selection import cross_validate

from surprise.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load Data

ratings_train = pd.read_csv('/kaggle/input/moviedataset/train.txt', sep='\t', header=None)

ratings_test = pd.read_csv('/kaggle/input/moviedataset/test.txt', sep='\t', header=None)

ratings = pd.read_csv('/kaggle/input/moviedataset/data.txt', sep='\t', header=None)



genres = pd.read_csv('/kaggle/input/moviedataset/movies.txt', sep='\t', header=None)



# Set the column labels for all of the data sets.

ratings_train.columns = ['User Id', 'Movie Id', 'Rating']

ratings_test.columns = ['User Id', 'Movie Id', 'Rating']

ratings.columns = ['User Id', 'Movie Id', 'Rating']



genres.columns = ['Movie Id', 'Movie Title', 'Unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
counts = genres['Movie Title'].value_counts()

print(f'There are {len(counts[counts > 1])} movies with multiple ids:')

counts[counts > 1]
print('Number of Ids = {}'.format(genres['Movie Id'].nunique()))

print('Number of Ids with at least one rating = {}'.format(ratings['Movie Id'].nunique()))

print('All movies have at least one rating.')
def prep(Y, genres, ret_gen=False):

    # Add a 'Movie Titles' Column to ratings

    Y['Movie Title'] = np.array(genres.loc[Y['Movie Id'] - 1, 'Movie Title'])



    # Drop duplicates from genres and assign new ids

    genres = genres.drop_duplicates(subset='Movie Title', keep='first')

    genres['Movie Id'] = np.arange(genres.shape[0]) + 1



    # Assign new ids to ratings using the 'Movie Titles' Column

    title_index_genres = genres.set_index(['Movie Title'])

    Y.loc[:, 'Movie Id'] = np.array(title_index_genres.loc[Y.loc[:, 'Movie Title'], 'Movie Id'])

    Y.drop(['Movie Title'], axis=1, inplace=True)

    

    return (Y, genres) if ret_gen else Y



# Preprocess the ratings and genres data sets.

ratings_train = prep(ratings_train, genres)

ratings_test = prep(ratings_test, genres)

ratings, genres = prep(ratings, genres, True)
# Create a histogram with all of the movie data.

plt.figure()

plt.hist(ratings['Rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True)

plt.xlabel('Movie Rating')

plt.ylabel('Frequency of Rating')

plt.title('Frequency of Rating for All Movies')

plt.show()
# Create a histogram with only the most popular movies.

popular_indices = ratings['Movie Id'].value_counts()[:10].index

popular = ratings[ratings['Movie Id'].isin(popular_indices)]



plt.figure()

plt.hist(popular['Rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True)

plt.xlabel('Movie Rating')

plt.ylabel('Frequency of Rating')

plt.title('Frequency of Rating for 10 Most Popular Movies')

plt.show()
# Create a histogram with only the best (highest mean rating) movies.

best_data = ratings.set_index(['Movie Id']).sort_index()

best_data = pd.DataFrame([[i, best_data.loc[i, 'Rating'].mean()] for i in np.unique(best_data.index)], columns=['Movie Id', 'Mean Rating'])

best_indices = best_data.sort_values(by=['Mean Rating'], ascending=False)['Movie Id'][:10]

best = ratings[ratings['Movie Id'].isin(best_indices)]



plt.figure()

plt.hist(best['Rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True)

plt.xlabel('Movie Rating')

plt.ylabel('Frequency of Rating')

plt.title('Frequency of Rating for 10 Best Movies')

plt.show()
# Create a histogram for three genres.

gen = ['Comedy', 'Drama', 'Film-Noir']



for g in gen:

    g_ratings = ratings[ratings['Movie Id'].isin(genres[genres[g] == 1]['Movie Id'])]

    

    plt.figure()

    plt.hist(g_ratings['Rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True)

    plt.xlabel('Movie Rating')

    plt.ylabel('Frequency of Rating')

    plt.title(f'Frequency of Rating for {g} genre')

    plt.show()
def visualize(V):

    '''

    Plot 10 random movies, the 10 most popular movies, the 10 best movies,

    and 10 movies from the given genres on a 2-dimensional projection of

    the collaborative filtering model.

    

    args:

        V. The matrix corresponding to movies in the collaborative filtering

           model.

    '''

    

    # Get the SVD of V

    A, S, B = np.linalg.svd(V.T, full_matrices=False)

    A = A[:, :2]



    # Calculate the 2-dimensional projections of U and V.

    V_proj = (V @ A).T



    # Normalise the U and V projection to have zero mean and unit variance.

    V_proj = (V_proj - V_proj.mean(axis=1).reshape((-1, 1))) / V_proj.std(axis=1).reshape((-1, 1))



    # Get ten random movies, and ten movies from each genre in the genre list.

    random_indices = np.random.choice(ratings['Movie Id'], size=10, replace=False)

    genre_indices = np.unique(np.array([np.array(genres[genres[g] == 1]['Movie Id'])[:10] for g in gen]).reshape(-1))

    

    graphs = [

        (random_indices, 'Visualization of 10 Random Movies'),

        (popular_indices, 'Visualization of 10 Most Popular Movies'),

        (best_indices, 'Visualization of 10 Best Movies'),

        (genre_indices, f'Visualization of 10 Movies from {", ".join(gen)} Genres')

    ]

    for indices, title in graphs:

        Vp = V_proj[:, indices-1]

        titles = np.array(genres[genres['Movie Id'].isin(indices)].loc[:, 'Movie Title'])



        # Plot the projections on the plane.

        plt.figure(figsize=(10, 10))

        plt.scatter(Vp[0], Vp[1])



        # Annotate all of the data points with the movie title.

        for i, pt in enumerate(Vp.T):

            plt.annotate(titles[i-1], pt, textcoords="offset points", xytext=(0,10), ha='center')



        plt.title(title)

        plt.show()
def grad_U(Ui, Yij, Vj, reg, eta):

    """

    Takes as input Ui (the ith row of U), a training point Yij, the column

    vector Vj (jth column of V^T), reg (the regularization parameter lambda),

    and eta (the learning rate).



    Returns the gradient of the regularized loss function with

    respect to Ui multiplied by eta.

    """

    

    # Calculate the gradient with respect to u_i.

    reg_term = reg * Ui

    pred_term = -(Yij - np.dot(Ui, Vj)) * Vj

    return eta * (reg_term + pred_term)



def grad_V(Vj, Yij, Ui, reg, eta):

    """

    Takes as input the column vector Vj (jth column of V^T), a training point Yij,

    Ui (the ith row of U), reg (the regularization parameter lambda),

    and eta (the learning rate).



    Returns the gradient of the regularized loss function with

    respect to Vj multiplied by eta.

    """

    

    # Calculate the gradient with respect to v_i.

    reg_term = reg * Vj

    pred_term = -(Yij - np.dot(Ui, Vj)) * Ui

    return eta * (reg_term + pred_term)

    

def get_err(U, V, Y, reg=0.0):

    """

    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,

    j is the index of a movie, and Y_ij is user i's rating of movie j and

    user/movie matrices U and V.



    Returns the mean regularized squared-error of predictions made by

    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.

    """

    

    # Calculate the error of the model.

    reg_term = 0.5 * reg * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)

    pred_term = 0.5 * np.mean([(Yij - np.dot(U[i-1], V[j-1]))**2 for i, j, Yij in Y])

    

    return reg_term + pred_term



def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, verbose=False):

    """

    Given a training data matrix Y containing rows (i, j, Y_ij)

    where Y_ij is user i's rating on movie j, learns an

    M x K matrix U and N x K matrix V such that rating Y_ij is approximated

    by (UV^T)_ij.



    Uses a learning rate of <eta> and regularization of <reg>. Stops after

    <max_epochs> epochs, or once the magnitude of the decrease in regularized

    MSE between epochs is smaller than a fraction <eps> of the decrease in

    MSE after the first epoch.



    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE

    of the model.

    """

    

    if verbose: print(f'Training latent factor model for {max_epochs} epochs: ', end='')

    

    # Initialise the U and V matrices to uniform random numbers.

    U = np.random.uniform(-0.5, 0.5, size=(M, K))

    V = np.random.uniform(-0.5, 0.5, size=(N, K))

    

    err = get_err(U, V, Y, reg)

    # Run SGD on the training data Y.

    for ep in range(max_epochs):

        if verbose: print('.', end='')

        

        # Update the matrices.

        for i, j, Yij in np.random.permutation(Y):

            gradu = grad_U(U[i-1], Yij, V[j-1], reg, eta)

            gradv = grad_V(V[j-1], Yij, U[i-1], reg, eta)

            

            U[i-1] = U[i-1] - gradu

            V[j-1] = V[j-1] - gradv

            

        # Calculate the change in error, and break if exit condition met.

        new_err = get_err(U, V, Y, reg)

        if ep == 0:

            delta_first_err = err - new_err

        elif err - new_err < eps * delta_first_err:

            if verbose: praint('\nStopping condition reached.', end='')

            break

        err = new_err

    

    if verbose: print()

    return U, V, get_err(U, V, Y)
def run(train_data, test_data=None, K=20, reg=0.1, eta=0.03):

    # Convert the data sets to np arrays for ease of manipulation.

    train_data = np.array(train_data)

    if test_data is not None:

        test_data = np.array(test_data)

    

    # The maximum user id and movie id.

    if test_data is not None:

        M = np.concatenate([train_data[:, 0], test_data[:, 0]]).max()

        N = np.concatenate([train_data[:, 1], test_data[:, 1]]).max()

    else:

        M = train_data[:, 0].max()

        N = train_data[:, 1].max()

    

    # Run the model and compute Ein and Eout

    U, V, err = train_model(M, N, K, eta, reg, train_data)

    

    # Return the training and testing errors.

    if test_data is not None:

        return U, V, err, get_err(U, V, test_data)

    else:

        return U, V, err





# Run validation to select the optimal lambda.

LAMBDA_RANGE = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]



E_in = []

E_out = []

for l in LAMBDA_RANGE:

    print(f'lambda: {l}')

    U, V, ei, eo = run(ratings_train, ratings_test, reg=l)

    E_in.append(ei)

    E_out.append(eo)

    

lam = LAMBDA_RANGE[np.argmin(E_out)]

print(f'Best lambda: {lam}')



# Plot the learning curves

plt.figure()

plt.plot(LAMBDA_RANGE, E_in, label='Train Error')

plt.plot(LAMBDA_RANGE, E_out, label='Test Error')

plt.xscale('log')

plt.xlabel('Regularization Parameter')

plt.title('Learning Curves for Different Regularization Strengths')

plt.legend()

plt.show()



# Run the model model on the whole data to get U and V.

U, V, E = run(ratings, reg=lam)



print(f'Error on whole data set: {E}')
visualize(V)
def grad_U(Ui, Yij, Vj, mu, ai, bj, reg, eta):

    """

    Takes as input Ui (the ith row of U), a training point Yij, the column

    vector Vj (jth column of V^T), the user bias term ai, the movie bias term bj,

    the average of all observation mu, reg (the regularization parameter lambda),

    and eta (the learning rate).



    Returns the gradient of the regularized loss function with

    respect to Ui multiplied by eta.

    """

    

    # Calculate the gradient with respect to u_i.

    reg_term = reg * Ui

    pred_term = -(Yij - mu - np.dot(Ui, Vj) - ai - bj) * Vj

    return eta * (reg_term + pred_term)



def grad_V(Ui, Yij, Vj, mu, ai, bj, reg, eta):

    """

    Takes as input the column vector Vj (jth column of V^T), a training point Yij,

    Ui (the ith row of U), reg (the regularization parameter lambda),

    and eta (the learning rate).



    Returns the gradient of the regularized loss function with

    respect to Vj multiplied by eta.

    """

    

    # Calculate the gradient with respect to v_i.

    reg_term = reg * Vj

    pred_term = -(Yij - mu - np.dot(Ui, Vj) - ai - bj) * Ui

    return eta * (reg_term + pred_term)



def grad_a(Ui, Yij, Vj, mu, ai, bj, reg, eta):

    """

    Takes as input Ui (the ith row of U), a training point Yij, the column

    vector Vj (jth column of V^T), the user bias term ai, the movie bias term bj,

    the average of all observation mu, reg (the regularization parameter lambda),

    and eta (the learning rate).



    Returns the gradient of the regularized loss function with

    respect to ai multiplied by eta.

    """

    

    # Calculate the gradient with respect to u_i.

    reg_term = reg * ai

    pred_term = -(Yij - mu - np.dot(Ui, Vj) - ai - bj)

    return eta * (reg_term + pred_term)



def grad_b(Ui, Yij, Vj, mu, ai, bj, reg, eta):

    """

    Takes as input the column vector Vj (jth column of V^T), a training point Yij,

    Ui (the ith row of U), reg (the regularization parameter lambda),

    and eta (the learning rate).



    Returns the gradient of the regularized loss function with

    respect to bj multiplied by eta.

    """

    

    # Calculate the gradient with respect to v_i.

    reg_term = reg * bj

    pred_term = -(Yij - mu - np.dot(Ui, Vj) - ai - bj)

    return eta * (reg_term + pred_term)

    

def get_err(U, V, Y, mu, a, b, reg=0.0):

    """

    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,

    j is the index of a movie, and Y_ij is user i's rating of movie j and

    user/movie matrices U and V.



    Returns the mean regularized squared-error of predictions made by

    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.

    """

    

    # Calculate the error of the model.

    reg_term = 0.5 * reg * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2 + np.linalg.norm(a)**2 + np.linalg.norm(b)**2)

    pred_term = 0.5 * np.mean([(Yij - mu - np.dot(U[i-1], V[j-1]) - a[i-1] - b[j-1])**2 for i, j, Yij in Y])

    

    return reg_term + pred_term



def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, verbose=False):

    """

    Given a training data matrix Y containing rows (i, j, Y_ij)

    where Y_ij is user i's rating on movie j, learns an

    M x K matrix U and N x K matrix V such that rating Y_ij is approximated

    by (UV^T)_ij.



    Uses a learning rate of <eta> and regularization of <reg>. Stops after

    <max_epochs> epochs, or once the magnitude of the decrease in regularized

    MSE between epochs is smaller than a fraction <eps> of the decrease in

    MSE after the first epoch.



    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE

    of the model.

    """

    

    if verbose: print(f'Training latent factor model for {max_epochs} epochs: ', end='')

    

    # Initialise the U and V matrices to uniform random numbers.

    U = np.random.uniform(-0.5, 0.5, size=(M, K))

    V = np.random.uniform(-0.5, 0.5, size=(N, K))

    

    # Find mu, and initialise a and b vectors.

    mu = Y[:, 2].mean()

    a = np.zeros(M)

    b = np.zeros(N)

    

    err = get_err(U, V, Y, mu, a, b, reg)

    # Run SGD on the training data Y.

    for ep in range(max_epochs):

        if verbose: print('.', end='')

        

        # Update the matrices.

        for i, j, Yij in np.random.permutation(Y):

            # Calculate all of the gradients.

            gradu = grad_U(U[i-1], Yij, V[j-1], mu, a[i-1], b[j-1], reg, eta)

            gradv = grad_V(V[j-1], Yij, U[i-1], mu, a[i-1], b[j-1], reg, eta)

            grada = grad_a(U[i-1], Yij, V[j-1], mu, a[i-1], b[j-1], reg, eta)

            gradb = grad_b(U[i-1], Yij, V[j-1], mu, a[i-1], b[j-1], reg, eta)

            

            # Update all of the parameters according to their gradients.

            U[i-1] = U[i-1] - gradu

            V[j-1] = V[j-1] - gradv

            a[i-1] = a[i-1] - grada

            b[j-1] = b[j-1] - gradb

            

        # Calculate the change in error, and break if exit condition met.

        new_err = get_err(U, V, Y, mu, a, b, reg)

        if ep == 0:

            delta_first_err = err - new_err

        elif err - new_err < eps * delta_first_err:

            if verbose: print('\nStopping condition reached.', end='')

            break

        err = new_err

    

    if verbose: print()

    return U, V, mu, a, b, get_err(U, V, Y, mu, a, b)
def run(train_data, test_data=None, K=20, reg=0.1, eta=0.03):

    # Convert the data sets to np arrays for ease of manoulation.

    train_data = np.array(train_data)

    if test_data is not None:

        test_data = np.array(test_data)

    

    # The maximum user id and movie id.

    if test_data is not None:

        M = np.concatenate([train_data[:, 0], test_data[:, 0]]).max()

        N = np.concatenate([train_data[:, 1], test_data[:, 1]]).max()

    else:

        M = train_data[:, 0].max()

        N = train_data[:, 1].max()

    

    # Run the model and compute Ein and Eout

    U, V, mu, a, b, err = train_model(M, N, K, eta, reg, train_data)

    

    # Return the training and testing errors.

    if test_data is not None:

        return U, V, mu, a, b, err, get_err(U, V, test_data, mu, a, b)

    else:

        return U, V, mu, a, b, err





# Run validation to select the optimal lambda.

LAMBDA_RANGE = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]



E_in = []

E_out = []

for l in LAMBDA_RANGE:

    print(f'lambda: {l}')

    U, V, mu, a, b, ei, eo = run(ratings_train, ratings_test, reg=l)

    E_in.append(ei)

    E_out.append(eo)

    

lam = LAMBDA_RANGE[np.argmin(E_out)]

print(f'Best lambda: {lam}')



# Plot the learning curves

plt.figure()

plt.plot(LAMBDA_RANGE, E_in, label='Train Error')

plt.plot(LAMBDA_RANGE, E_out, label='Test Error')

plt.xscale('log')

plt.xlabel('Regularization Parameter')

plt.title('Learning Curves for Different Regularization Strengths')

plt.legend()

plt.show()



# Run the model model on the whole data to get U and V.

U, V, mu, a, b, E = run(ratings, reg=lam)



print(f'Error on whole data set: {E}')
# Visualise the V matrix.

visualize(V)
## Export the projection for Virtualitics.



# Get the SVD of V

A, S, B = np.linalg.svd(V.T, full_matrices=False)

A = A[:, :3]



# Calculate the 3-dimensional projections of U and V.

V_proj = (V @ A).T



# Normalise the U and V projection to have zero mean and unit variance.

V_proj = (V_proj - V_proj.mean(axis=1).reshape((-1, 1))) / V_proj.std(axis=1).reshape((-1, 1))



# Get a list of all of the genres

gen = genres.columns[2:]



# Get a list of all of the movies from each genre, appended to the end.

gen_indices = [np.array(genres.loc[genres[g] == 1, 'Movie Id']) for g in gen]

indices = []

for g in gen_indices:

    indices.extend(g)

indices = np.array(indices)



# Get the projection of all of the movies, and add them as rows.

V_selected = V_proj[:, indices-1]

viz_data = pd.DataFrame(V_selected.T)



# Add the Movie Id as a column.

viz_data['Movie Id'] = indices



# Add the Movie Title as a column.

id_index_genres = genres.set_index(['Movie Id'])

viz_data['Movie Title'] = np.array(id_index_genres.loc[indices, 'Movie Title'])



# Add the rating as a column.

viz_data['Rating'] = np.array(best_data.set_index(['Movie Id']).loc[indices, 'Mean Rating'])



# Add the genres as a column.

gens = []

for i, g in enumerate(gen):

    gens.extend([g] * len(gen_indices[i]))

gens = np.array(gens)

viz_data['Genre'] = np.array(gens)



# Download the DataFrame as a CSV.

viz_data.to_csv('genres_10lf.csv')