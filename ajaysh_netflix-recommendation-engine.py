# this is just to know how much time will it take to run this entire ipython notebook 

from datetime import datetime

# globalstart = datetime.now()

import pandas as pd

import numpy as np

import matplotlib

matplotlib.use('nbagg')



import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})



import seaborn as sns

sns.set_style('whitegrid')

import os

from scipy import sparse

from scipy.sparse import csr_matrix



from sklearn.decomposition import TruncatedSVD

from sklearn.metrics.pairwise import cosine_similarity

import random
start = datetime.now()

if not os.path.isfile('data.csv'):

    # Create a file 'data.csv' before reading it

    # Read all the files in netflix and store them in one big file('data.csv')

    # reading from each of the four files and appending each rating to a global file 'train.csv'

    data = open('data.csv', mode='w')

    

    row = list()

    files=['data_folder/combined_data_1.txt','data_folder/combined_data_2.txt', 

           'data_folder/combined_data_3.txt', 'data_folder/combined_data_4.txt']

    for file in files:

        print("Reading ratings from {}...".format(file))

        with open(file) as f:

            for line in f: 

                del row[:]

                line = line.strip()

                if line.endswith(':'):

                    # All below are ratings for this movie, until another movie appears.

                    movie_id = line.replace(':', '')

                else:

                    row = [x for x in line.split(',')]

                    row.insert(0, movie_id)

                    data.write(','.join(row))

                    data.write('\n')

        print("Done.\n")

    data.close()

print('Time taken :', datetime.now() - start)
print("creating the dataframe from data.csv file..")

df = pd.read_csv('data.csv', sep=',', 

                       names=['movie', 'user','rating','date'])

df.date = pd.to_datetime(df.date)

print('Done.\n')



# arranging the ratings according to time.

print('Sorting the dataframe by date..')

df.sort_values(by='date', inplace=True)

print('Done..')
df.head()
df.describe()['rating']
# just to make sure that all Nan containing rows are deleted..

print("No of Nan values in our dataframe : ", sum(df.isnull().any()))
dup_bool = df.duplicated(['movie','user','rating'])

dups = sum(dup_bool) # by considering all columns..( including timestamp)

print("There are {} duplicate rating entries in the data..".format(dups))
print("Total data ")

print("-"*50)

print("\nTotal no of ratings :",df.shape[0])

print("Total No of Users   :", len(np.unique(df.user)))

print("Total No of movies  :", len(np.unique(df.movie)))
if not os.path.isfile('train.csv'):

    # create the dataframe and store it in the disk for offline purposes..

    df.iloc[:int(df.shape[0]*0.80)].to_csv("train.csv", index=False)



if not os.path.isfile('test.csv'):

    # create the dataframe and store it in the disk for offline purposes..

    df.iloc[int(df.shape[0]*0.80):].to_csv("test.csv", index=False)



train_df = pd.read_csv("train.csv", parse_dates=['date'])

test_df = pd.read_csv("test.csv")
# movies = train_df.movie.value_counts()

# users = train_df.user.value_counts()

print("Training data ")

print("-"*50)

print("\nTotal no of ratings :",train_df.shape[0])

print("Total No of Users   :", len(np.unique(train_df.user)))

print("Total No of movies  :", len(np.unique(train_df.movie)))
print("Test data ")

print("-"*50)

print("\nTotal no of ratings :",test_df.shape[0])

print("Total No of Users   :", len(np.unique(test_df.user)))

print("Total No of movies  :", len(np.unique(test_df.movie)))
# method to make y-axis more readable

def human(num, units = 'M'):

    units = units.lower()

    num = float(num)

    if units == 'k':

        return str(num/10**3) + " K"

    elif units == 'm':

        return str(num/10**6) + " M"

    elif units == 'b':

        return str(num/10**9) +  " B"
fig, ax = plt.subplots()

plt.title('Distribution of ratings over Training dataset', fontsize=15)

sns.countplot(train_df.rating)

ax.set_yticklabels([human(item, 'M') for item in ax.get_yticks()])

ax.set_ylabel('No. of Ratings(Millions)')



plt.show()
# It is used to skip the warning ''SettingWithCopyWarning''.. 

pd.options.mode.chained_assignment = None  # default='warn'



train_df['day_of_week'] = train_df.date.dt.weekday_name



train_df.tail()
ax = train_df.resample('m', on='date')['rating'].count().plot()

ax.set_title('No of ratings per month (Training data)')

plt.xlabel('Month')

plt.ylabel('No of ratings(per month)')

ax.set_yticklabels([human(item, 'M') for item in ax.get_yticks()])

plt.show()
no_of_rated_movies_per_user = train_df.groupby(by='user')['rating'].count().sort_values(ascending=False)



no_of_rated_movies_per_user.head()
fig = plt.figure(figsize=plt.figaspect(.5))



ax1 = plt.subplot(121)

sns.kdeplot(no_of_rated_movies_per_user, shade=True, ax=ax1)

plt.xlabel('No of ratings by user')

plt.title("PDF")



ax2 = plt.subplot(122)

sns.kdeplot(no_of_rated_movies_per_user, shade=True, cumulative=True,ax=ax2)

plt.xlabel('No of ratings by user')

plt.title('CDF')



plt.show()
no_of_rated_movies_per_user.describe()
quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')
plt.title("Quantiles and their Values")

quantiles.plot()

# quantiles with 0.05 difference

plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")

# quantiles with 0.25 difference

plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")

plt.ylabel('No of ratings by user')

plt.xlabel('Value at the quantile')

plt.legend(loc='best')



# annotate the 25th, 50th, 75th and 100th percentile values....

for x,y in zip(quantiles.index[::25], quantiles[::25]):

    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500)

                ,fontweight='bold')





plt.show()
quantiles[::5]
print('\n No of ratings at last 5 percentile : {}\n'.format(sum(no_of_rated_movies_per_user>= 749)) )
no_of_ratings_per_movie = train_df.groupby(by='movie')['rating'].count().sort_values(ascending=False)



fig = plt.figure(figsize=plt.figaspect(.5))

ax = plt.gca()

plt.plot(no_of_ratings_per_movie.values)

plt.title('# RATINGS per Movie')

plt.xlabel('Movie')

plt.ylabel('No of Users who rated a movie')

ax.set_xticklabels([])



plt.show()
fig, ax = plt.subplots()

sns.countplot(x='day_of_week', data=train_df, ax=ax)

plt.title('No of ratings on each day...')

plt.ylabel('Total no of ratings')

plt.xlabel('')

ax.set_yticklabels([human(item, 'M') for item in ax.get_yticks()])

plt.show()
start = datetime.now()

fig = plt.figure(figsize=plt.figaspect(.45))

sns.boxplot(y='rating', x='day_of_week', data=train_df)

plt.show()

print(datetime.now() - start)
avg_week_df = train_df.groupby(by=['day_of_week'])['rating'].mean()

print(" AVerage ratings")

print("-"*30)

print(avg_week_df)

print("\n")
start = datetime.now()

if os.path.isfile('train_sparse_matrix.npz'):

    print("It is present in your pwd, getting it from disk....")

    # just get it from the disk instead of computing it

    train_sparse_matrix = sparse.load_npz('train_sparse_matrix.npz')

    print("DONE..")

else: 

    print("We are creating sparse_matrix from the dataframe..")

    # create sparse_matrix and store it for after usage.

    # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)

    # It should be in such a way that, MATRIX[row, col] = data

    train_sparse_matrix = sparse.csr_matrix((train_df.rating.values, (train_df.user.values,

                                               train_df.movie.values)),)

    

    print('Done. It\'s shape is : (user, movie) : ',train_sparse_matrix.shape)

    print('Saving it into disk for furthur usage..')

    # save it into disk

    sparse.save_npz("train_sparse_matrix.npz", train_sparse_matrix)

    print('Done..\n')



print(datetime.now() - start)
us,mv = train_sparse_matrix.shape

elem = train_sparse_matrix.count_nonzero()



print("Sparsity Of Train matrix : {} % ".format(  (1-(elem/(us*mv))) * 100) )
start = datetime.now()

if os.path.isfile('test_sparse_matrix.npz'):

    print("It is present in your pwd, getting it from disk....")

    # just get it from the disk instead of computing it

    test_sparse_matrix = sparse.load_npz('test_sparse_matrix.npz')

    print("DONE..")

else: 

    print("We are creating sparse_matrix from the dataframe..")

    # create sparse_matrix and store it for after usage.

    # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)

    # It should be in such a way that, MATRIX[row, col] = data

    test_sparse_matrix = sparse.csr_matrix((test_df.rating.values, (test_df.user.values,

                                               test_df.movie.values)))

    

    print('Done. It\'s shape is : (user, movie) : ',test_sparse_matrix.shape)

    print('Saving it into disk for furthur usage..')

    # save it into disk

    sparse.save_npz("test_sparse_matrix.npz", test_sparse_matrix)

    print('Done..\n')

    

print(datetime.now() - start)
us,mv = test_sparse_matrix.shape

elem = test_sparse_matrix.count_nonzero()



print("Sparsity Of Test matrix : {} % ".format(  (1-(elem/(us*mv))) * 100) )
# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)



def get_average_ratings(sparse_matrix, of_users):

    

    # average ratings of user/axes

    ax = 1 if of_users else 0 # 1 - User axes,0 - Movie axes



    # ".A1" is for converting Column_Matrix to 1-D numpy array 

    sum_of_ratings = sparse_matrix.sum(axis=ax).A1

    # Boolean matrix of ratings ( whether a user rated that movie or not)

    is_rated = sparse_matrix!=0

    # no of ratings that each user OR movie..

    no_of_ratings = is_rated.sum(axis=ax).A1

    

    # max_user  and max_movie ids in sparse matrix 

    u,m = sparse_matrix.shape

    # creae a dictonary of users and their average ratigns..

    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i]

                                 for i in range(u if of_users else m) 

                                    if no_of_ratings[i] !=0}



    # return that dictionary of average ratings

    return average_ratings
train_averages = dict()

# get the global average of ratings in our train set.

train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()

train_averages['global'] = train_global_average

train_averages
train_averages['user'] = get_average_ratings(train_sparse_matrix, of_users=True)

print('\nAverage rating of user 10 :',train_averages['user'][10])
train_averages['movie'] =  get_average_ratings(train_sparse_matrix, of_users=False)

print('\n AVerage rating of movie 15 :',train_averages['movie'][15])
start = datetime.now()

# draw pdfs for average rating per user and average

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(.5))

fig.suptitle('Avg Ratings per User and per Movie', fontsize=15)



ax1.set_title('Users-Avg-Ratings')

# get the list of average user ratings from the averages dictionary..

user_averages = [rat for rat in train_averages['user'].values()]

sns.distplot(user_averages, ax=ax1, hist=False, 

             kde_kws=dict(cumulative=True), label='Cdf')

sns.distplot(user_averages, ax=ax1, hist=False,label='Pdf')



ax2.set_title('Movies-Avg-Rating')

# get the list of movie_average_ratings from the dictionary..

movie_averages = [rat for rat in train_averages['movie'].values()]

sns.distplot(movie_averages, ax=ax2, hist=False, 

             kde_kws=dict(cumulative=True), label='Cdf')

sns.distplot(movie_averages, ax=ax2, hist=False, label='Pdf')



plt.show()

print(datetime.now() - start)
total_users = len(np.unique(df.user))

users_train = len(train_averages['user'])

new_users = total_users - users_train



print('\nTotal number of Users  :', total_users)

print('\nNumber of Users in Train data :', users_train)

print("\nNo of Users that didn't appear in train data: {}({} %) \n ".format(new_users,

                                                                        np.round((new_users/total_users)*100, 2)))
total_movies = len(np.unique(df.movie))

movies_train = len(train_averages['movie'])

new_movies = total_movies - movies_train



print('\nTotal number of Movies  :', total_movies)

print('\nNumber of Users in Train data :', movies_train)

print("\nNo of Movies that didn't appear in train data: {}({} %) \n ".format(new_movies,

                                                                        np.round((new_movies/total_movies)*100, 2)))
from sklearn.metrics.pairwise import cosine_similarity





def compute_user_similarity(sparse_matrix, compute_for_few=False, top = 100, verbose=False, verb_for_n_rows = 20,

                            draw_time_taken=True):

    no_of_users, _ = sparse_matrix.shape

    # get the indices of  non zero rows(users) from our sparse matrix

    row_ind, col_ind = sparse_matrix.nonzero()

    row_ind = sorted(set(row_ind)) # we don't have to

    time_taken = list() #  time taken for finding similar users for an user..

    

    # we create rows, cols, and data lists.., which can be used to create sparse matrices

    rows, cols, data = list(), list(), list()

    if verbose: print("Computing top",top,"similarities for each user..")

    

    start = datetime.now()

    temp = 0

    

    for row in row_ind[:top] if compute_for_few else row_ind:

        temp = temp+1

        prev = datetime.now()

        

        # get the similarity row for this user with all other users

        sim = cosine_similarity(sparse_matrix.getrow(row), sparse_matrix).ravel()

        # We will get only the top ''top'' most similar users and ignore rest of them..

        top_sim_ind = sim.argsort()[-top:]

        top_sim_val = sim[top_sim_ind]

        

        # add them to our rows, cols and data

        rows.extend([row]*top)

        cols.extend(top_sim_ind)

        data.extend(top_sim_val)

        time_taken.append(datetime.now().timestamp() - prev.timestamp())

        if verbose:

            if temp%verb_for_n_rows == 0:

                print("computing done for {} users [  time elapsed : {}  ]"

                      .format(temp, datetime.now()-start))

            

        

    # lets create sparse matrix out of these and return it

    if verbose: print('Creating Sparse matrix from the computed similarities')

    #return rows, cols, data

    

    if draw_time_taken:

        plt.plot(time_taken, label = 'time taken for each user')

        plt.plot(np.cumsum(time_taken), label='Total time')

        plt.legend(loc='best')

        plt.xlabel('User')

        plt.ylabel('Time (seconds)')

        plt.show()

        

    return sparse.csr_matrix((data, (rows, cols)), shape=(no_of_users, no_of_users)), time_taken      
start = datetime.now()

u_u_sim_sparse, _ = compute_user_similarity(train_sparse_matrix, compute_for_few=True, top = 100,

                                                     verbose=True)

print("-"*100)

print("Time taken :",datetime.now()-start)
from datetime import datetime

from sklearn.decomposition import TruncatedSVD



start = datetime.now()



# initilaize the algorithm with some parameters..

# All of them are default except n_components. n_itr is for Randomized SVD solver.

netflix_svd = TruncatedSVD(n_components=500, algorithm='randomized', random_state=15)

trunc_svd = netflix_svd.fit_transform(train_sparse_matrix)



print(datetime.now()-start)
expl_var = np.cumsum(netflix_svd.explained_variance_ratio_)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(.5))



ax1.set_ylabel("Variance Explained", fontsize=15)

ax1.set_xlabel("# Latent Facors", fontsize=15)

ax1.plot(expl_var)

# annote some (latentfactors, expl_var) to make it clear

ind = [1, 2,4,8,20, 60, 100, 200, 300, 400, 500]

ax1.scatter(x = [i-1 for i in ind], y = expl_var[[i-1 for i in ind]], c='#ff3300')

for i in ind:

    ax1.annotate(s ="({}, {})".format(i,  np.round(expl_var[i-1], 2)), xy=(i-1, expl_var[i-1]),

                xytext = ( i+20, expl_var[i-1] - 0.01), fontweight='bold')



change_in_expl_var = [expl_var[i+1] - expl_var[i] for i in range(len(expl_var)-1)]

ax2.plot(change_in_expl_var)







ax2.set_ylabel("Gain in Var_Expl with One Additional LF", fontsize=10)

ax2.yaxis.set_label_position("right")

ax2.set_xlabel("# Latent Facors", fontsize=20)



plt.show()
for i in ind:

    print("({}, {})".format(i, np.round(expl_var[i-1], 2)))
# Let's project our Original U_M matrix into into 500 Dimensional space...

start = datetime.now()

trunc_matrix = train_sparse_matrix.dot(netflix_svd.components_.T)

print(datetime.now()- start)
type(trunc_matrix), trunc_matrix.shape
if not os.path.isfile('trunc_sparse_matrix.npz'):

    # create that sparse sparse matrix

    trunc_sparse_matrix = sparse.csr_matrix(trunc_matrix)

    # Save this truncated sparse matrix for later usage..

    sparse.save_npz('trunc_sparse_matrix', trunc_sparse_matrix)

else:

    trunc_sparse_matrix = sparse.load_npz('trunc_sparse_matrix.npz')
trunc_sparse_matrix.shape
start = datetime.now()

trunc_u_u_sim_matrix, _ = compute_user_similarity(trunc_sparse_matrix, compute_for_few=True, top=50, verbose=True, 

                                                 verb_for_n_rows=10)

print("-"*50)

print("time:",datetime.now()-start)
start = datetime.now()

if not os.path.isfile('m_m_sim_sparse.npz'):

    print("It seems you don't have that file. Computing movie_movie similarity...")

    start = datetime.now()

    m_m_sim_sparse = cosine_similarity(X=train_sparse_matrix.T, dense_output=False)

    print("Done..")

    # store this sparse matrix in disk before using it. For future purposes.

    print("Saving it to disk without the need of re-computing it again.. ")

    sparse.save_npz("m_m_sim_sparse.npz", m_m_sim_sparse)

    print("Done..")

else:

    print("It is there, We will get it.")

    m_m_sim_sparse = sparse.load_npz("m_m_sim_sparse.npz")

    print("Done ...")



print("It's a ",m_m_sim_sparse.shape," dimensional matrix")



print(datetime.now() - start)
m_m_sim_sparse.shape
movie_ids = np.unique(m_m_sim_sparse.nonzero()[1])
start = datetime.now()

similar_movies = dict()

for movie in movie_ids:

    # get the top similar movies and store them in the dictionary

    sim_movies = m_m_sim_sparse[movie].toarray().ravel().argsort()[::-1][1:]

    similar_movies[movie] = sim_movies[:100]

print(datetime.now() - start)



# just testing similar movies for movie_15

similar_movies[15]
# First Let's load the movie details into soe dataframe..

# movie details are in 'netflix/movie_titles.csv'



movie_titles = pd.read_csv("data_folder/movie_titles.csv", sep=',', header = None,

                           names=['movie_id', 'year_of_release', 'title'], verbose=True,

                      index_col = 'movie_id', encoding = "ISO-8859-1")



movie_titles.head()
mv_id = 67



print("\nMovie ----->",movie_titles.loc[mv_id].values[1])



print("\nIt has {} Ratings from users.".format(train_sparse_matrix[:,mv_id].getnnz()))



print("\nWe have {} movies which are similarto this  and we will get only top most..".format(m_m_sim_sparse[:,mv_id].getnnz()))
similarities = m_m_sim_sparse[mv_id].toarray().ravel()



similar_indices = similarities.argsort()[::-1][1:]



similarities[similar_indices]



sim_indices = similarities.argsort()[::-1][1:] # It will sort and reverse the array and ignore its similarity (ie.,1)

                                               # and return its indices(movie_ids)
plt.plot(similarities[sim_indices], label='All the ratings')

plt.plot(similarities[sim_indices[:100]], label='top 100 similar movies')

plt.title("Similar Movies of {}(movie_id)".format(mv_id), fontsize=20)

plt.xlabel("Movies (Not Movie_Ids)", fontsize=15)

plt.ylabel("Cosine Similarity",fontsize=15)

plt.legend()

plt.show()
movie_titles.loc[sim_indices[:10]]
def get_sample_sparse_matrix(sparse_matrix, no_users, no_movies, path, verbose = True):

    """

        It will get it from the ''path'' if it is present  or It will create 

        and store the sampled sparse matrix in the path specified.

    """



    # get (row, col) and (rating) tuple from sparse_matrix...

    row_ind, col_ind, ratings = sparse.find(sparse_matrix)

    users = np.unique(row_ind)

    movies = np.unique(col_ind)



    print("Original Matrix : (users, movies) -- ({} {})".format(len(users), len(movies)))

    print("Original Matrix : Ratings -- {}\n".format(len(ratings)))



    # It just to make sure to get same sample everytime we run this program..

    # and pick without replacement....

    np.random.seed(15)

    sample_users = np.random.choice(users, no_users, replace=False)

    sample_movies = np.random.choice(movies, no_movies, replace=False)

    # get the boolean mask or these sampled_items in originl row/col_inds..

    mask = np.logical_and( np.isin(row_ind, sample_users),

                      np.isin(col_ind, sample_movies) )

    

    sample_sparse_matrix = sparse.csr_matrix((ratings[mask], (row_ind[mask], col_ind[mask])),

                                             shape=(max(sample_users)+1, max(sample_movies)+1))



    if verbose:

        print("Sampled Matrix : (users, movies) -- ({} {})".format(len(sample_users), len(sample_movies)))

        print("Sampled Matrix : Ratings --", format(ratings[mask].shape[0]))



    print('Saving it into disk for furthur usage..')

    # save it into disk

    sparse.save_npz(path, sample_sparse_matrix)

    if verbose:

            print('Done..\n')

    

    return sample_sparse_matrix
start = datetime.now()

path = "sample/small/sample_train_sparse_matrix.npz"

if os.path.isfile(path):

    print("It is present in your pwd, getting it from disk....")

    # just get it from the disk instead of computing it

    sample_train_sparse_matrix = sparse.load_npz(path)

    print("DONE..")

else: 

    # get 10k users and 1k movies from available data 

    sample_train_sparse_matrix = get_sample_sparse_matrix(train_sparse_matrix, no_users=10000, no_movies=1000,

                                             path = path)



print(datetime.now() - start)
start = datetime.now()



path = "sample/small/sample_test_sparse_matrix.npz"

if os.path.isfile(path):

    print("It is present in your pwd, getting it from disk....")

    # just get it from the disk instead of computing it

    sample_test_sparse_matrix = sparse.load_npz(path)

    print("DONE..")

else:

    # get 5k users and 500 movies from available data 

    sample_test_sparse_matrix = get_sample_sparse_matrix(test_sparse_matrix, no_users=5000, no_movies=500,

                                                 path = "sample/small/sample_test_sparse_matrix.npz")

print(datetime.now() - start)
sample_train_averages = dict()
# get the global average of ratings in our train set.

global_average = sample_train_sparse_matrix.sum()/sample_train_sparse_matrix.count_nonzero()

sample_train_averages['global'] = global_average

sample_train_averages
sample_train_averages['user'] = get_average_ratings(sample_train_sparse_matrix, of_users=True)

print('\nAverage rating of user 1515220 :',sample_train_averages['user'][1515220])
sample_train_averages['movie'] =  get_average_ratings(sample_train_sparse_matrix, of_users=False)

print('\n AVerage rating of movie 15153 :',sample_train_averages['movie'][15153])
print('\n No of ratings in Our Sampled train matrix is : {}\n'.format(sample_train_sparse_matrix.count_nonzero()))

print('\n No of ratings in Our Sampled test  matrix is : {}\n'.format(sample_test_sparse_matrix.count_nonzero()))
# get users, movies and ratings from our samples train sparse matrix

sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(sample_train_sparse_matrix)
############################################################

# It took me almost 10 hours to prepare this train dataset.#

############################################################

start = datetime.now()

if os.path.isfile('sample/small/reg_train.csv'):

    print("File already exists you don't have to prepare again..." )

else:

    print('preparing {} tuples for the dataset..\n'.format(len(sample_train_ratings)))

    with open('sample/small/reg_train.csv', mode='w') as reg_data_file:

        count = 0

        for (user, movie, rating)  in zip(sample_train_users, sample_train_movies, sample_train_ratings):

            st = datetime.now()

        #     print(user, movie)    

            #--------------------- Ratings of "movie" by similar users of "user" ---------------------

            # compute the similar Users of the "user"        

            user_sim = cosine_similarity(sample_train_sparse_matrix[user], sample_train_sparse_matrix).ravel()

            top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.

            # get the ratings of most similar users for this movie

            top_ratings = sample_train_sparse_matrix[top_sim_users, movie].toarray().ravel()

            # we will make it's length "5" by adding movie averages to .

            top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])

            top_sim_users_ratings.extend([sample_train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))

        #     print(top_sim_users_ratings, end=" ")    





            #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------

            # compute the similar movies of the "movie"        

            movie_sim = cosine_similarity(sample_train_sparse_matrix[:,movie].T, sample_train_sparse_matrix.T).ravel()

            top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.

            # get the ratings of most similar movie rated by this user..

            top_ratings = sample_train_sparse_matrix[user, top_sim_movies].toarray().ravel()

            # we will make it's length "5" by adding user averages to.

            top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])

            top_sim_movies_ratings.extend([sample_train_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 

        #     print(top_sim_movies_ratings, end=" : -- ")



            #-----------------prepare the row to be stores in a file-----------------#

            row = list()

            row.append(user)

            row.append(movie)

            # Now add the other features to this data...

            row.append(sample_train_averages['global']) # first feature

            # next 5 features are similar_users "movie" ratings

            row.extend(top_sim_users_ratings)

            # next 5 features are "user" ratings for similar_movies

            row.extend(top_sim_movies_ratings)

            # Avg_user rating

            row.append(sample_train_averages['user'][user])

            # Avg_movie rating

            row.append(sample_train_averages['movie'][movie])



            # finalley, The actual Rating of this user-movie pair...

            row.append(rating)

            count = count + 1



            # add rows to the file opened..

            reg_data_file.write(','.join(map(str, row)))

            reg_data_file.write('\n')        

            if (count)%10000 == 0:

                # print(','.join(map(str, row)))

                print("Done for {} rows----- {}".format(count, datetime.now() - start))





print(datetime.now() - start)
reg_train = pd.read_csv('sample/small/reg_train.csv', names = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5','smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating'], header=None)

reg_train.head()
# get users, movies and ratings from the Sampled Test 

sample_test_users, sample_test_movies, sample_test_ratings = sparse.find(sample_test_sparse_matrix)
sample_train_averages['global']
start = datetime.now()



if os.path.isfile('sample/small/reg_test.csv'):

    print("It is already created...")

else:



    print('preparing {} tuples for the dataset..\n'.format(len(sample_test_ratings)))

    with open('sample/small/reg_test.csv', mode='w') as reg_data_file:

        count = 0 

        for (user, movie, rating)  in zip(sample_test_users, sample_test_movies, sample_test_ratings):

            st = datetime.now()



        #--------------------- Ratings of "movie" by similar users of "user" ---------------------

            #print(user, movie)

            try:

                # compute the similar Users of the "user"        

                user_sim = cosine_similarity(sample_train_sparse_matrix[user], sample_train_sparse_matrix).ravel()

                top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.

                # get the ratings of most similar users for this movie

                top_ratings = sample_train_sparse_matrix[top_sim_users, movie].toarray().ravel()

                # we will make it's length "5" by adding movie averages to .

                top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])

                top_sim_users_ratings.extend([sample_train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))

                # print(top_sim_users_ratings, end="--")



            except (IndexError, KeyError):

                # It is a new User or new Movie or there are no ratings for given user for top similar movies...

                ########## Cold STart Problem ##########

                top_sim_users_ratings.extend([sample_train_averages['global']]*(5 - len(top_sim_users_ratings)))

                #print(top_sim_users_ratings)

            except:

                print(user, movie)

                # we just want KeyErrors to be resolved. Not every Exception...

                raise







            #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------

            try:

                # compute the similar movies of the "movie"        

                movie_sim = cosine_similarity(sample_train_sparse_matrix[:,movie].T, sample_train_sparse_matrix.T).ravel()

                top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.

                # get the ratings of most similar movie rated by this user..

                top_ratings = sample_train_sparse_matrix[user, top_sim_movies].toarray().ravel()

                # we will make it's length "5" by adding user averages to.

                top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])

                top_sim_movies_ratings.extend([sample_train_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 

                #print(top_sim_movies_ratings)

            except (IndexError, KeyError):

                #print(top_sim_movies_ratings, end=" : -- ")

                top_sim_movies_ratings.extend([sample_train_averages['global']]*(5-len(top_sim_movies_ratings)))

                #print(top_sim_movies_ratings)

            except :

                raise



            #-----------------prepare the row to be stores in a file-----------------#

            row = list()

            # add usser and movie name first

            row.append(user)

            row.append(movie)

            row.append(sample_train_averages['global']) # first feature

            #print(row)

            # next 5 features are similar_users "movie" ratings

            row.extend(top_sim_users_ratings)

            #print(row)

            # next 5 features are "user" ratings for similar_movies

            row.extend(top_sim_movies_ratings)

            #print(row)

            # Avg_user rating

            try:

                row.append(sample_train_averages['user'][user])

            except KeyError:

                row.append(sample_train_averages['global'])

            except:

                raise

            #print(row)

            # Avg_movie rating

            try:

                row.append(sample_train_averages['movie'][movie])

            except KeyError:

                row.append(sample_train_averages['global'])

            except:

                raise

            #print(row)

            # finalley, The actual Rating of this user-movie pair...

            row.append(rating)

            #print(row)

            count = count + 1



            # add rows to the file opened..

            reg_data_file.write(','.join(map(str, row)))

            #print(','.join(map(str, row)))

            reg_data_file.write('\n')        

            if (count)%1000 == 0:

                #print(','.join(map(str, row)))

                print("Done for {} rows----- {}".format(count, datetime.now() - start))

    print("",datetime.now() - start)  
reg_test_df = pd.read_csv('sample/small/reg_test.csv', names = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',

                                                          'smr1', 'smr2', 'smr3', 'smr4', 'smr5',

                                                          'UAvg', 'MAvg', 'rating'], header=None)

reg_test_df.head(4)
from surprise import Reader, Dataset
# It is to specify how to read the dataframe.

# for our dataframe, we don't have to specify anything extra..

reader = Reader(rating_scale=(1,5))



# create the traindata from the dataframe...

train_data = Dataset.load_from_df(reg_train[['user', 'movie', 'rating']], reader)



# build the trainset from traindata.., It is of dataset format from surprise library..

trainset = train_data.build_full_trainset() 
testset = list(zip(reg_test_df.user.values, reg_test_df.movie.values, reg_test_df.rating.values))

testset[:3]
models_evaluation_train = dict()

models_evaluation_test = dict()



models_evaluation_train, models_evaluation_test
# to get rmse and mape given actual and predicted ratings..

def get_error_metrics(y_true, y_pred):

    rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))

    mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100

    return rmse, mape



###################################################################

###################################################################

def run_xgboost(algo,  x_train, y_train, x_test, y_test, verbose=True):

    """

    It will return train_results and test_results

    """

    

    # dictionaries for storing train and test results

    train_results = dict()

    test_results = dict()

    

    

    # fit the model

    print('Training the model..')

    start =datetime.now()

    algo.fit(x_train, y_train, eval_metric = 'rmse')

    print('Done. Time taken : {}\n'.format(datetime.now()-start))

    print('Done \n')



    # from the trained model, get the predictions....

    print('Evaluating the model with TRAIN data...')

    start =datetime.now()

    y_train_pred = algo.predict(x_train)

    # get the rmse and mape of train data...

    rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)

    

    # store the results in train_results dictionary..

    train_results = {'rmse': rmse_train,

                    'mape' : mape_train,

                    'predictions' : y_train_pred}

    

    #######################################

    # get the test data predictions and compute rmse and mape

    print('Evaluating Test data')

    y_test_pred = algo.predict(x_test) 

    rmse_test, mape_test = get_error_metrics(y_true=y_test.values, y_pred=y_test_pred)

    # store them in our test results dictionary.

    test_results = {'rmse': rmse_test,

                    'mape' : mape_test,

                    'predictions':y_test_pred}

    if verbose:

        print('\nTEST DATA')

        print('-'*30)

        print('RMSE : ', rmse_test)

        print('MAPE : ', mape_test)

        

    # return these train and test results...

    return train_results, test_results

    
# it is just to makesure that all of our algorithms should produce same results

# everytime they run...



my_seed = 15

random.seed(my_seed)

np.random.seed(my_seed)



##########################################################

# get  (actual_list , predicted_list) ratings given list 

# of predictions (prediction is a class in Surprise).    

##########################################################

def get_ratings(predictions):

    actual = np.array([pred.r_ui for pred in predictions])

    pred = np.array([pred.est for pred in predictions])

    

    return actual, pred



################################################################

# get ''rmse'' and ''mape'' , given list of prediction objecs 

################################################################

def get_errors(predictions, print_them=False):



    actual, pred = get_ratings(predictions)

    rmse = np.sqrt(np.mean((pred - actual)**2))

    mape = np.mean(np.abs(pred - actual)/actual)



    return rmse, mape*100



##################################################################################

# It will return predicted ratings, rmse and mape of both train and test data   #

##################################################################################

def run_surprise(algo, trainset, testset, verbose=True): 

    '''

        return train_dict, test_dict

    

        It returns two dictionaries, one for train and the other is for test

        Each of them have 3 key-value pairs, which specify ''rmse'', ''mape'', and ''predicted ratings''.

    '''

    start = datetime.now()

    # dictionaries that stores metrics for train and test..

    train = dict()

    test = dict()

    

    # train the algorithm with the trainset

    st = datetime.now()

    print('Training the model...')

    algo.fit(trainset)

    print('Done. time taken : {} \n'.format(datetime.now()-st))

    

    # ---------------- Evaluating train data--------------------#

    st = datetime.now()

    print('Evaluating the model with train data..')

    # get the train predictions (list of prediction class inside Surprise)

    train_preds = algo.test(trainset.build_testset())

    # get predicted ratings from the train predictions..

    train_actual_ratings, train_pred_ratings = get_ratings(train_preds)

    # get ''rmse'' and ''mape'' from the train predictions.

    train_rmse, train_mape = get_errors(train_preds)

    print('time taken : {}'.format(datetime.now()-st))

    

    if verbose:

        print('-'*15)

        print('Train Data')

        print('-'*15)

        print("RMSE : {}\n\nMAPE : {}\n".format(train_rmse, train_mape))

    

    #store them in the train dictionary

    if verbose:

        print('adding train results in the dictionary..')

    train['rmse'] = train_rmse

    train['mape'] = train_mape

    train['predictions'] = train_pred_ratings

    

    #------------ Evaluating Test data---------------#

    st = datetime.now()

    print('\nEvaluating for test data...')

    # get the predictions( list of prediction classes) of test data

    test_preds = algo.test(testset)

    # get the predicted ratings from the list of predictions

    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)

    # get error metrics from the predicted and actual ratings

    test_rmse, test_mape = get_errors(test_preds)

    print('time taken : {}'.format(datetime.now()-st))

    

    if verbose:

        print('-'*15)

        print('Test Data')

        print('-'*15)

        print("RMSE : {}\n\nMAPE : {}\n".format(test_rmse, test_mape))

    # store them in test dictionary

    if verbose:

        print('storing the test results in test dictionary...')

    test['rmse'] = test_rmse

    test['mape'] = test_mape

    test['predictions'] = test_pred_ratings

    

    print('\n'+'-'*45)

    print('Total time taken to run this algorithm :', datetime.now() - start)

    

    # return two dictionaries train and test

    return train, test
import xgboost as xgb
# prepare Train data

x_train = reg_train.drop(['user','movie','rating'], axis=1)

y_train = reg_train['rating']



# Prepare Test data

x_test = reg_test_df.drop(['user','movie','rating'], axis=1)

y_test = reg_test_df['rating']



# initialize Our first XGBoost model...

first_xgb = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15, n_estimators=100)

train_results, test_results = run_xgboost(first_xgb, x_train, y_train, x_test, y_test)



# store the results in models_evaluations dictionaries

models_evaluation_train['first_algo'] = train_results

models_evaluation_test['first_algo'] = test_results



xgb.plot_importance(first_xgb)

plt.show()
from surprise import BaselineOnly 


# options are to specify.., how to compute those user and item biases

bsl_options = {'method': 'sgd',

               'learning_rate': .001

               }

bsl_algo = BaselineOnly(bsl_options=bsl_options)

# run this algorithm.., It will return the train and test results..

bsl_train_results, bsl_test_results = run_surprise(my_bsl_algo, trainset, testset, verbose=True)





# Just store these error metrics in our models_evaluation datastructure

models_evaluation_train['bsl_algo'] = bsl_train_results 

models_evaluation_test['bsl_algo'] = bsl_test_results
# add our baseline_predicted value as our feature..

reg_train['bslpr'] = models_evaluation_train['bsl_algo']['predictions']

reg_train.head(2) 
# add that baseline predicted ratings with Surprise to the test data as well

reg_test_df['bslpr']  = models_evaluation_test['bsl_algo']['predictions']



reg_test_df.head(2)
# prepare train data

x_train = reg_train.drop(['user', 'movie','rating'], axis=1)

y_train = reg_train['rating']



# Prepare Test data

x_test = reg_test_df.drop(['user','movie','rating'], axis=1)

y_test = reg_test_df['rating']



# initialize Our first XGBoost model...

xgb_bsl = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15, n_estimators=100)

train_results, test_results = run_xgboost(xgb_bsl, x_train, y_train, x_test, y_test)



# store the results in models_evaluations dictionaries

models_evaluation_train['xgb_bsl'] = train_results

models_evaluation_test['xgb_bsl'] = test_results



xgb.plot_importance(xgb_bsl)

plt.show()

from surprise import KNNBaseline
# we specify , how to compute similarities and what to consider with sim_options to our algorithm

sim_options = {'user_based' : True,

               'name': 'pearson_baseline',

               'shrinkage': 100,

               'min_support': 2

              } 

# we keep other parameters like regularization parameter and learning_rate as default values.

bsl_options = {'method': 'sgd'} 



knn_bsl_u = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)

knn_bsl_u_train_results, knn_bsl_u_test_results = run_surprise(knn_bsl_u, trainset, testset, verbose=True)



# Just store these error metrics in our models_evaluation datastructure

models_evaluation_train['knn_bsl_u'] = knn_bsl_u_train_results 

models_evaluation_test['knn_bsl_u'] = knn_bsl_u_test_results

# we specify , how to compute similarities and what to consider with sim_options to our algorithm



# 'user_based' : Fals => this considers the similarities of movies instead of users



sim_options = {'user_based' : False,

               'name': 'pearson_baseline',

               'shrinkage': 100,

               'min_support': 2

              } 

# we keep other parameters like regularization parameter and learning_rate as default values.

bsl_options = {'method': 'sgd'}





knn_bsl_m = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)



knn_bsl_m_train_results, knn_bsl_m_test_results = run_surprise(knn_bsl_m, trainset, testset, verbose=True)



# Just store these error metrics in our models_evaluation datastructure

models_evaluation_train['knn_bsl_m'] = knn_bsl_m_train_results 

models_evaluation_test['knn_bsl_m'] = knn_bsl_m_test_results

# add the predicted values from both knns to this dataframe

reg_train['knn_bsl_u'] = models_evaluation_train['knn_bsl_u']['predictions']

reg_train['knn_bsl_m'] = models_evaluation_train['knn_bsl_m']['predictions']



reg_train.head(2)
reg_test_df['knn_bsl_u'] = models_evaluation_test['knn_bsl_u']['predictions']

reg_test_df['knn_bsl_m'] = models_evaluation_test['knn_bsl_m']['predictions']



reg_test_df.head(2)
# prepare the train data....

x_train = reg_train.drop(['user', 'movie', 'rating'], axis=1)

y_train = reg_train['rating']



# prepare the train data....

x_test = reg_test_df.drop(['user','movie','rating'], axis=1)

y_test = reg_test_df['rating']



# declare the model

xgb_knn_bsl = xgb.XGBRegressor(n_jobs=10, random_state=15)

train_results, test_results = run_xgboost(xgb_knn_bsl, x_train, y_train, x_test, y_test)



# store the results in models_evaluations dictionaries

models_evaluation_train['xgb_knn_bsl'] = train_results

models_evaluation_test['xgb_knn_bsl'] = test_results





xgb.plot_importance(xgb_knn_bsl)

plt.show()
from surprise import SVD
# initiallize the model

svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)

svd_train_results, svd_test_results = run_surprise(svd, trainset, testset, verbose=True)



# Just store these error metrics in our models_evaluation data structure

models_evaluation_train['svd'] = svd_train_results 

models_evaluation_test['svd'] = svd_test_results
from surprise import SVDpp
# initiallize the model

svdpp = SVDpp(n_factors=50, random_state=15, verbose=True)

svdpp_train_results, svdpp_test_results = run_surprise(svdpp, trainset, testset, verbose=True)



# Just store these error metrics in our models_evaluation datastructure

models_evaluation_train['svdpp'] = svdpp_train_results 

models_evaluation_test['svdpp'] = svdpp_test_results

# add the predicted values from both knns to this dataframe

reg_train['svd'] = models_evaluation_train['svd']['predictions']

reg_train['svdpp'] = models_evaluation_train['svdpp']['predictions']



reg_train.head(2) 
reg_test_df['svd'] = models_evaluation_test['svd']['predictions']

reg_test_df['svdpp'] = models_evaluation_test['svdpp']['predictions']



reg_test_df.head(2) 
# prepare x_train and y_train

x_train = reg_train.drop(['user', 'movie', 'rating',], axis=1)

y_train = reg_train['rating']



# prepare test data

x_test = reg_test_df.drop(['user', 'movie', 'rating'], axis=1)

y_test = reg_test_df['rating']







xgb_final = xgb.XGBRegressor(n_jobs=10, random_state=15)

train_results, test_results = run_xgboost(xgb_final, x_train, y_train, x_test, y_test)



# store the results in models_evaluations dictionaries

models_evaluation_train['xgb_final'] = train_results

models_evaluation_test['xgb_final'] = test_results





xgb.plot_importance(xgb_final)

plt.show()
# prepare train data

x_train = reg_train[['knn_bsl_u', 'knn_bsl_m', 'svd', 'svdpp']]

y_train = reg_train['rating']



# test data

x_test = reg_test_df[['knn_bsl_u', 'knn_bsl_m', 'svd', 'svdpp']]

y_test = reg_test_df['rating']





xgb_all_models = xgb.XGBRegressor(n_jobs=10, random_state=15)

train_results, test_results = run_xgboost(xgb_all_models, x_train, y_train, x_test, y_test)



# store the results in models_evaluations dictionaries

models_evaluation_train['xgb_all_models'] = train_results

models_evaluation_test['xgb_all_models'] = test_results



xgb.plot_importance(xgb_all_models)

plt.show()
# Saving our TEST_RESULTS into a dataframe so that you don't have to run it again

pd.DataFrame(models_evaluation_test).to_csv('sample/small/small_sample_results.csv')

models = pd.read_csv('sample/small/small_sample_results.csv', index_col=0)

models.loc['rmse'].sort_values()
print("-"*100)

print("Total time taken to run this entire notebook ( with saved files) is :",datetime.now()-globalstart)