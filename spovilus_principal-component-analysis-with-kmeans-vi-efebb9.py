import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

import matplotlib.pyplot as plt # Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline 
movie = pd.read_csv('../input/movie_metadata.csv') # reads the csv and creates the dataframe called movie

movie_genres = pd.concat([movie,movie.genres.str.get_dummies("|")],axis=1)



#movie.head()

#print(movie.budget)

movie_filtered = movie[np.isfinite(movie['budget'])]

movie_filtered = movie_filtered[np.isfinite(movie_filtered['gross'])]

movie_filtered = movie_filtered[movie_filtered['country'] == 'USA']

movie_filtered = movie_filtered[movie_filtered['title_year'] >= 2007.0]

movie_filtered = movie_filtered.reset_index()

#print(movie.budget)

#print(movie.country)

#print(movie.title_year)

movie_filtered_genres = pd.concat([movie_filtered,movie_filtered.genres.str.get_dummies("|")],axis=1)

del movie_filtered_genres['index']

movie_filtered_genres.head()
movieGenre = []

movieGenre_number = []

for genre in movie_filtered.genres.str.get_dummies("|").columns.values:

    movieGenre.append(movie_filtered[movie_filtered_genres[genre] != 0])

    movieGenre[-1] = movieGenre[-1].reset_index()

    #movieGenre.to_csv("movie_filtered_" + genre + ".csv")

    str_list_filtered = [] # empty list to contain columns with strings (words)

    for colname, colvalue in movieGenre[-1].iteritems():

        if type(colvalue[1]) == str:

            str_list_filtered.append(colname)

    # Get to the numeric columns by inversion            

    num_list =(movieGenre[-1].columns.difference(str_list_filtered))

    movieGenre_number.append([movieGenre[-1][num_list],genre])

    del movieGenre_number[-1][0]['level_0']

    del movieGenre_number[-1][0]['index']
movie_filtered_Documentary = movie_filtered[movie_filtered_genres['Documentary'] == 1]

movie_filtered_Documentary.head()
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in movie.iteritems():

    if type(colvalue[1]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = movie.columns.difference(str_list)       



str_list_filtered = [] # empty list to contain columns with strings (words)

for colname, colvalue in movie_filtered.iteritems():

    if type(colvalue[1]) == str:

         str_list_filtered.append(colname)

# Get to the numeric columns by inversion            

num_list_filtered = movie_filtered.columns.difference(str_list_filtered) 
movie_num = movie[num_list]

#del movie # Get rid of movie df as we won't need it now

movie_num.head()



movie_num_filtered = movie_filtered[num_list_filtered]

#del movie # Get rid of movie df as we won't need it now

movie_num_filtered.head()
movie_num = movie_num.fillna(value=0, axis=1)

movie_num_filtered = movie_num_filtered.fillna(value=0, axis=1)
X = movie_num.values

# Data Normalization

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)



X_filtered = movie_num_filtered.values

# Data Normalization

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X_filtered)

#movie.plot(y= 'imdb_score', x ='duration',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Duration',figsize=(12,8))

#movie.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))
#movie_filtered.plot(y= 'imdb_score', x ='duration',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Duration',figsize=(12,8))

#movie_filtered.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Movie Features')

# Draw the heatmap using seaborn

sns.heatmap(movie_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Movie Features_filtered')

# Draw the heatmap using seaborn

sns.heatmap(movie_num_filtered.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)



i = 0

for genre2 in movieGenre_number: 

    i = i + 1

    f, ax = plt.subplots(figsize=(12, 10))

    plt.title('Pearson Correlation of Movie Features ' + str(i) + " " + genre2[1])

    # Draw the heatmap using seaborn

    sns.heatmap(genre2[0].astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)

# Calculating Eigenvectors and eigenvalues of Cov matirx

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 

plt.figure(figsize=(10, 5))

plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')

plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()
pca = PCA(n_components=9)

x_9d = pca.fit_transform(X_std)
plt.figure(figsize = (9,7))

plt.scatter(x_9d[:,0],x_9d[:,1], c='goldenrod',alpha=0.5)

plt.ylim(-10,30)

plt.show()
# Set a 3 KMeans clustering

kmeans = KMeans(n_clusters=3)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_9d)



# Define our own color map

LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_9d[:,0],x_9d[:,2], c= label_color, alpha=0.5) 

plt.show()
# Create a temp dataframe from our PCA projection data "x_9d"

df = pd.DataFrame(x_9d)

df = df[[0,1,2]] # only want to visualise relationships between first 3 projections

df['X_cluster'] = X_clustered
# Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data

sns.pairplot(df, hue='X_cluster', palette= 'Dark2', diag_kind='kde',size=1.85)