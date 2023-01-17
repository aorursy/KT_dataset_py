import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors
# Seaborn advanced settings



sns.set(style='ticks',          # 'ticks', 'darkgrid'

        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'

        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'

        rc = {

           'figure.autolayout': True,

           'figure.figsize': (14, 8),

           'legend.frameon': True,

           'patch.linewidth': 2.0,

           'lines.markersize': 6,

           'lines.linewidth': 2.0,

           'font.size': 20,

           'legend.fontsize': 20,

           'axes.labelsize': 16,

           'axes.titlesize': 22,

           'axes.grid': True,

           'grid.color': '0.9',

           'grid.linestyle': '-',

           'grid.linewidth': 1.0,

           'xtick.labelsize': 20,

           'ytick.labelsize': 20,

           'xtick.major.size': 8,

           'ytick.major.size': 8,

           'xtick.major.pad': 10.0,

           'ytick.major.pad': 10.0,

           }

       )



plt.rcParams['image.cmap'] = 'viridis'
books = pd.read_csv("../input/bookcrossing-dataset/Books.csv", sep=";")

users = pd.read_csv("../input/bookcrossing-dataset/Users.csv", sep=";")

ratings = pd.read_csv("../input/bookcrossing-dataset/Ratings.csv", sep=";")
# Additional data cleaning

# There was probably no check for valid ISBN...

ratings["ISBN"] = ratings["ISBN"].apply(lambda x: x.strip().strip("\'").strip("\\").strip('\"').strip("\#").strip("("))
books.head()
users.head()
ratings.head()
# Group by and create basic additional users

user_groupby = ratings.groupby("User-ID")

book_groupby = ratings.groupby("ISBN")

average_user_rating = user_groupby["Rating"].mean()

number_of_ratings_by_user = user_groupby["Rating"].count()

average_book_rating = book_groupby["Rating"].mean()

number_of_book_ratings = book_groupby["Rating"].count()



average_user_rating.name = "avg_rating"

number_of_ratings_by_user.name = "N_ratings"

average_book_rating.name = "avg_rating"

number_of_book_ratings.name = "N_ratings"
# Merge with original dataframes

users = users.join(number_of_ratings_by_user, on="User-ID")

users = users.join(average_user_rating, on="User-ID")

books = books.join(number_of_book_ratings, on="ISBN")

books = books.join(average_book_rating, on="ISBN")



users["N_ratings"] = users["N_ratings"].fillna(0)

books["N_ratings"] = books["N_ratings"].fillna(0)



users["N_ratings"] = users["N_ratings"].astype("int64")

books["N_ratings"] = books["N_ratings"].astype("int64")
print(f"Out of {users.shape[0]} users only {users['N_ratings'].gt(0).sum(axis=0)} rated at least 1 book.")

print(f"Only {users['N_ratings'].gt(1).sum(axis=0)} rated at least 2 books.")

print(f"Only {users['N_ratings'].gt(9).sum(axis=0)} rated at least 10 books.")

print(f"Most active user rated {users['N_ratings'].max()} books.")

print()

print(f"Out of {books.shape[0]} books only {books['N_ratings'].gt(0).sum(axis=0)} are rated at least once.")

print(f"Only {books['N_ratings'].gt(1).sum(axis=0)} have at least 2 ratings.")

print(f"Only {books['N_ratings'].gt(9).sum(axis=0)} have at least 10 ratings.")

print(f"Most rated book was rated {books['N_ratings'].max()} times.")
users[users["N_ratings"].gt(0)].describe()
# Get the most rated book in the dataset

books[books["N_ratings"] == books["N_ratings"].max()]
# Get top 20 best rated books in our dataset

books.loc[books["N_ratings"] > 20].sort_values(by="avg_rating", ascending=False).head(20)
# Get all Harry Potter books and editions written by Rowling

books[books["Title"].str.contains("Harry Potter") & books["Author"].str.contains("Rowling")]
ratings["Rating"] = ratings["Rating"].astype("int8")
pd_matrix = pd.merge(books.loc[books["N_ratings"] > 20, "ISBN"], ratings, how="left", left_on="ISBN", right_on="ISBN").drop_duplicates()
pd_matrix
# Reshape so that ISBN is row index, User-ID is column index and values are ratings

pd_matrix = pd_matrix.pivot(index='ISBN', columns='User-ID', values='Rating').fillna(0).astype("int8")
pd_matrix
# Change to sparse matrix if we didn't have enough memory

matrix = csr_matrix(pd_matrix.values)
# Create a model

N_predicted_neighbours = 11

KNN = NearestNeighbors(metric='cosine', n_neighbors=N_predicted_neighbours, n_jobs=-1)
# Fit the model

KNN.fit(matrix)
# Predict

distances, indices = KNN.kneighbors(matrix)



# Note that we do not have to split the data to train, valid and test, as we only need to compute distances between current data
print("Index of first Harry Potter book is:", np.where(pd_matrix.index=="059035342X")[0][0])
selected_index = 4865
# Just check it once again

books.loc[books["ISBN"] == pd_matrix.index[selected_index], "Title"].values[0]
# Predictions



print(f"Because you liked {books.loc[books['ISBN'] == pd_matrix.index[indices[4865][0]], 'Title'].values[0]} you may like:")

print()

for i in range(1, N_predicted_neighbours):

    print(f"{books.loc[books['ISBN'] == pd_matrix.index[indices[4865][i]], 'Title'].values[0]} with distance {distances[4865][i]:.3f}.")
def recommend_similar_book(isbn, indices, ratings_matrix, books_table, N_recommendations=1, distances=None):

    """

    Recommends a book title.

    

    Parameters

    ----------

    ISBN: str

        ISBN of a book a user liked

    indices: np.array

        indices of ratings_matrix as predicted by KNN

    ratings_matrix: pd.DataFrame

        user-book-rating matrix with ratings as values

    N_recommendations: int (default 1)

        How many books to recommend?

    distances: np.array

        How distant are books from each other by KNN?

    """

    # TODO: This should be rather split in separate variables, this reads terribly

    print(f"Because you liked {books_table.loc[books_table['ISBN'] == ratings_matrix.index[indices[isbn][0]], 'Title'].values[0]} you may like:")

    print()

    for i in range(1, 1+N_recommendations):

        if distances:

            print(f"{books_table.loc[books_table['ISBN'] == ratings_matrix.index[indices[isbn][i]], 'Title'].values[0]} with distance {distances[isbn][i]:.3f}.")

        else:

            print(f"{books_table.loc[books_table['ISBN'] == ratings_matrix.index[indices[isbn][i]], 'Title'].values[0]}.")
recommend_similar_book(4865, indices, pd_matrix, books)
harry_potter_isbns = books.loc[books["Title"].str.contains("Harry Potter") & books["Author"].str.contains("Rowling"), "ISBN"].values

harry_potter_ratings = ratings.loc[ratings["ISBN"].isin(harry_potter_isbns)]
# Group by and create new features

user_groupby = harry_potter_ratings.groupby("User-ID")

average_user_rating = user_groupby["Rating"].mean()

number_of_ratings_by_user = user_groupby["Rating"].count()



average_user_rating.name = "HP_avg_rating"

number_of_ratings_by_user.name = "HP_N_ratings"
# Merge with the main dataframe

users = users.join(number_of_ratings_by_user, on="User-ID")

users = users.join(average_user_rating, on="User-ID")



users["N_ratings"] = users["N_ratings"].fillna(0)



users["N_ratings"] = users["N_ratings"].astype("int64")
# Get some statistics for those users who have read at least one book from the HP series

users.loc[users["HP_N_ratings"].gt(0)].describe()
plt.figure()

# sns.distplot(users.loc[users["HP_N_ratings"].gt(0)]["HP_avg_rating"], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) - 0.5

n, bins, patches = plt.hist(users.loc[users["HP_N_ratings"].gt(0), "HP_avg_rating"], bins=bins)

cm = plt.cm.get_cmap('RdYlGn')

bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]

col = bin_centers - min(bin_centers)

col /= max(col)

for c, p in zip(col, patches):

    plt.setp(p, 'facecolor', cm(c))

plt.title("Ratings of Harry Potter books")

plt.xlabel("Number of stars")

plt.ylabel("Number of ratings")

plt.show()
# Get users who read seven books of the series

users.loc[users["HP_N_ratings"]==7]



# Note that this selection does not have to mean, they have read all parts of the series, just that they rated seven editions.

# Nevertheless, as first testing data, it is probably OK.
# Transform User-ID to the index in our user-book-rating matrix

selected_matrix_indices = [pd_matrix.columns.get_loc(user_ID) for user_ID in users.loc[users["HP_N_ratings"]==7].sort_values(by="HP_avg_rating", ascending=False)["User-ID"].values]
KNN3 = NearestNeighbors(metric='cosine', n_neighbors=8, n_jobs=-1)
KNN3.fit(matrix.T[np.ix_(selected_matrix_indices)])
distances3, indices3 = KNN3.kneighbors(matrix.T[np.ix_(selected_matrix_indices)])
ind = np.argsort(indices3, axis=1)
sorted_distances = np.take_along_axis(distances3, ind, axis=1)
plt.figure()

ax = sns.heatmap(sorted_distances, linewidth=0.5, cmap="viridis")

plt.title("Distances between users; darker = closer")

plt.show()
# Create a model

KNN2 = NearestNeighbors(metric='cosine', n_neighbors=20, n_jobs=-1)
# Fit

KNN2.fit(matrix.T)
# Predict

distances2, indices2 = KNN2.kneighbors(matrix.T)
# Transform User-ID to the index in our user-book-rating matrix

pd_matrix.columns.get_loc(175003)
# Get most similar users

indices2[34620]
# Transform back and get User-ID of nearest neighbor

pd_matrix.columns[50133]
users.loc[users["User-ID"] == 252829]
def recommend_favourite_book_of_similar_user(userID, indices, ratings_matrix, users_table, books_table, ratings_table, N_recommendations=1, distances=None):

    """

    Recommends a book title based on favourite books of ten most similar users.

    

    The order of books is following:

    Take the most similar user, sort his books by rating,

    exclude everything the current predicted user already read.

    Output books one by one.

    If there is only a few books from the most similar user and

    we run out of books, take next similar user and output

    his favorite books in a similar fashion.

    

    Parameters

    ----------

    userID: int

        ID of a user we want a recommendation for

    indices: np.array

        indices of ratings_matrix as predicted by KNN

    ratings_matrix: pd.DataFrame

        user-book-rating matrix with ratings as values

    users_table: pd.DataFrame

        Information about users

    books_table: pd.DataFrame

        Information about books

    ratings_table: pd.DataFrame

        Information about ratings

    N_recommendations: int (default 1)

        How many books to recommend?

    distances: np.array

        How distant are books from each other by KNN?

    """

    selected_index = ratings_matrix.columns.get_loc(userID)

    already_read_book_isbns = list(ratings_table.loc[ratings_table["User-ID"] == userID, "ISBN"].values)

    not_read_books = ratings_table.loc[~ratings_table["ISBN"].isin(already_read_book_isbns)]

    books_to_recommend = list()

    for i in range(1,10):

        similar_user_index = indices[selected_index][i]

        similar_user_ID = ratings_matrix.columns[similar_user_index]

        possible_to_recommend = not_read_books.loc[not_read_books["User-ID"] == similar_user_ID]

        possible_to_recommend = possible_to_recommend.sort_values(by="Rating", ascending=False)

        for a, row in possible_to_recommend.iterrows():

            books_to_recommend.append(books_table.loc[books["ISBN"] == row["ISBN"], "Title"].values[0])

            if len(books_to_recommend) > N_recommendations-1:

                break

        if len(books_to_recommend) > N_recommendations-1:

            break

    print(f"Based on users who like similar books as you, you may like:")

    print()

    for book_name in books_to_recommend:

        print(book_name)
recommend_favourite_book_of_similar_user(175003, indices2, pd_matrix, users, books, ratings, N_recommendations=3, distances=distances2)