import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/movie_metadata.csv')

df.info()
first_actors = set(df.actor_1_name.unique())

second_actors = set(df.actor_2_name.unique())

third_actors = set(df.actor_3_name.unique())

print('Those only in first name', len(first_actors - second_actors - third_actors))

print('Those only in second name', len(second_actors - first_actors - third_actors))

print('Those only in third name', len(third_actors - first_actors - second_actors))

# ----is it color or not

df.color = df.color.map({'Color': 1, ' Black and White':0})

# ---- Genres as on-off flags instead of strings

unique_genre_labels = set()

for genre_flags in df.genres.str.split('|').values:

    unique_genre_labels = unique_genre_labels.union(set(genre_flags))

for label in unique_genre_labels:

    df['Genre='+label] = df.genres.str.contains(label).astype(int)

df = df.drop('genres', axis=1)



# Titles are supposed to be unique right?

if len(df.drop_duplicates(subset=['movie_title',

                                  'title_year',

                                  'movie_imdb_link'])) < len(df):

    print('Duplicate Titles Exist')

    # Let's see these duplicates.

    duplicates = df[df.movie_title.map(df.movie_title.value_counts() > 1)]

    duplicates.sort('movie_title')[['movie_title', 'title_year']]

    # Looks like there are duplicates after all. Let's drop those.

    df = df.drop_duplicates(subset=['movie_title', 'title_year', 'movie_imdb_link'])

    # df.info()

counts = df.language.value_counts()

df.language = df.language.map(counts)

#df.language

count = df.country.value_counts()

df.country = df.country.map(count)

#df.country

counts = df.content_rating.value_counts()

df.content_rating = df.content_rating.map(counts)

#df.content_rating

#df.plot_keywords.head()

unique_words = set()

for wordlist in df.plot_keywords.str.split('|').values:

    if wordlist is not np.nan:

        unique_words = unique_words.union(set(wordlist))

plot_wordbag = list(unique_words)

for word in plot_wordbag:

    df['plot_has_' + word.replace(' ', '-')] = df.plot_keywords.str.contains(word).astype(float)

df = df.drop('plot_keywords', axis=1)

# Is anything left to be done other than imputing?

print(df.select_dtypes(include=['O']).columns)

# We replace director name with counts of movies they've done

df.director_name = df.director_name.map(df.director_name.value_counts())

# We replace actor names with the number of movies they appear in.

counts = pd.concat([df.actor_1_name, df.actor_2_name, df.actor_3_name]).value_counts()

#counts.head()

df.actor_1_name = df.actor_1_name.map(counts)

df.actor_2_name = df.actor_2_name.map(counts)

df.actor_3_name = df.actor_3_name.map(counts)

# I have no clue what to do with the title. I'll keep it for now in order to search by name

df = df.drop(['movie_imdb_link'], axis=1)

# Let's check if anything is left as object

df.select_dtypes(include=['O']).columns
df.shape
# hold your horses, we still need to fill those missing values.

new_style = {'grid': False}

matplotlib.rc('axes', **new_style)

plt.matshow(~df.isnull())

plt.title('Missing values in the data')
# Let's get those rows which are mostly incomplete. I suspect this was because of our

# new features being created from old ones which were null.

nullcount = df.isnull().sum(axis=1)

# Let's just keep those who have less than a hundred missing values

ndf = df.dropna(thresh=100)

print(ndf.shape, df.shape)

# Let's see those nulls again

plt.matshow(~ndf.isnull())

plt.title('Missing values in the data')
# We'll treat fillna as a regression / classification problem here.

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



def reg_class_fill(df, column, classifier):

    """Treat missing values as a classification / regresion problem"""

    ndf = df.dropna(subset=[col for col in df.columns if col != column])

    nullmask = ndf[column].isnull()

    train, test  = ndf[~nullmask], ndf[nullmask]

    train_x, train_y = train.drop(column, axis=1), train[column]

    classifier.fit(train_x, train_y)

    if len(test) > 0:

        test_x, test_y = test.drop(column, axis=1), test[column]

        values = classifier.predict(test_x)

        test_y = values

        new_x, new_y = pd.concat([train_x, test_x]), pd.concat([train_y, test_y])

        newdf = new_x[column] = new_y

        return newdf

    else:

        return ndf
r, c = KNeighborsRegressor, KNeighborsClassifier  # Regress or classify

title_encoder = LabelEncoder()

title_encoder.fit(ndf.movie_title)

ndf.movie_title = title_encoder.transform(ndf.movie_title)
print(ndf[ndf.columns[:25]].isnull().sum())
# Since our imputation will impact other imputations, we specify an order

# Typically we should do this independently and then combine the results, but meh for now

impute_order = [('director_name', c), ('title_year', c),

                ('actor_1_name', c), ('actor_2_name', c), ('actor_3_name', c),

                ('gross', r), ('budget', r), ('aspect_ratio', r),

                ('content_rating', r), ('num_critic_for_reviews', r)]

for col, classifier in impute_order:

    ndf = reg_class_fill(ndf, col, classifier())

    print(col, 'Done')
# Again we check for what else needs to be imputed.

ndf[ndf.columns[:25]].isnull().sum()
# Did we get everything?

ndf.isnull().sum().sum()
# YAY! We did indeed get everything, though it may not have been very good.

# Now we redo the movie title transformation for our searches.

titles = title_encoder.inverse_transform(ndf.movie_title)

#titles = [i.lower().strip() for i in titles]
# Give us 5 movies that you liked

def get_movies(names):

    movies = []

    for name in names:

        found = [i for i in titles if name.lower() in i.lower()]

        if len(found) > 0:

            movies.append(found[0])

            print(name, ': ', found, 'added', movies[-1], 'to movies')

        else:

            print(name, ': ', found)

    print('-'*10)

    print(movies)

    moviecodes = title_encoder.transform(movies)

    return moviecodes, movies

names = ['fight club', 'gump', # This one is Forrest Gump

                 'usual suspects', 'silence of the lambs']

moviecodes, movies = get_movies(names)
data = ndf.drop('movie_title', axis=1)

data = MinMaxScaler().fit_transform(data)
# We assume KNN's assumptions as valid and proceede to compute a distance_matrix

from sklearn.neighbors import KDTree

from collections import Counter
movies
titles
tree = KDTree(data, leaf_size=2)
def recommend(movies, tree, titles, data):

    """

    It is assumed that the movies are in order of decreasing like-able-ness

    Recommend movies on the basis of the KDTree generated.

    Return them in order of increasing distance form knowns.

    """

    titles = list(titles)

    length, recommendations = len(movies) + 1,[]

    

    for i, movie in enumerate(movies):

        weight = length - i

        dist, index = tree.query([data[titles.index(movie)]], k=3)

        for d, m in zip(dist[0], index[0]):

            recommendations.append((d*weight, titles[m]))

    recommendations.sort()

    # Stuff is reorganized by frequency.

    rec = [i[1].strip() for i in recommendations if i[1] not in movies]

    rec = [i[1] for i in sorted([(v, k) for k, v in Counter(rec).items()],

                                reverse=True)]

    return rec


rec = recommend(movies, tree, titles, data)



print('Rank | Movie')

print('-----|------')

fmt = '{}.   | {}'

for index, movie in enumerate(rec[:9]):

    print(fmt.format(index + 1, movie))
names = ['hesher', 'leaving las vegas'] # dedicated to A.S.

moviecodes, movies = get_movies(names)

rec = recommend(movies, tree, titles, data)

print('-'*50)

print('Recommending on the basis of the above movies')

print('-'*50)

print()

print('+-----|------')

print('|Rank | Movie')

print('+-----|------')

fmt = '|{}.   | {}'

for index, movie in enumerate(rec[:9]):

    print(fmt.format(index + 1, movie))

print('+-----|------')