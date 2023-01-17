# Author: Alaa Awad

# Project: https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset/kernels

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Author: Alaa Awad

data = pd.read_csv('../input/movie_metadata.csv')
data.head()
data.describe()
data.columns
print("Some imdb_score statistics\n")

print(data['imdb_score'].describe())

print("\nThe median of the imdb_score is: ", data['imdb_score'].median(axis = 0))
matplotlib.rcParams['figure.figsize'] = (9.0, 5.0)

scores = pd.DataFrame({"imdb score":data["imdb_score"]})

scores.hist(bins=20)
corr = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(7, 7))

sns.heatmap(corr, vmax=1, square=True)
cor_dict = corr['imdb_score'].to_dict()

del cor_dict['imdb_score']

print("List the numerical features decendingly by their correlation with IMDB score:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))
sns.regplot(x = 'num_voted_users', y = 'imdb_score', data = data, color = 'Green')
plt.figure(1)



f, axarr = plt.subplots(4, 3, figsize=(8, 8))

score = data.imdb_score.values



axarr[0, 0].scatter(data.num_voted_users.values, score)

axarr[0, 0].set_title('num_voted_users')

axarr[0, 1].scatter(data.num_user_for_reviews.values, score)

axarr[0, 1].set_title('num_user_for_reviews')

axarr[0, 2].scatter(data.duration.values, score)

axarr[0, 2].set_title('duration')

axarr[1, 0].scatter(data.movie_facebook_likes.values, score)

axarr[1, 0].set_title('movie_facebook_likes')

axarr[1, 1].scatter(data.title_year.values, score)

axarr[1, 1].set_title('title_year')

axarr[1, 2].scatter(data.gross.values, score)

axarr[1, 2].set_title('gross')



axarr[2, 0].scatter(np.log1p(data.director_facebook_likes.values), score)

axarr[2, 0].set_title('director_facebook_likes')

axarr[2, 1].scatter(np.log1p(data.cast_total_facebook_likes.values), score)

axarr[2, 1].set_title('cast_total_facebook_likes')

axarr[2, 2].scatter(np.log1p(data.facenumber_in_poster.values), score)

axarr[2, 2].set_title('facenumber_in_poster')





axarr[3, 0].scatter(np.log1p(data.actor_1_facebook_likes.values), score)

axarr[3, 0].set_title('actor_1_facebook_likes')

axarr[3, 1].scatter(np.log1p(data.actor_2_facebook_likes.values), score)

axarr[3, 1].set_title('actor_2_facebook_likes')

axarr[3, 2].scatter(np.log1p(data.actor_3_facebook_likes.values), score)

axarr[3, 2].set_title('actor_3_facebook_likes')





f.text(-0.01, 0.5, 'IMDB Score', va='center', rotation='vertical', fontsize = 12)

plt.tight_layout()

plt.show()
print(data.select_dtypes(include=['object']).columns.values)
plt.figure(figsize = (8, 4))

sns.boxplot(x = 'color', y = 'imdb_score',  data = data)

xt = plt.xticks(rotation=45)
plt.figure(figsize = (12, 6))

sns.countplot(x = 'language', data = data)

xt = plt.xticks(rotation=45)
plt.figure(figsize = (8, 4))

sns.countplot(x = 'content_rating', data = data)

xt = plt.xticks(rotation=45)
plt.figure(figsize = (8, 4))

sns.boxplot(x = 'content_rating', y = 'imdb_score',  data = data)

xt = plt.xticks(rotation=45)
sns.violinplot('content_rating', 'imdb_score', data = data)
sns.factorplot('content_rating', 'imdb_score', hue = 'color', estimator = np.mean, data = data, 

             size = 4.5, aspect = 1.4, order = ['G', 'PG', 'PG-13', 'NC-17', 'R', 'Not Rated', 'Unrated'])