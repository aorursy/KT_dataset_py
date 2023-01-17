import numpy as np

import pandas as pd

from sklearn import cross_validation

from sklearn import tree

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt



datadir = "../input/movie_metadata.csv"

#df = pd.read_csv(datadir)

#print(df)



#process data



##extract relevant features

movie_features = pd.read_csv(datadir, usecols=['duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','genres','content_rating','budget','actor_2_facebook_likes','aspect_ratio','movie_facebook_likes','gross','num_user_for_reviews', 'num_critic_for_reviews','imdb_score']).dropna()

##extract gross

movie_gross = movie_features['gross']

##extract imdb rating and number for reviews (may consolidate later)

movie_imdb_rating = movie_features[['num_user_for_reviews', 'num_critic_for_reviews','imdb_score']]



##consolidate actor and movie likes and clean matrices

like_col_list = ['director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','movie_facebook_likes']

movie_features['like_totals'] = movie_features[like_col_list].sum(axis=1)

movie_features.drop(like_col_list, axis=1, inplace=True)

movie_features.drop(['gross','num_user_for_reviews', 'num_critic_for_reviews','imdb_score'], axis=1, inplace=True)



###remove string features for now cause it's complicated

movie_features.drop(['content_rating'], axis=1, inplace=True)

movie_features.drop(['genres'], axis=1, inplace=True)



##convert everything so far to a format readable by sklearn

movie_features_matrix = movie_features.reset_index().values

movie_gross_matrix = movie_gross.reset_index().values

movie_gross_array = []

for row in movie_gross_matrix:

    movie_gross_array += [row[1]]

#print(movie_features_matrix)

#print(movie_gross_array)



##convert genre tags into binary data

#genre_list = ['Action','Adventure','Fantasy','Sci-Fi','Thriller','Drama','Romance','Documentary','Comedy','Animation','Musical','Family','Mystery','Western','History','Sport','Crime']

#genre_enum = enumerate(genre_list)

#movie_genre_df = pd.DataFrame(columns=genre_list)

#print(movie_genre)

#for row in movie_genre_df:

#    for genre in genre_enum:

#        if genre in row[2]:

#            movie_genre_df.set_value(index, genre, 1)

#print()





# cross validate

X_train, X_test, y_train, y_test = cross_validation.train_test_split(movie_features_matrix, movie_gross_array, test_size=0.3, random_state=0)





# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=2)

regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(movie_features_matrix, movie_gross_array)

regr_2.fit(movie_features_matrix, movie_gross_array)

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)



x_array = []

y_array = []



for index in np.arange(0,len(y_test)):

    x_array += [index]

    y_array += [(y_test[index], y_1[index] ,y_2[index])]

#print(y_array)

#print(x_array)



for xe, ye in zip(x_array, y_array):

    plt.scatter([xe] * len(ye), ye)



#plt.xticks(x_array)

#plt.axes().set_xticklabels(['cat1', 'cat2'])

plt.savefig('t.png')



    



# fit classifier model

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)



print(clf.score(X_test, y_test))

#print(y_test)

#print(clf.predict(X_test))



#print(y_test-clf.predict(X_test))