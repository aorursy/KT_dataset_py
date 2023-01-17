import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import chi2

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import KFold
def get_data():

    movies = pd.read_csv('../input/netflix-shows/netflix_titles.csv')

    

    # Replace NaN values with empty string

    movies.dropna(inplace=True)

    

    # Uniformisation of labels with same signification

    movies['rating'].replace(to_replace='PG-13', value='TV-PG', inplace=True)

    movies['rating'].replace(to_replace='PG', value='TV-PG', inplace=True)

    movies['rating'].replace(to_replace='TV-Y7-FV', value='TV-Y7', inplace=True)

    movies['rating'].replace(to_replace='G', value='TV-G', inplace=True)

    movies['rating'].replace(to_replace='NC-17', value='R', inplace=True)

    # Drop rows that don't have any labels (NR = UR = Unrated)

    movies.drop(movies[movies['rating']=='NR'].index, inplace=True)

    movies.drop(movies[movies['rating']==''].index, inplace=True)

    movies.drop(movies[movies['rating']=='UR'].index, inplace=True)

    

    

    

    #Drop useless labels

    movies.drop(movies[movies['rating']=='TV-G'].index, inplace=True)

    movies.drop(movies[movies['rating']=='TV-Y'].index, inplace=True)

    movies.drop(movies[movies['rating']=='TV-Y7'].index, inplace=True)





    #Drop useless colmns

    movies = movies[["type", "title", "description", "director", "cast", 'rating', "listed_in", "country"]]

    

    return movies
movies = get_data()



#Plot label repartition

fig = plt.figure(figsize=(8,6))

movies.groupby('rating').title.count().plot.bar(ylim=0)

plt.show()
#Convert labels to int and create a new column

movies['category_id'] = movies['rating'].factorize()[0]

#Get unique values

category_id_df = movies[["rating", "category_id"]].drop_duplicates().sort_values('category_id')

#Dicts associating rating and its int value

category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'rating']].values)

labels = movies.category_id
#Group all the columns together

movies ['txt'] = movies['type'] + " " + movies['title'] + " " + movies['description'] + " " + movies['listed_in'] 

movies['txt'] = movies['txt'] + " " + movies['director'] + " " + movies['cast'] + " " + movies["country"]

movies = movies[['rating', 'txt']]



#Cross val

mskf = KFold(n_splits = 15, random_state=0, shuffle = True)

acc = []



for train_index, test_index in mskf.split(movies['txt']) :

    X_train, X_test = movies['txt'].iloc[train_index], movies['txt'].iloc[test_index]

    y_train, y_test = movies['rating'].iloc[train_index], movies['rating'].iloc[test_index]

    

    #Preprocess

    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    

    #Fit

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    

    #score

    acc.append(clf.score(count_vect.transform(X_test.values), y_test.values))
plot_confusion_matrix(clf, count_vect.transform(X_test), np.array(y_test.values))

plt.show()
print("Accuracy with Naive Bayes: " + str(np.mean(acc) * 100) + "%")
movies = get_data().drop(['rating'], axis = 1)



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')



for rating, category_id in sorted(category_to_id.items()):

    print("######### '{}':".format(rating))

    for col in movies.columns :

        features = tfidf.fit_transform(movies[col]).toarray()

        features_chi2 = chi2(features, labels == category_id)

        mean = np.mean(features_chi2[0])

        print("    '{}':".format(col))

        print("        {}".format(mean*100))
movies = get_data()

#Group all the columns together

movies ['txt'] = movies['director'] + " " + movies['listed_in'] + " " + movies['type'] + " " + movies['country'] + " " + movies['cast'] 

movies = movies[['rating', 'txt']]
#Cross val

mskf = KFold(n_splits = 15, random_state=0, shuffle = True)

acc = []



for train_index, test_index in mskf.split(movies['txt']) :

    X_train, X_test = movies['txt'].iloc[train_index], movies['txt'].iloc[test_index]

    y_train, y_test = movies['rating'].iloc[train_index], movies['rating'].iloc[test_index]

    

    #Preprocess

    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    

    #Fit

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    

    #score

    acc.append(clf.score(count_vect.transform(X_test.values), y_test.values))
print("Accuracy with Naive Bayes: " + str(np.mean(acc) * 100) + "%")
plot_confusion_matrix(clf, count_vect.transform(X_test), np.array(y_test.values))

plt.show()
print("Accuracy with Naive Bayes on the whole dataset: " + str(clf.score(count_vect.transform(movies['txt'].values), movies['rating'].values) * 100) + "%")