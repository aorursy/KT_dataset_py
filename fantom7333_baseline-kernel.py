import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from collections import Counter
import sklearn

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
train = pd.read_csv('/kaggle/input/kpitmovies/train_data.csv')

test = pd.read_csv('/kaggle/input/kpitmovies/test_data.csv').drop(61)
train
directors = list(train[train['target']=="GOOD"].director)
basic_features = ['runtime', 'revenue','budget', 'vote_count', 'director', 'production_countries', 'production_companies', 'keywords','genres', 'cast','release_date']

basic_X = train[basic_features]

basic_y = train['target']
test_X = test[basic_features]
test_basic_X = pd.concat([basic_X, test_X], axis=0)
test_basic_X.release_date = pd.to_datetime(test_basic_X["release_date"])
test_basic_X["Year"] = test_basic_X["release_date"].apply(lambda x: x.year)
pd.options.display.max_rows = 1150
test_basic_X_copy_time = test_basic_X[0:768]
test_X_copy_time = test_basic_X[768:]
x_728 = test_X_copy_time[0:1]
test_basic_X_copy_time = pd.concat([test_basic_X_copy_time,x_728], axis=0)
test_X_copy_time = test_X_copy_time.drop(768)
# test_basic_X_copy_time = test_basic_X_copy_time.drop(627)
# test_X_copy_time = test_X_copy_time.drop(78)
def find_actors(cast):

    for_test = cast

    for_test = for_test[1:-1]

    for_test = for_test[1:-1]

    for_test = for_test[1:-1]

    for_test_arr = for_test.split("},")

    print(cast)

    for_test_arr_valid = []

    for i in for_test_arr:

        i = i.replace("{", "")

        for_test_arr_valid.append(i)

    for_test_arr_valid

    for_test_arr_split_name_values = []

    for i in for_test_arr_valid:

        i = i.split(",")

        try:

            i = i[5].split(":")[1].replace('"', "'").strip(" ").strip("'").strip('"')

            for_test_arr_split_name_values.append(i)

        except IndexError:

            for_test_arr_split_name_values.append("")

    print(for_test_arr_split_name_values)

    return for_test_arr_split_name_values
for i in test_basic_X_copy_time.index:

    test_basic_X_copy_time.cast[i] = find_actors(test_basic_X_copy_time.cast[i])

   
for i in test_X_copy_time.index:

    test_X_copy_time.cast[i] = find_actors(test_X_copy_time.cast[i])

    
actors_list = []
for i in test_basic_X_copy_time.index:

    actors_list.extend(test_basic_X_copy_time.cast[i])
for i in test_X_copy_time.index:

    actors_list.extend(test_X_copy_time.cast[i])
counter_actors_list = Counter(actors_list)
counter_actors_list = counter_actors_list.most_common(20)
for i in counter_actors_list:

    test_basic_X_copy_time[i[0]] = 1
for i in counter_actors_list:

    test_X_copy_time[i[0]] = 1
def one_or_zero_5_actors(data):

    for i in counter_actors_list:

        for a in data.index:

            if i[0] in list(data.cast[a]):

                print("Это i "+ i[0])

                print("Это a "+ str(data.cast[a]))

                data[i[0]][a] = 1

            else:

                print(a)

                print("всё плохо")

                data[i[0]][a]= 0
one_or_zero_5_actors(test_basic_X_copy_time)

one_or_zero_5_actors(test_X_copy_time)
genres_list = []
def parse_genres(data):

    for_test = data

    for_test = for_test.strip('"]').lstrip('"[').split(",")

    for i in for_test:

        var_genre = i.strip(" '")

        genres_list.append(var_genre)

    return True
for i in test_basic_X_copy_time.index:

    parse_genres(test_basic_X_copy_time.genres[i])
for i in test_X_copy_time.index:

    parse_genres(test_X_copy_time.genres[i])
counter_genres_list = Counter(genres_list)
genres_list = counter_genres_list.most_common(3)
genres_list
for i in counter_genres_list:

    test_basic_X_copy_time[i] = 1
for i in counter_genres_list:

    test_X_copy_time[i] = 1
def past_genres(data):

    for_test = data

    for_test = for_test.strip('"]').lstrip('"[').split(",")

    list_for_genres = []

    for i in for_test:

        var_genre = i.strip(" '")

        list_for_genres.append(var_genre)

    return list_for_genres
for i in test_basic_X_copy_time.index:

    test_basic_X_copy_time.genres[i] = past_genres(test_basic_X_copy_time.genres[i])
for i in test_X_copy_time.index:

    test_X_copy_time.genres[i] = past_genres(test_X_copy_time.genres[i])
def one_or_zero_10_genres(data):

    for i in genres_list:

        for a in data.index:

            if i[0] in list(data.genres[a]):

                print("Это i "+ i[0])

                print("Это a "+ str(data.genres[a]))

                data[i[0]][a] = 1

            else:

                print(a)

                print("всё плохо")

                data[i[0]][a]= 0
one_or_zero_10_genres(test_basic_X_copy_time)
one_or_zero_10_genres(test_X_copy_time)
companies_list = []
def parse_companies(data):

    for_test = data

    for_test = for_test.strip('"]').lstrip('"[').split(",")

    for i in for_test:

        var_genre = i.strip(" '").strip("{")

        companies_list.append(var_genre)

    return True
for i in test_basic_X_copy_time.index:

    parse_companies(test_basic_X_copy_time.production_companies[i])
for i in test_X_copy_time.index:

    parse_companies(test_X_copy_time.production_companies[i])
counter_companies_list = Counter(companies_list)
companies_list = counter_companies_list.most_common(10)
def past_companies(data):

    for_test = data

    for_test = for_test.strip('"]').lstrip('"[').split(",")

    list_for_companies = []

    for i in for_test:

        var_genre = i.strip(" '")

        list_for_companies.append(var_genre)

    return list_for_companies
for i in test_basic_X_copy_time.index:

    test_basic_X_copy_time.production_companies[i] = past_companies(test_basic_X_copy_time.production_companies[i])
for i in test_X_copy_time.index:

    test_X_copy_time.production_companies[i] = past_companies(test_X_copy_time.production_companies[i])
for i in companies_list:

    test_basic_X_copy_time[i[0]] = 1
for i in companies_list:

    test_X_copy_time[i[0]] = 1
def one_or_zero_12_companies(data):

    for i in companies_list:

        for a in data.index:

            if i[0] in list(data.production_companies[a]):

                print("Это i "+ i[0])

                print("Это a "+ str(data.production_companies[a]))

                data[i[0]][a] = 1

            else:

                print(a)

                print("всё плохо")

                data[i[0]][a]= 0
one_or_zero_12_companies(test_basic_X_copy_time)
one_or_zero_12_companies(test_X_copy_time)
keywords = []
def parse_keywords(data):

    for_test = data

    try:

        for_test = for_test.strip('"]').lstrip('"[').split(",")

    except: 

        return True

    for i in for_test:

        keyword = i.strip(" '").strip("{")

        keywords.append(keyword)

    return True
for i in test_basic_X_copy_time.index:

    parse_keywords(test_basic_X_copy_time.keywords[i])
for i in test_X_copy_time.index:

    parse_keywords(test_X_copy_time.keywords[i])
counter_keywords = Counter(keywords)
keywords = counter_keywords.most_common(15)
def past_keywords(data):

    for_test = data

    try:

        for_test = for_test.strip('"]').lstrip('"[').split(",")

    except:

        return []

    list_for_keywords = []

    for i in for_test: 

        keyword = i.strip(" '")    

        list_for_keywords.append(keyword)

    return list_for_keywords
for i in test_basic_X_copy_time.index:

    test_basic_X_copy_time.keywords[i] = past_keywords(test_basic_X_copy_time.keywords[i])
for i in test_X_copy_time.index:

    test_X_copy_time.keywords[i] = past_keywords(test_X_copy_time.keywords[i])
for i in keywords:

    test_basic_X_copy_time[i[0]] = 1
for i in keywords:

    test_X_copy_time[i[0]] = 1
def one_or_zero_20_keywords(data):

    for i in keywords:

        for a in data.index:

            if i[0] in list(data.keywords[a]):

                print("Это i "+ i[0])

                print("Это a "+ str(data.keywords[a]))

                data[i[0]][a] = 1

            else:

                print(i[0])

                print(data.keywords[a])

                print(a)

                print("всё плохо")

                data[i[0]][a]= 0
one_or_zero_20_keywords(test_basic_X_copy_time)
one_or_zero_20_keywords(test_X_copy_time)
# test_X_copy_time
countries = []
def parse_countries(data):

    try:

        data = data.split(",")

    except AttributeError:

        data = " "

    for i in data:

        print(i)

        countries.append(i.strip(" "))

    return True
for i in test_basic_X_copy_time.index:

    print(i)

    parse_countries(test_basic_X_copy_time.production_countries[i])
for i in test_X_copy_time.index:

    parse_countries(test_X_copy_time.production_countries[i])
countries_counter = Counter(countries)
countries = countries_counter.most_common(10)
def past_countries(data):

    countries_arr = []

    try:

        data = data.split(",")

    except AttributeError:

        data = " "

    for i in data:

        print(i)

        countries_arr.append(i.strip(" "))

    return countries_arr
for i in test_basic_X_copy_time.index:

    test_basic_X_copy_time.production_countries[i] = past_countries(test_basic_X_copy_time.production_countries[i])
for i in test_X_copy_time.index:

    test_X_copy_time.production_countries[i] = past_countries(test_X_copy_time.production_countries[i])
for i in countries:

    test_basic_X_copy_time[i[0]] = 1
for i in countries:

    test_X_copy_time[i[0]] = 1
def one_or_zero_25_countries(data):

    for i in countries:

        for a in data.index:

            if i[0] in list(data.production_countries[a]):

                print("Это i "+ i[0])

                print("Это a "+ str(data.production_countries[a]))

                data[i[0]][a] = 1

            else:

                print(a)

                print("всё плохо")

                data[i[0]][a]= 0
one_or_zero_25_countries(test_basic_X_copy_time)
one_or_zero_25_countries(test_X_copy_time)
directors_counter = Counter(directors)
directors = directors_counter.most_common(20)
directors
for i in directors:

    test_basic_X_copy_time[i[0]] = 1
for i in directors:

    test_X_copy_time[i[0]] = 1
def one_or_zero_10_directors(data):

    for i in directors:

        for a in data.index:

            if i[0]==(data.director[a]):

                print("Это i "+ i[0])

                print("Это a "+ str(data.director[a]))

                data[i[0]][a] = 1

            else:

                print(i[0])

                print(data.director[a])

                print(a)

                print("всё плохо")

                data[i[0]][a]= 0
one_or_zero_10_directors(test_basic_X_copy_time)
one_or_zero_10_directors(test_X_copy_time)
assert basic_X.shape[0] == basic_y.shape[0]
# for i in test_basic_X_copy_time.index:

#     print(type(test_basic_X_copy_time.cast[i]))

#     test_basic_X_copy_time.cast[i] = pd.Series(test_basic_X_copy_time.cast[i], index=range(len(test_basic_X_copy_time.cast[i])))
# for i in test_X_copy_time.index:

#     print(type(test_X_copy_time.cast[i]))

#     test_X_copy_time.cast[i] = pd.Series(test_X_copy_time.cast[i], index=range(len(test_X_copy_time.cast[i])))

test_basic_X = pd.concat([test_basic_X_copy_time, test_X_copy_time], axis=0)
# new_cast_dummies = pd.get_dummies(new_cast_DataFrame_X["cast"], prefix = "actor")
# language_dummies = pd.get_dummies(test_basic_X["language"], prefix = "lang")
# keywords_dummies =  pd.get_dummies(test_basic_X["keywords"], prefix = "keyword")
# cast_dummies = pd.get_dummies(test_basic_X["cast"], prefix = "actor")
# director_dummies = pd.get_dummies(test_basic_X["director"], prefix = "director")
# countries_dummies = pd.get_dummies(test_basic_X["production_countries"], prefix="country")
# companies = pd.get_dummies(test_basic_X["production_companies"], prefix="country")
# genres_dummies = pd.get_dummies(test_basic_X["genres"], prefix="country")
test_basic_X = test_basic_X.drop(['director', 'production_countries', 'production_companies', 'genres','cast', 'release_date', 'keywords'], axis=1)
basic_X = test_basic_X
# basic_X = pd.concat([test_basic_X, language_dummies], axis=1)
basic_X_copy = basic_X
basic_X = basic_X_copy[0:769]

test_X = basic_X_copy[769:]
basic_X_train, basic_X_validate, basic_y_train, basic_y_validate = train_test_split(basic_X, basic_y)
logres =  LogisticRegression()

logres.fit(basic_X_train, basic_y_train)

logres_y_pred = logres.predict(basic_X_validate)
print('Accuracy / train:\t',cross_val_score(logres, basic_X_train, basic_y_train).mean())

print('Accuracy / validation:  ',accuracy_score(logres_y_pred, basic_y_validate))
tree =  DecisionTreeClassifier()

tree.fit(basic_X_train, basic_y_train)

tree_y_pred = tree.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(tree, basic_X_train, basic_y_train).mean())

print('Accuracy / validation:  ', accuracy_score(basic_y_validate,tree_y_pred))
knn = KNeighborsClassifier()

knn.fit(basic_X_train, basic_y_train)

knn_y_pred = knn.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(knn, basic_X_train, basic_y_train).mean())

print('Accuracy / validation:  ', accuracy_score(basic_y_validate,knn_y_pred))
rand_forest = sklearn.ensemble.RandomForestClassifier()

rand_forest.fit(basic_X_train, basic_y_train)

rand_forest_pred = rand_forest.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(rand_forest, basic_X_train, basic_y_train).mean())

print('Accuracy / validation:  ', accuracy_score(basic_y_validate,rand_forest_pred))
rand_forest_pred
poll = VotingClassifier(estimators=[('lgrs', logres), ('tree', tree), ('knn', knn), ('rfor', sklearn.ensemble.RandomForestClassifier())], weights=[0.8, 0, 0, 1], voting='hard')

poll.fit(basic_X_train, basic_y_train)

poll_y_pred = poll.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(poll, basic_X_train, basic_y_train).mean())

print('Accuracy / validation:  ', accuracy_score(basic_y_validate,poll_y_pred))
# test = pd.read_csv('/kaggle/input/kpitmovies/test_data.csv').drop(61)
submission = pd.read_csv('/kaggle/input/kpitmovies/sample_submission.csv')
tree_prediction = tree.predict(test_X)
rand_forest_prediction = rand_forest.predict(test_X)
poll_prediction = poll.predict(test_X)
len(poll_prediction)
submission.movie_id = test.movie_id.values

submission.target = tree_prediction
submission.movie_id = test.movie_id.values

submission.target = poll_prediction
submission.movie_id = test.movie_id.values

submission.target = rand_forest_prediction
submission.target
poll_prediction
tree_prediction
submission.head()
submission.to_csv('poll_baseline.csv', index=False)