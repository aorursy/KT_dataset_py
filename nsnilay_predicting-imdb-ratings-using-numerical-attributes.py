import pandas as pd

import csv

import numpy as np

from sklearn import cross_validation



imdb_data = pd.read_csv('../input/movie_metadata.csv')

imdb_data.info()
imdb_data['num_critic_for_reviews']=imdb_data['num_critic_for_reviews'].fillna(0.0).astype(np.float32)

imdb_data['director_facebook_likes']=imdb_data['director_facebook_likes'].fillna(0.0).astype(np.float32)

imdb_data['actor_3_facebook_likes'] = imdb_data['actor_3_facebook_likes'].fillna(0.0).astype(np.float32)

imdb_data['actor_1_facebook_likes'] = imdb_data['actor_1_facebook_likes'].fillna(0.0).astype(np.float32)

imdb_data['gross'] = imdb_data['gross'].fillna(0.0).astype(np.float32)

imdb_data['num_voted_users'] = imdb_data['num_voted_users'].fillna(0.0).astype(np.float32)

imdb_data['cast_total_facebook_likes'] = imdb_data['cast_total_facebook_likes'].fillna(0.0).astype(np.float32)

imdb_data['num_user_for_reviews'] = imdb_data['num_user_for_reviews'].fillna(0.0).astype(np.float32)

imdb_data['facenumber_in_poster'] = imdb_data['facenumber_in_poster'].fillna(0.0).astype(np.float32)

imdb_data['actor_2_facebook_likes'] = imdb_data['actor_2_facebook_likes'].fillna(0.0).astype(np.float32)

imdb_data['budget'] = imdb_data['budget'].fillna(0.0).astype(np.float32)

imdb_data['movie_facebook_likes'] = imdb_data['movie_facebook_likes'].fillna(0.0).astype(np.float32)





imdb_data['imdb_score']=imdb_data['imdb_score'].fillna(0.0).astype(int)

imdb_data.info()
my=list(zip(imdb_data['director_facebook_likes'],imdb_data['actor_1_facebook_likes'],imdb_data['actor_2_facebook_likes'],imdb_data['actor_3_facebook_likes']))

u = np.array(my)

u

X=u[:,:-1]

y=u[:,-1]

X

y

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn import tree

clf=tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)

"DecisionTree",clf.score(X_test,y_test)
from sklearn.svm import SVC

clf=SVC(kernel="rbf")

clf.fit(X_train,y_train)

"SVM",clf.score(X_test,y_test)
from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()

clf.fit(X_train,y_train)

"NaiveBayes",clf.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

clf=RandomForestClassifier()

clf.fit(X_train,y_train)

"RandomForest",clf.score(X_test,y_test)



#After repeating Random Forest many times the average accuracy values comes out to be around 70%
clf=AdaBoostClassifier()

clf.fit(X_train,y_train)

"AdaBoost",clf.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier()

clf.fit(X_train,y_train)

"KNeighbors",clf.score(X_test,y_test)