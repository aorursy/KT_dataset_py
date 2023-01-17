#!pip install swifter

#!pip install -U scikit-learn

#!pip install smogn
# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



details = pd.read_json('/kaggle/input/imdb-spoiler-dataset/IMDB_movie_details.json',lines=True)



details.head()

plt.figure(figsize=(10,7))

plt.hist(details.rating,bins=details.rating.nunique())

plt.title("Distribution of ratings in dataset")

plt.xlabel("Rating")

plt.ylabel("Number of movies")
reviews = pd.read_json('/kaggle/input/imdb-spoiler-dataset/IMDB_reviews.json',lines=True)

reviews_cleaned = reviews[["movie_id","review_text"]]

reviews.head()

                          
for i in range(20,30):

    print(reviews.iloc[i].review_text)

    print('---------------------------------------------------------------')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

def get_vader_score(sent):

    return(pd.Series(sid.polarity_scores(sent)))

#import swifter

#scores=reviews_cleaned.review_text.swifter.apply(lambda x: get_vader_score(x))

#print(scores)

#reviews_with_score = pd.concat([reviews_cleaned,scores], axis=1)

#reviews_with_score

# Saved as file because of long run time

#reviews_with_score.to_csv("my_file.gz", compression="gzip")
#Importing saved dataset from the VADER classification made in the code above

imdb_reviews_vader_score = pd.read_csv("/kaggle/input/vaderscoredreviews/imdb-reviews-vader-score.csv", index_col=0)

imdb_reviews_vader_score
imdb_no_tex=imdb_reviews_vader_score[["movie_id", "neg","neu","pos","compound"]]

imdb_no_tex
count_reviews=imdb_no_tex.groupby("movie_id",as_index=False).count().neg.values

# calculating the mean of vader values

imdb_grouped = imdb_no_tex.groupby("movie_id",as_index=False).mean()

imdb_grouped
plt.scatter(imdb_grouped.index,count_reviews)
rating_sorted = details[["movie_id","rating"]].sort_values('movie_id')
print(imdb_grouped.iloc[95])

rating_sorted
# merge mean review rating and real imdb rating

imdb_grouped["rating"]=rating_sorted["rating"].values

imdb_with_rating = imdb_grouped



imdb_with_rating.iloc[:,1:]
# Calculating mean rating and creating boolean vector

rating_mean = imdb_with_rating.rating.mean()

imdb_with_rating["binMean"] = imdb_with_rating.rating >= rating_mean 

imdb_with_rating
# Transforming boolean vetor to binary int

imdb_with_rating["binMean"] = imdb_with_rating["binMean"].astype('int')

imdb_with_rating["binMean"]

#imdb_cat_rating= imdb_with_rating

#imdb_cat_rating["rating"]=pd.cut(imdb_with_rating.rating, np.arange(0,10.1,0.1))

#imdb_cat_rating
#split in train and test

data_to_split =imdb_with_rating

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    data_to_split[data_to_split.columns[1:-1]],data_to_split["binMean"], test_size=0.33, random_state=41)

print(X_train.iloc[:,:-1])

print(y_train)

#plt.hist(y_smog)
# Prerforming hyperparameter opimization with random grid search

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 15, num = 5)]

max_depth.append(None) # to try auto

# Minimum number of samples required to split a node

min_samples_split =[2,4,6, 10,15]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4,6]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)



cl_test = RandomForestClassifier()



rf_random = RandomizedSearchCV(estimator = cl_test, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train.iloc[:,:-1], y_train)
rf_random.best_params_
#Training classifier

cl = RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', max_depth= 4,bootstrap= True)



cl.fit(X_train.iloc[:,:-1], y_train)

#Making predictions

preds=cl.predict(X_test.iloc[:,:-1])

compare_df = pd.DataFrame(columns=["preds","true","rating"])
# Creating df for comparison

compare_df["true"]=y_test

compare_df["preds"]=preds

compare_df["rating"]=X_test.iloc[:,-1]

compare_df
from sklearn.metrics import accuracy_score

print("accuracy model: ",accuracy_score(compare_df["true"], compare_df["preds"]))

print("accuracy majority default: ", accuracy_score(compare_df["true"], np.repeat(1, len(compare_df["true"]))))

# 0.6228813559322034
points_wrongly_predicted = compare_df.loc[compare_df["preds"] != compare_df["true"]].rating

dist_poinst_wrongly_predicted = compare_df.loc[compare_df["preds"] != compare_df["true"]].preds

dist_poinst_correct_predicted = compare_df.loc[compare_df["preds"] == compare_df["true"]].preds

fig ,axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

axes[0].set_title("Distribution wrongly predicted ratings")

axes[0].hist(points_wrongly_predicted,bins=30)

axes[1].set_title("Distribution ratings testset")

axes[1].hist(X_test.iloc[:,-1], color="orange",bins=30)



plt.show()
plt.figure(figsize=(14, 6))

plt.hist(X_test.iloc[:,-1],bins=np.arange(3,10,0.1), alpha=0.5, label='Test data ratings')

plt.hist(points_wrongly_predicted,bins=np.arange(3,10.1,0.1), alpha=1, label='Ratings of faulty predicted movies')

plt.legend(loc='upper right')

plt.title("Test data rating distribution vs faulty predicted movie ratings")

plt.show()
points_wrongly_predicted.nunique()
for col in ['true', 'preds']:

    

    sns.kdeplot(compare_df[col], shade=True)



fig ,axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

plt.hist(compare_df['true'], alpha=0.5, label='Test set')

plt.legend(loc='center')

plt.title("Amount of predicted above or below average compared to true distribution")

plt.show()
plt.figure(figsize=(14, 6))

plt.hist(dist_poinst_wrongly_predicted)



plt.legend(loc='center')

plt.title("Distribution for faulty predicted values, 0=below average and 1=above average")

plt.show()
plt.figure(figsize=(14, 6))

plt.hist(dist_poinst_correct_predicted)

plt.legend(loc='center')

plt.title("Distribution for correct predicted values, 0=below average and 1=above average")

plt.show()
print(len(dist_poinst_correct_predicted))

print(len(dist_poinst_wrongly_predicted))

print(len(compare_df["true"]))

print(y_train.value_counts())
for col in [X_train.columns]:

    plt.hist(X_train.iloc[:,3],bins=32)