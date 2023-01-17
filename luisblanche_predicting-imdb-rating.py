import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
mov=pd.read_csv("../input/movie_metadata.csv")

print(mov.columns.values)

labels=mov["imdb_score"]

mov.drop(["imdb_score", "aspect_ratio", "movie_imdb_link"], inplace=True, axis=1)
numeric_features=mov._get_numeric_data().columns.values.tolist()



text_features=mov.columns.values.tolist()

text_features=[i for i in text_features if i not in numeric_features]







string_features=["movie_title", "plot_keywords"]



categorical_features=[i for i in text_features if i not in string_features]



numeric_features.remove("title_year") 

categorical_features.append("title_year")



### Title_year  is categorical

###all the others can be considered continuous 

###(See about facenumber_in_poster too)
%matplotlib inline

import matplotlib.pyplot as plt

font = {'fontname':'Arial', 'size':'14'}

title_font = { 'weight' : 'bold','size':'16'}

plt.hist(labels, bins=20)

plt.title("Distribution of the IMDB ratings")

plt.show()
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler 

## we use standard scaler to keep as much variance as possible (compared to minmax)

imp=Imputer(missing_values='NaN',strategy="most_frequent", axis=0)

mov[numeric_features]=imp.fit_transform(mov[numeric_features])



scl=StandardScaler()

mov[numeric_features]=scl.fit_transform(mov[numeric_features])



mov[numeric_features].head() 
import operator



from scipy.stats import pearsonr

correl={}

for f in numeric_features:

    correl[f]=pearsonr(mov[f], labels)

sorted_cor = sorted(correl.items(), key=operator.itemgetter(1), reverse=True)

print (sorted_cor)
import seaborn as sns

import matplotlib.pyplot as plt

def corrmap(features, title):

    sns.set(context="paper", font="monospace")

    corrmat = mov[features].corr()

    f, ax = plt.subplots(figsize=(12, 9))

    plt.title(title, **title_font)

# Draw the heatmap using seaborn

    sns.heatmap(corrmat, vmax=.8, square=True)

corrmap(numeric_features,"Correlation matrix for numeric features")
mov["movie_success"]=(mov['num_critic_for_reviews']+mov["num_voted_users"]

                +mov["num_user_for_reviews"]+mov["gross"]+mov["movie_facebook_likes"])/6

mov["other_actors_facebook_likes"]=mov["actor_2_facebook_likes"]+mov["actor_3_facebook_likes"]

num_features_2=[x for x in numeric_features if x not in ["cast_total_facebook_likes",

                                                         'num_critic_for_reviews',

                                                         "num_voted_users",

                                                         "num_user_for_reviews",

                                                        "gross","movie_facebook_likes",

                                                        "actor_2_facebook_likes",

                                                        "actor_3_facebook_likes"]]

num_features_2.extend(["movie_success", "other_actors_facebook_likes"])
corrmap(num_features_2, "Correlation matrix with new numeric features")
import operator



from scipy.stats import pearsonr

correl={}

for f in num_features_2:

    correl[f]=pearsonr(mov[f], labels)

sorted_cor = sorted(correl.items(), key=operator.itemgetter(1), reverse=True)

print (sorted_cor)
from sklearn.ensemble import RandomForestRegressor

RFR=RandomForestRegressor(max_features="sqrt")

parameters={ "max_depth":[5,8,25], 

             "min_samples_split":[1,2,5], "n_estimators":[800,1200]}

             
from sklearn.grid_search import GridSearchCV

clf = GridSearchCV(RFR, parameters)

clf.fit(mov[num_features_2],labels)
from operator import itemgetter

# Utility function to report best scores

def report(grid_scores, n_top=3):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):

        print("Rank: {0}".format(i + 1))

        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

              score.mean_validation_score,

              np.std(score.cv_validation_scores)))

        print("Parameters: {0}".format(score.parameters))

        print("")

report(clf.grid_scores_)
for feat in categorical_features:

    mov=pd.concat([mov, pd.get_dummies(mov[feat], prefix=feat, dummy_na=True)],axis=1)
cat_dummies=[i for i in mov.columns.values.tolist() if i not in numeric_features]

cat_dummies=[i for i in cat_dummies if i not in text_features]

cat_dummies.remove("title_year")

cat_dummies[-5:]
import operator



from scipy.stats import pearsonr

correl={}

for f in cat_dummies:

    correl[f]=pearsonr(mov[f], labels)

sorted_cor = sorted(correl.items(), key=operator.itemgetter(1), reverse=True)



print (sorted_cor[0:10])

print("")

print (sorted_cor[-10:])
predictors=["movie_success","duration","director_facebook_likes", "title_year_nan", 

            "color_ Black and White",

            "director_name_nan", "country_UK", "content_rating_TV-MA", "genres_Drama",

            "genres_Crime|Drama", 'other_actors_facebook_likes','actor_1_facebook_likes',

           "content_rating_Approved", "genres_Drama|Romance", "title_year_2015.0", 

           "director_name_Jason Friedberg","genres_Horror","genres_Comedy|Romance","director_name_Uwe Boll", "country_USA","content_rating_PG-13","color_Color", "language_English"]
corrmap(predictors, "Correlation matrix for all relevant predictors")
predictors=["movie_success","duration","director_facebook_likes", "title_year_nan", 

            "color_ Black and White",

             "content_rating_TV-MA", "genres_Drama",

            "genres_Crime|Drama", 'other_actors_facebook_likes','actor_1_facebook_likes',

           "content_rating_Approved", "genres_Drama|Romance", "title_year_2015.0", 

           "director_name_Jason Friedberg","genres_Horror","genres_Comedy|Romance","director_name_Uwe Boll",

            "country_USA","content_rating_PG-13", "language_English"]
corrmap(predictors, "Correlation matrix for all relevant and independant predictors")
from sklearn.ensemble import RandomForestRegressor

RFR=RandomForestRegressor(max_features="sqrt")

parameters={ "max_depth":[5,8,25], 

             "min_samples_split":[1,2,5], "n_estimators":[800,1200]}

from sklearn.grid_search import GridSearchCV

clf = GridSearchCV(RFR, parameters)

clf.fit(mov[predictors],labels)


report(clf.grid_scores_)
from sklearn.feature_extraction.text import CountVectorizer

mov["plot_keywords"]=mov["plot_keywords"].fillna("None")



def token(text):

    return(text.split("|"))



cv=CountVectorizer(max_features=200,tokenizer=token )

plot_keywords_words=cv.fit_transform(mov["plot_keywords"])



plot_keywords_words=plot_keywords_words.toarray()



words = cv.get_feature_names()

words=["Keyword_"+w for w in words]



keywords=pd.DataFrame(plot_keywords_words, columns=words)
keys=[w for w in words if keywords[w].sum()>80] 

### takes the keywords that concern at least 80 (totally arbitrary) of the movies

len(keys)
mov=pd.concat([mov, keywords[keys]],axis=1)
num_cat_key_feat=predictors+keys

import operator



from scipy.stats import pearsonr

correl={}

for f in keys:

    correl[f]=pearsonr(mov[f], labels)

sorted_cor = sorted(correl.items(), key=operator.itemgetter(1), reverse=True)



print (sorted_cor[0:10])

print (sorted_cor[-10:])
corrmap(predictors+keys, "Correlation matrix for all relevant predictors")
from sklearn.ensemble import RandomForestRegressor

RFR=RandomForestRegressor(max_features="sqrt")

parameters={ "max_depth":[2,5,8,25], 

             "min_samples_split":[1,2,5], "n_estimators":[800,1200]}

from sklearn.grid_search import GridSearchCV

clf = GridSearchCV(RFR, parameters)

clf.fit(mov[num_cat_key_feat],labels)
report(clf.grid_scores_)