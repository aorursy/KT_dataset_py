# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

import seaborn as sns



# sklearn for feature extraction & modeling

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

# Iteratively read files

import glob

import os



# For displaying images in ipython

import seaborn as sns

sns.set(color_codes = True)

%matplotlib inline
df = pd.read_csv("/kaggle/input/food-com-recipes-and-user-interactions/RAW_interactions.csv")
print(df.shape)

df.head()
df["count"] = [1] * len(df)

df.head()
# pivot_table = df.pivot_table(values = "count" , index = "recipe_id" , columns = "rating", aggfunc= np.sum,

#                             fill_value = 0)
# pivot_table.head(n=4)
# pivot_table["Total Rating"] = pivot_table.apply(lambda x : np.sum(x),axis = 1)
# pivot_table.head()
df["rating"].value_counts()
df = df.dropna()

df = df[df["rating"] !=0]

df.shape
df["rating"].value_counts()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def create_wordcloud(dframe):

    unique_ratings = dframe["rating"].unique()

    # Create stopword list:

    stopwords = set(STOPWORDS)

    for rating in unique_ratings:

        temp_text = dframe[dframe["rating"]== rating]["review"]

        collapsed_temp_text = temp_text.str.cat(sep=' ')

        

        print("Word Cloud for Rating: %s"%(rating))



        # Generate a word cloud image

        wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_temp_text)



        # Display the generated image:

        # the matplotlib way:1

        plt.imshow(wordcloud, interpolation='bilinear')

        plt.axis("off")

        plt.show()
create_wordcloud(dframe = df.iloc[0:1000,:])
# Building Pipeline for raw text transformation

clf = Pipeline([

    ('vect', CountVectorizer(stop_words= "english")),

    ('tfidf', TfidfTransformer()),

    ('classifier', MultinomialNB(

                    fit_prior=True, class_prior=None)),

    ])
X_train, X_test, y_train, y_test = train_test_split(df["review"][0:100000]

                                                    , df["rating"][0:100000],random_state = 42,

                                                   test_size = 0.20)

X_train.shape,X_test.shape,y_train.shape
model = clf.fit(X_train , y_train)
print("Accuracy of Prediction on Test Data: %s"%model.score(X_test,y_test))
# Building Pipeline for raw text transformation

clf = Pipeline([

    ('vect', CountVectorizer(stop_words= "english")),

    ('tfidf', TfidfTransformer()),

    ('classifier', RandomForestClassifier()),

    ])
X_train, X_test, y_train, y_test = train_test_split(df["review"][0:100000]

                                                    , df["rating"][0:100000],random_state = 42,

                                                   test_size = 0.20)

X_train.shape,X_test.shape,y_train.shape
model = clf.fit(X_train , y_train)
print("Accuracy of Prediction on Test Data: %s"%model.score(X_test,y_test))
param_grid = {

    'classifier__criterion': ["gini","entropy"],

    #'classifier__max_features': ["auto","sqrt","log2"],

    'classifier__max_depth':[4,6],

    'classifier__n_estimators':[100,150,200]

}



grid_search = GridSearchCV(clf, param_grid, cv=3, iid=False,verbose = 1,n_jobs= -1)

grid_search.fit(X_train, y_train)
print("Accuracy of Prediction on Test Data: %s"%grid_search.score(X_test,y_test))