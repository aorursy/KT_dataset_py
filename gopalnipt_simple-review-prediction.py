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
data=pd.read_csv("/kaggle/input/Restaurant_Reviews.tsv",delimiter="\t",quoting=3)
data.head()
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
Review=data["Review"]

like=data["Liked"]
Tokenizer=RegexpTokenizer(r"\w+[a-z]+")   ## only word

stop_word=set(stopwords.words("english"))
Review[1]
fresh_data=[]

for ix in Review:

    sen=ix.lower()

    sen=Tokenizer.tokenize(sen)

    sen_word=[(word) for word in sen if len(word)>=2 ]

    sentance=" ".join(sen_word)

    fresh_data.append(sentance)
text_clf = Pipeline([('vect', CountVectorizer(stop_words="english")),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(Review,like)
review_1=['Farriers International is a excellent place to go with family and have a quality time Buffet have a wide range of variety specially for the vegetarian with excellent dessert serving .Awesome Ambience and racecourse view is a added to the really fantastic service . Deserve all the praises for a five star brunch .',

          'The food quality is very very bad had order some soup it was so terrible could eat more than a spoonful. They need to change the chef at the earliest. The service and ambiance is okay.']



#https://www.tripadvisor.in/ShowUserReviews-g295424-d3650163-r363847145-Farriers_International_All_Day_Dining_Restaurant-Dubai_Emirate_of_Dubai.html#']
Test=[]

for ix in review_1:

    ss=Tokenizer.tokenize(ix)

    sp=" ".join(ss)

    Test.append(sp)

    

predict_Review = gs_clf.predict(Test)

for my_review in predict_Review:

    print("MY_review",my_review)
# n=int(input('No of review'))

# texts =[input() for ix in range(n)]

# Test=[]

# for ix in texts:

#     ss=Tokenizer.tokenize(ix)

#     sp=" ".join(ss)

#     Test.append(sp)

    

# predict_Review = gs_clf.predict(Test)

# for my_review in predict_Review:

#     print("MY_review",my_review)