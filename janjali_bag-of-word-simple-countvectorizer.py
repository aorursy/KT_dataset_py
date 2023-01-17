# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
bag_train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, \
delimiter="\t", quoting=3)

test_bag = pd.read_csv('../input/testData.tsv', header = 0, \
delimiter="\t", quoting=3)
# printing the element 
print(bag_train.shape)
print(test_bag.shape)
# Any results you write to the current directory are saved as output.
print(bag_train['review'][0])
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

clean_review = []
def review_words( raw_review):
        text = BeautifulSoup(raw_review, features="html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", text)            
        words = letters_only.lower().split() 
        stops = set(stopwords.words("english"))                         
        meaningful_words = [w for w in words if not w in stops]         
        return(" ".join( meaningful_words ))

           
for i in range( 0, len(bag_train["review"])):
        clean_review.append(review_words( bag_train["review"][i]))

print(clean_review[0])

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

vect_review=vectorizer.fit_transform(clean_review)
np.asarray(vect_review)

from sklearn.ensemble import RandomForestClassifier

tree = RandomForestClassifier(n_estimators = 100)
train_text= tree.fit(vect_review, bag_train["sentiment"] )

clean_test= []

for i in range(0,len(test_bag["review"])):
        clean_test.append(review_words(test_bag["review"][i]))
        
test_feature= vectorizer.transform(clean_test)
np.asarray(test_feature)
        