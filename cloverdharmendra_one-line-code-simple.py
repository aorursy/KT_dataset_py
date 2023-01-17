# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Gender classification using universal data set



#import libaries

import numpy as np

import pandas as pd

#ML Packages

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import DictVectorizer

#load teh data set

data_set = pd.read_csv("../input/names-dataset/names_dataset.csv")

xfeatures  = data_set["name"]



#feature extraction

cv = CountVectorizer()

X = cv.fit_transform(xfeatures)





data_set.sex.replace({'F':0,'M':1},inplace=True)



#features

X = X

#label

data_set.drop_duplicates(keep="first", inplace=True)

y  =data_set.sex

from collections import  Counter

print("ty",Counter(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





# Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X, y)



def Predict(data):

    test_name = [data]

    vector = cv.transform(test_name).toarray()

    result = clf.predict(vector)[0]

    return result



train_set = pd.read_csv("../input/gendered-pronoun-resolution/test_stage_1.tsv", encoding="utf-8", error_bad_lines=False, delimiter='\t')





train_set["A"] = train_set["A"].apply(Predict)

train_set["B"] = train_set["B"].apply(Predict)

train_set = train_set[["ID", "A", 'B'] ]
train_set = train_set.assign(NEITHER=lambda r: 1-r.A-r.B)


train_set["NEITHER"] = abs(train_set["NEITHER"].astype(int))
train_set.to_csv('sub.csv', index=False)