# The input files are Sheet_1 & Sheet_2.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/deepnlp"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/deepnlp/Sheet_1.csv', encoding='latin-1')

df1 = df1.drop(["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6","Unnamed: 7"], axis = 1)

df1 = df1.rename(columns={"v1":"class", "v2":"Responses"})



df1.head()
df2 = pd.read_csv('../input/deepnlp/Sheet_1.csv', encoding='latin-1')

df2 = df2.drop(["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6","Unnamed: 7"], axis = 1)

df2 = df2.rename(columns={"v1":"class", "v2":"Responses"})



df2.head()
print (df1["class"].value_counts())

print (df2["class"].value_counts())
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

import pandas as pd

from sklearn.metrics import accuracy_score

import pickle

from sklearn.model_selection import train_test_split



stopWords = set(nltk.corpus.stopwords.words('english'))



vect = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',

                                 decode_error='ignore',stop_words=stopWords)
output_dict = {

    0:"not_flagged",

    1:"flagged"}
combined_df  = pd.concat([df1,df2])

combined_df.head()
X_train, X_test, y_train, y_test = train_test_split(combined_df["response_text"], combined_df["class"], test_size=0.33, random_state=42)
xTrain = X_train

yTrain = y_train



tfidf = vect.fit(xTrain.values.astype('U'))

xTrainvect = vect.fit_transform(xTrain)

yTrainvect = yTrain



xTestvect = vect.transform(X_test)

yTestvect = y_test



model = MultinomialNB(alpha=0.01, fit_prior=True)

model.fit(xTrainvect, yTrainvect)



ypred = model.predict(xTestvect)

score = accuracy_score(yTestvect, ypred)

print ("Accuracy: ",score)
test = "i cant think of one really...i think i may hav "

new_pred = model.predict(vect.transform([test]))

print(new_pred)