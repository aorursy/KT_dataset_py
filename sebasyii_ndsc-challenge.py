import pandas as pd

import numpy as np

import re

from tqdm import tqdm

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("../input/train.csv")

df.head()

df_test = pd.read_csv("../input/test.csv")
X = np.array(df['title'])

y = np.array(df['Category'])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53)

my_tags = [str(tag) for tag in set(y_test)]
model = Pipeline([('vectorizer', CountVectorizer(min_df=2,max_features=None,analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,3))),

    ('tfidf', TfidfTransformer(use_idf=False)),

    ('clf', OneVsRestClassifier(LinearSVC(C=1)))])
#fit model with training data

model.fit(X_train, y_train)

#evaluation on test data

pred = model.predict(X_test)

print('accuracy %s' % accuracy_score(y_test,pred))

print(classification_report(y_test, pred,target_names=my_tags))
from tqdm import tqdm

infile = open("predictions.csv",'w+')

infile.write('itemid,Category\n')



for i in tqdm(range(len(df_test))):

    a = df_test["title"][i]

    b = model.predict([a])[0]

    infile.write(str(df_test["itemid"][i]))

    infile.write(',')

    infile.write(str(b))

    infile.write('\n')

    

print("done")

infile.close()