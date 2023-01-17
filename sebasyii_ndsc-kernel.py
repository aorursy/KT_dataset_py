import pandas as pd

import numpy as np

import re

from tqdm import tqdm

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from dask_ml.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("../input/train.csv")

df.head()

df_test = pd.read_csv("../input/test.csv")
'''

def clean_str(string):

    """

    Tokenization/string cleaning for dataset

    Every dataset is lower cased except

    """

    string = re.sub(r"\n", "", string)    

    string = re.sub(r"\r", "", string) 

    string = re.sub(r"[0-9]", "", string)

    string = re.sub(r"\'", "", string)    

    string = re.sub(r"\"", "", string)

    string = re.sub(r"\b\w{1,1}\b","", string)

    string = re.sub(r"[,@\'?\.$%_&^!#*]", "", string)

    string = re.sub(r"\s+", " ", string)

    return string.strip().lower()

X = []

for i in tqdm(range(df.shape[0])):

    X.append(clean_str(df.iloc[i][1]))

y = np.array(df["Category"])

'''
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
from sklearn.externals import joblib

import pickle

model_text_classification_pkl = open('model_text_classification.pkl', 'wb')

pickle.dump(model, model_text_classification_pkl)

model_text_classification_pkl.close()
# Loading the saved decision tree model pickle

model_text_classification_pkl = open('model_text_classification.pkl', 'rb')

model = pickle.load(model_text_classification_pkl)

print ("Loaded Text classification model :: ", model_text_classification_pkl)
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