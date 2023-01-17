# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#for preprocessing
import re
import random
from string import punctuation
import nltk
from nltk.corpus import stopwords

#for building ml models
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


#for building a neural network
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
#read small chunk of data
data = pd.read_csv("../input/enron-email-dataset/emails.csv", chunksize=10000)
df_chunk = next(data)
df_chunk.head()
df_chunk.tail()
df_chunk['file'][0]
df_chunk['message'][1].split("\n")
df_chunk['message'][1].split("\n")[15:]
#extract author of email
extract_name = re.compile(r'^([a-z0-9\-]*)/')
df_chunk['name'] = df_chunk['file'].str.extract(extract_name)
df_chunk.head()
def extract_text(Series):
    """
    returns the string of content of the email in the "message" column while discarding those that contains the headers specified below.
    
    Series is the "message" column of the enron email dataset.
    
    """
    
    result = []
    
    headers = ['Message-ID:', 'Date:', 'From:', 'To:', 'Subject:', 'Mime-Version:', 'Content-Type',
              'Content-Transfer-Encoding:', 'X-From:', 'X-To:', 'X-cc:', 'X-bcc:', 'X-Folder:', 
              'X-Origin:', 'X-FileName:', 'Cc:', 'Bcc:', '-----Original Message-----',
              '----------------------', "Sent:", "cc:"]
    
    for row, message in enumerate(Series):
        strings = message.split("\n")
        accepted_strings = [string for string in strings if all(header not in string for header in headers)]
        result.append(" ".join(accepted_strings))
            
    return result
#extract content of email
df_chunk['text'] = extract_text(df_chunk['message'])
df_chunk.head()
def clean_text(Series):
    """
    returns the string of cleaned text of the message of the dataset.
    
    Series is the series of string containing text that wants to be cleaned
    
    """
    
    result = []
    sw = stopwords.words("english")
    strings = Series.str.lower()
    
    for string in strings:
        new_string = []
        words = string.split(" ")
        
        for word in words:
            word = word.strip(punctuation) 
            
            if word in sw:
                continue
            if re.search(r'[\W\d]',word):
                continue
                
            new_string.append(word)
                
        new_string = " ".join(new_string)
        
        result.append(new_string)
    
    return result
#cleaning the text column
df_chunk['text'] = clean_text(df_chunk['text'])
df_chunk.head()
df_chunk['name'].value_counts()
#grouping and combining the text according to the author's name
df_grouped = df_chunk.groupby("name")['text'].apply(' '.join).reset_index()
df_grouped.head()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
features = vectorizer.fit_transform(df_grouped['text'])
#read and preprocess the whole dataset
df = pd.read_csv("../input/enron-email-dataset/emails.csv")
df['name'] = df['file'].str.extract(extract_name)
df['text'] = extract_text(df['message'])
df['text'] = clean_text(df['text'])
df.head()
df_grouped = df.groupby("name")['text'].apply(' '.join).reset_index()
df_grouped.head()
poi_names = ['lay-k', 'skilling-j', 'forney-j', 'delainey-d']
df_grouped['poi'] = 0

for idx, name in enumerate(df_grouped['name']):
    if name in poi_names:
        df_grouped.loc[idx, 'poi'] = 1
        
print("Number of POI : {}".format(sum(df_grouped['poi'])))
df_grouped.head()
# save the processed dataframe in order to save time for the next time you want to use it
# df_grouped.to_csv("df_grouped.csv", index=False)
df_grouped = pd.read_csv("../input/groupeddata/df_grouped.csv")
df_grouped[df_grouped['poi']==1]
poi_idx = df_grouped.index[df_grouped['poi']==1].to_list()

#select rows that are not POI, then shuffle the index
shuffle_idx = [idx for idx in df_grouped.index.to_list() if idx not in poi_idx]
random.shuffle(shuffle_idx)

#select 70% as training data, then add 3 more data which are POIs
train_idx = shuffle_idx[:round(len(shuffle_idx)*0.7)]
train_idx = train_idx + poi_idx[:3]

#select the rest as test data, adding 1 more data which is POI
test_idx = shuffle_idx[round(len(shuffle_idx)*0.7):]
test_idx = test_idx + poi_idx[3:]
df_train = df_grouped.loc[train_idx, :].reset_index(drop=True)
df_test = df_grouped.loc[test_idx, :].reset_index(drop=True)
df_train.head()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
features_train = vectorizer.fit_transform(df_train['text']).toarray()
features_test = vectorizer.transform(df_test['text']).toarray()
#decision tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(features_train, df_train['poi'])
dt_pred = dt_clf.predict(features_test)

#gaussian naive-bayes
nb_clf = GaussianNB()
nb_clf.fit(features_train, df_train['poi'])
nb_pred = nb_clf.predict(features_test)

#knn
knn_clf = KNeighborsClassifier()
knn_clf.fit(features_train, df_train['poi'])
knn_pred = knn_clf.predict(features_test)

#adaboost
ab_clf = GradientBoostingClassifier()
ab_clf.fit(features_train, df_train['poi'])
ab_pred = ab_clf.predict(features_test)

preds = {"Decision Tree":dt_pred,
         "Naive-bayes":nb_pred,
         "K-Nearest Neighbor":knn_pred,
         "Adaboost":ab_pred}
#print the confusion matrix for the predictions of each models
for pred in preds:
    print("\n", pred)
    print(pd.DataFrame(confusion_matrix(df_test['poi'], preds[pred]), 
                 columns=["0(Predicted)", "1(Predicted)"], index=["0(Actual)", "1(Actual)"]))
print("Size of input layer of neural network : {}".format(features_train.shape[1]))
### I tried using pytorch, but it still isnt working, so I used the sklearn implementation instead
# nn_input = torch.from_numpy(features_train)

# model = nn.Sequential(nn.Linear(features_train.shape[1], 1000),
#                       nn.ReLU(),
#                       nn.Linear(1000, 100),
#                       nn.ReLU(),
#                       nn.Linear(100, 2),
#                       nn.LogSoftmax(dim=0))
# model = model.float()

# criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003)
# epochs = 5
# for e in range(epochs):
#     running_loss = 0
#     for idx in range(len(df_train['poi'])):
    
#         optimizer.zero_grad()
#         output = model(nn_input[idx].float())
#         print("output : {}, target : {}".format(output, df_train.loc[idx, 'poi']))
#         loss = criterion(output, df_train.loc[idx, 'poi'])
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#     else:
#         print(f"Training loss: {running_loss/len(df_train['poi'])}")
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(100, 10, 2), random_state=1)
nn_clf.fit(features_train, df_train['poi'])
nn_pred = nn_clf.predict(features_test)
print("Neural Network")
print(pd.DataFrame(confusion_matrix(df_test['poi'], nn_pred), 
                 columns=["0(Predicted)", "1(Predicted)"], index=["0(Actual)", "1(Actual)"]))