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
df = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv')
df.head()
def read_questions(row,column_name):

    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))
import gensim
sentences = []

for index, row in df.iterrows():

    sentences.append(read_questions(row,"question1"))

    sentences.append(read_questions(row,"question2"))
model = gensim.models.Word2Vec(sentences)
def getSentVec(sentence, model, sentIndex):

    sentVec = np.zeros(100)

    if len(sentence):

        for index, word in enumerate(sentence):

            if word in model.wv:

                sentVec = sentVec +  np.array(model.wv[word])

        sentVec = sentVec/len(sentence)

    else:

        emptyArrayIndices.append(sentIndex)

    return sentVec.tolist()
sentVecsQ1 = []

sentVecsQ2 = []

emptyArrayIndices = []

for index, row in df.iterrows():

    sentVecsQ1.append(getSentVec(read_questions(row,"question1"), model, index))

    sentVecsQ2.append(getSentVec(read_questions(row,"question2"), model, index))    
is_duplicate = df['is_duplicate']
is_duplicate = is_duplicate.tolist()
for index in emptyArrayIndices:

    sentVecsQ1.pop(index)

    sentVecsQ2.pop(index)

    is_duplicate.pop(index)
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
new_df = pd.DataFrame()

new_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['minkowski_distance'] = [minkowski(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(sentVecsQ1, sentVecsQ2)]

new_df['is_duplicate'] = is_duplicate
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier

new_df.head()
new_df.shape
new_df.dropna(inplace=True)

new_df.isnull().sum()
X = new_df.iloc[:,:new_df.shape[1]-1]

y = new_df.iloc[:,new_df.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 

                                                    random_state=2018,

                                                    stratify=y)

# model_rf = RandomForestClassifier(random_state=1211,

#                                   n_estimators=500,oob_score=True)

# model_rf.fit( X_train , y_train )

# y_pred = model_rf.predict(X_test)



# print(confusion_matrix(y_test, y_pred))

# print(classification_report(y_test, y_pred))

# print(accuracy_score(y_test, y_pred))



# from sklearn.metrics import roc_curve, roc_auc_score



# # Compute predicted probabilities: y_pred_prob

# y_pred_prob = model_rf.predict_proba(X_test)[:,1]
# from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import StratifiedKFold



# kfold = StratifiedKFold(n_splits=5, random_state=42)

# log_reg = RandomForestClassifier(n_estimators=500,oob_score=True)

# results = cross_val_score(log_reg, X, y, cv=kfold, 

#                          scoring='roc_auc')

# print(results)

# print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))
# log_reg = RandomForestClassifier(n_estimators=200,oob_score=True)

# results = cross_val_score(log_reg, X, y, cv=kfold, 

#                          scoring='neg_log_loss')

# print("Log Loss: %.4f (%.4f)" % (results.mean(), results.std()))