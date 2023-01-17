# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer





import re



import matplotlib.pyplot as plt

%matplotlib inline

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
df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
rslt_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]

rslt_df2 = df[(df['toxic'] == 1) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]

new1 = rslt_df[['id', 'comment_text', 'toxic']].iloc[:23000].copy() 

new2 = rslt_df2[['id', 'comment_text', 'toxic']].iloc[:900].copy()

new = pd.concat([new1, new2], ignore_index=True)

new.head()
from nltk.corpus import stopwords

my_stopwords = stopwords.words('english')
import nltk

tk=nltk.tokenize.TreebankWordTokenizer()

comment_tokens = [tk.tokenize(sent) for sent in new['comment_text']]
type(comment_tokens)
comment_tokens[0]
len(comment_tokens)
from nltk.corpus import stopwords

for i in range(len(comment_tokens)):

    comment_tokens[i] = [w for w in comment_tokens[i] if w not in stopwords.words('english')]
#glove embeddings

from numpy import array

from numpy import asarray

from numpy import zeros



embeddings_dictionary = dict()



glove_file = open('/kaggle/input/nlpword2vecembeddingspretrained/glove.6B.100d.txt', encoding = "utf8")
for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary[word] = vector_dimensions

glove_file.close()    
print(word)
print(records)
print(vector_dimensions)
print(embeddings_dictionary['hello'])
vocab = embeddings_dictionary.keys()
len(vocab)
# Let's find the top 7 words that are closest to 'compute'

u = embeddings_dictionary['compute']

norm_u = np.linalg.norm(u)

similarity = []



for word in embeddings_dictionary.keys():

    v = embeddings_dictionary[word]

    cosine = np.dot(u, v)/norm_u/np.linalg.norm(v)

    similarity.append((word, cosine))

print(len(similarity))
sorted(similarity, key=lambda x: x[1], reverse=True)[:10]
# ## Now let's do vector algebra.

# 

# ### First we subtract the vector for `france` from `paris`. This could be imagined as a vector pointing from country to its capital. Then we add the vector of `nepal`. Let's see if it does point to the country's capital

output = embeddings_dictionary['paris'] - embeddings_dictionary['france'] + embeddings_dictionary['nepal']

norm_out = np.linalg.norm(output)
similarity = []

for word in embeddings_dictionary.keys():

    v = embeddings_dictionary[word]

    cosine = np.dot(output, v)/norm_out/np.linalg.norm(v)

    similarity.append((word, cosine))

    

print(len(similarity))



sorted(similarity, key=lambda x: x[1], reverse=True)[:7]    
documents = []

for x in comment_tokens:

    document = [word for word in x if word in vocab]

    documents.append(document)

#now this document have only those words which are present in our model's vocab

documents[1:5]   
documents[0]
len(documents)
#checking if there is any empty list inside documents

counter = 0

for i in range (0,len(documents)):

    if documents[i] == []:

        counter += 1

print(counter)
#document embeddings

list_v=[]

for i in range (0,len(documents)):

    if documents[i] == []:

        list_v.append(np.zeros(100,))

    else:

        vec = []

        for j in documents[i]:

            v = embeddings_dictionary[j]

            vec.append(v)

        list_v.append(np.mean(vec, axis=0))
len(documents[i])
len(list_v[0])
from collections import Counter

print('Original dataset shape before smote %s' % Counter(new['toxic']))

from imblearn.over_sampling import SMOTE

oversample = SMOTE()

X, y = oversample.fit_resample(list_v, new['toxic'])

print('Original dataset shape after smote %s' % Counter(y))
#test-train split

from sklearn.model_selection import train_test_split

Xw_train, Xw_test, yw_train, yw_test = train_test_split(X,y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(max_iter=1000)

clf.fit(Xw_train,yw_train)
predicted_res=clf.predict(Xw_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(yw_test,predicted_res)

accuracy
import numpy as np



z=1.96

interval = z * np.sqrt( (0.8244 * (1 - 0.8244)) / yw_test.shape[0])

interval
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

import numpy as np



clf3 = RandomForestClassifier() #Initialize with whatever parameters you want to



# 10-Fold Cross validation

scores = cross_val_score(clf3,Xw_train,yw_train, cv=5)
y_p3 = clf3.fit(Xw_train, yw_train).predict(Xw_test)

accuracy = accuracy_score(yw_test, y_p3)

print('Accuracy: %f' % accuracy)



import numpy as np



z=1.96

interval = z * np.sqrt( (0.9629 * (1 - 0.9629)) / yw_test.shape[0])

interval