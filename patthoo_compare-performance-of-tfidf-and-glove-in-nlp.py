import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from nltk import word_tokenize

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

from matplotlib import pyplot as plt

import time
# !pip install nltk
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
print(train.shape)

test.shape
idx = []

for i in range(train.shape[0]):

    # print those docs with error

    if len(train.Text[i]) <= 10:

        print(train.iloc[i,:])

    else:

        idx.append(i)
# remove errors from the train dataset

train = train.iloc[idx,:]

train.shape
# plot to see how all labels are distributed

train.label.value_counts().plot(kind="barh")
# load the GloVe vectors in a dictionary

embeddings_index = {}

f = open('../input/glove.42B.300d.txt', encoding="utf8")

for line in tqdm(f):

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embeddings_index
# this function creates a normalized vector for an entire document

def doc2vec(s):

    words = str(s).lower()#.decode('utf-8')

    words = word_tokenize(words) # tokenize the sentence

    words = [w for w in words if not w in stop_words] # remove stop words

    words = [w for w in words if w.isalpha()] # remove numbers

    

    # create an list of word vectors excluding stopwords

    M = []

    for w in words:

        try:

            M.append(embeddings_index[w])

        except:

            continue

    M = np.array(M)

    

    # if there is an error, create an array with 300 zeros

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    

    return v / np.sqrt((v ** 2).sum())
# split the data into training and validation sets.

xtrain, xvalid, ytrain, yvalid = train_test_split(train.Text.values, train.label, stratify=train.label, random_state=42, 

                                                  test_size=0.2, shuffle=True)
# create a tfidf converter

tfidf_converter = TfidfVectorizer(min_df=2,  max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', 

                      ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')



# fit_transform(raw_documents[, y]) Learn vocabulary and idf, return term-document matrix.

# Fitting TF-IDF to both training

tfidf_converter.fit(list(xtrain) + list(xvalid)) # learn vocabulary and idf from training set and valid set

xtrain_tfidf =  tfidf_converter.transform(xtrain)  # Transform documents to document-term matrix.

xvalid_tfidf = tfidf_converter.transform(xvalid)
# Fitting a simple SVM

svm_mdl = SVC(C=1, kernel="linear")
start = time.time()



# fitting model

svm_tfidf = svm_mdl.fit(xtrain_tfidf, ytrain)



# generate precision score

tfidf_acc = svm_tfidf.score(xvalid_tfidf, yvalid)

tfidf_acc

end = time.time()



print("Process is finished in {}".format((end-start)/60))

print("Accuracy: ", tfidf_acc)
# convert documents into vectors using the above function for training and validation set

xtrain_glove = [doc2vec(x) for x in tqdm(xtrain)]

xvalid_glove = [doc2vec(x) for x in tqdm(xvalid)]



# transform into n-dimension arrays

xtrain_glove = np.vstack(xtrain_glove)

xvalid_glove = np.vstack(xvalid_glove)



# just carefully check for error before fitting a model

for item in xtrain_glove:

    if len(item) != 300:

        print(item)
start = time.time()



svm_glove = svm_mdl.fit(xtrain_glove, ytrain)

# generate precision score

glove_acc = svm_glove.score(xvalid_glove, yvalid)

glove_acc

end = time.time()



print("Process is finished in {}".format((end-start)/60))

print("Accuracy: ", glove_acc)
start = time.time()



# Fitting a radial basic function model

rbf_svm = SVC(C=1, kernel="rbf")

rbf_mdl = rbf_svm.fit(xtrain_glove, ytrain)



# generate precision score

rbf_acc = rbf_mdl.score(xvalid_glove, yvalid)

end = time.time()



print("Process is finished in {}".format((end-start)/60))

print("Accuracy: ", rbf_acc)