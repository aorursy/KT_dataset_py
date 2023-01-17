# Libraries

import numpy as np 

import pandas as pd

import os

import random

import re

from pathlib import Path

import fastText as ft

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from collections import Counter

import matplotlib.pyplot as plt
# import data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
train_values = df_train['target'].values

zeros = np.where(train_values == 0)



train_values = df_train['target'].values

ones = np.where(train_values == 1)



y = [len(zeros[0]), len(ones[0])]

x = [0, 1]



plt.bar([0, 1], y)

plt.xticks(x, (0, 1))

plt.title('Class distribution')

plt.show()
import re, string, unicodedata

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer, WordNetLemmatizer
stemmer = LancasterStemmer()

lemmatizer = WordNetLemmatizer()



def remove_non_ascii(word):

    """Remove non-ASCII characters from list of tokenized words"""

    new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return new_word



def to_lowercase(word):

    """Convert all characters to lowercase from list of tokenized words"""

    return word.lower()



def remove_punctuation(word):

    """Remove punctuation from list of tokenized words"""

    new_word = re.sub(r'[^\w\s]', '', word)

    return new_word



def remove_stopwords(word):

    """Remove stop words from list of tokenized words"""

    if word not in stopwords.words('english'):

        return word

    return ''



def stem_words(word):

    """Stem words in list of tokenized words"""

    return stemmer.stem(word)



def lemmatize_verbs(word):

    """Lemmatize verbs in list of tokenized words"""

    return lemmatizer.lemmatize(word, pos='v')



def normalize(word):

    word = remove_non_ascii(word)

    word = to_lowercase(word)

    word = remove_punctuation(word)

    # word = remove_stopwords(word)

    word = lemmatize_verbs(word)

    return word



def get_processed_text(string):

    words = nltk.word_tokenize(string)

    new_words = []

    for word in words:

        new_word = normalize(word)

        if new_word != '':

            new_words.append(new_word)

    return ' '.join(new_words)
df_train.question_text = df_train.question_text.apply(lambda x: get_processed_text(x))

df_train.head()
df_train['label_and_text'] = '__label__' + df_train.target.map(str) + ' '+ df_train.question_text

df_train.head()
df_test.question_text = df_test.question_text.apply(lambda x: get_processed_text(x))

df_test.head()
# Write training data to a file as required by fasttext

training_file = open('train.txt','w')

training_file.writelines(df_train.label_and_text + '\n')

training_file.close()
# Function to do K-fold CV across different fasttext parameter values

def tune(Y, X, YX, k, lr, wordNgrams, epoch, loss, ws):

    # Record results

    results = []

    for lr_val in lr:

        for wordNgrams_val in wordNgrams:

            for epoch_val in epoch:  

                for loss_val in loss:

                    for ws_val in ws:

                        # K-fold CV

                        kf = KFold(n_splits=k, shuffle=True)

                        fold_results = []

                        # For each fold

                        for train_index, test_index in kf.split(X):

                            # Write the training data for this CV fold

                            training_file = open('train_cv.txt','w')

                            training_file.writelines(YX[train_index] + '\n')

                            training_file.close()

                            # Fit model for this set of parameter values

                            model = ft.FastText.train_supervised(

                                'train_cv.txt',

                                lr=lr_val,

                                wordNgrams=wordNgrams_val,

                                epoch=epoch_val,

                                loss=loss_val,

                                ws=ws_val

                            )

                            # Predict the holdout sample

                            pred = model.predict(X[test_index].tolist())

                            pred = pd.Series(pred[0]).apply(lambda x: int(re.sub('__label__', '', x[0])))

                            # Compute accuracy for this CV fold

                            fold_results.append(accuracy_score(Y[test_index], pred.values))

                        # Compute mean accuracy across 10 folds 

                        mean_acc = pd.Series(fold_results).mean()

                        print([lr_val, wordNgrams_val, epoch_val, loss_val, ws_val, mean_acc])

    # Add current parameter values and mean accuracy to results table

    results.append([lr_val, wordNgrams_val, epoch_val, loss_val, ws_val, mean_acc])         

    # Return as a DataFrame 

    results = pd.DataFrame(results)

    results.columns = ['lr','wordNgrams','epoch','loss','ws_val','mean_acc']

    return(results)
# results = tune(

#     Y = df_train.target,

#     X = df_train.question_text,

#     YX = df_train.label_and_text,

#     k = 5, 

#     lr = [0.05, 0.1, 0.2],

#     wordNgrams = [1, 2, 3],

#     epoch = [1, 5],

#     ws=[5, 10],

#     loss=['ns', 'hs', 'softmax']

# )
# results = tune(

#     Y = df_train.target,

#     X = df_train.question_text,

#     YX = df_train.label_and_text,

#     k = 5, 

#     lr = [0.025, 0.05],

#     wordNgrams = [2, 3],

#     epoch = [5, 10, 15, 20],

#     ws=[5, 10, 30],

#     loss=['hs', 'softmax']

# )
# train the classifier

classifier1 = ft.FastText.train_supervised(

    'train.txt',  lr=0.05,   wordNgrams=3,  epoch=5,  loss='hs',  ws=5

)

classifier2 = ft.FastText.train_supervised(

    'train.txt',  lr=0.025,  wordNgrams=3,  epoch=1,  loss='hs',  ws=30

)

classifier3 = ft.FastText.train_supervised(

    'train.txt',  lr=0.05,   wordNgrams=3,  epoch=5,  loss='hs',  ws=5

)
# make predictions for test data

predictions1 = classifier1.predict(df_test.question_text.tolist())

predictions2 = classifier2.predict(df_test.question_text.tolist())

predictions3 = classifier3.predict(df_test.question_text.tolist())
# Combine predictions

most_common = np.array([])

for i in range(len(predictions1[0])):

    most_common = np.append(

        most_common, 

        Counter([

            predictions1[0][i][0],

            predictions2[0][i][0],

            predictions3[0][i][0]

        ]).most_common(1)[0][0])
# Write submission file

submit = pd.DataFrame({

    'qid': df_test.qid,

    'prediction':  pd.Series(most_common)

})

submit.prediction = submit.prediction.apply(lambda x: re.sub('__label__', '', x))

submit.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')