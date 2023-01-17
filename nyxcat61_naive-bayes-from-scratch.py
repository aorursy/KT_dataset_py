import numpy as np

import pandas as pd 

import re

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/labeledTrainData.tsv', sep='\t')

df.shape
df.head()
xtrain, xvali, ytrain, yvali = train_test_split(df.review, df.sentiment, test_size=0.2, random_state=0)
def CleanDocument(document):

    # replace < br/> with new_line_tag

    clean_doc = re.sub(r'\<br /\>', 'new_line_tag ', document)

    # remove punctuation

    clean_doc = re.sub(r'\W', ' ', clean_doc)

    # map numbers to NUMBERS

    return clean_doc.lower()
# clean train data

xtrain_after_clean = xtrain.apply(CleanDocument)

xtrain_after_clean.head()
# map word to id

def GetVocabulary(data):

    vocabulary = dict()

    wid = 0

    for document in data:

        words = document.split()

        for w in words:

            if w not in vocabulary:

                vocabulary[w] = wid

                wid += 1

    return vocabulary



vocab_dict = GetVocabulary(xtrain_after_clean)
# convert document to vectors

def Document2Vector(document, vocabulary):

    doc_vec = np.zeros(len(vocabulary))

    out_of_voc = 0



    words = document.split()

    for w in words:

        if w in vocabulary:

            wid = vocabulary[w]

            doc_vec[wid] += 1

        else:

            out_of_voc += 1

    return doc_vec, out_of_voc

train_matrix = []



for document in xtrain_after_clean:

    doc_vec, _ = Document2Vector(document,vocab_dict)

    train_matrix.append(doc_vec)



print(len(train_matrix))

print(train_matrix[0])
def NaiveBayes_train(train_matrix, labels):

    '''

    Calculate the log of p(pos), p(neg), p(word|pos) vector, p(word|neg) vector

    '''

    num_docs = len(train_matrix)

    num_words = len(train_matrix[0])

    

    pos_count, neg_count = 0, 0

    pos_total_word, neg_total_word = 0, 0

    pos_word_vector = np.ones(num_words)

    neg_word_vector = np.ones(num_words)

    

    for i in range(num_docs):

        if (i + 1) % 2000 == 0:

            print('Training %d/%d...' % (i+1, num_docs))

        if labels[i] == 1:

            pos_count += 1

            pos_total_word += sum(train_matrix[i])

            pos_word_vector += train_matrix[i]

        else:

            neg_count += 1

            neg_total_word += sum(train_matrix[i])

            neg_word_vector += train_matrix[i]

            

    p_pos = np.log(pos_count / num_docs)

    p_neg = np.log(neg_count / num_docs)

    p_pos_word_vector = np.log(pos_word_vector / (pos_total_word + num_words))

    p_neg_word_vector = np.log(neg_word_vector / (neg_total_word + num_words))

    

    return p_pos, p_pos_word_vector, p_neg, p_neg_word_vector, pos_total_word, neg_total_word





p_pos, p_pos_word_vector, p_neg, p_neg_word_vector, pos_total_word, neg_total_word = NaiveBayes_train(train_matrix, ytrain.values)
# making predictions with NB classifier

def predict(test_vector, p_pos, p_pos_word_vector, p_neg, p_neg_word_vector, pos_smoothing, neg_smoothing):

    pos = np.sum(test_vector * p_pos_word_vector) + p_pos + pos_smoothing

    neg = np.sum(test_vector * p_neg_word_vector) + p_neg + neg_smoothing

    if pos > neg:

        return 1

    else:

        return 0
# clean validation set

xvali_after_clean = xvali.apply(CleanDocument)

num_words = len(vocab_dict)

pred_vali = []



for i, document in enumerate(xvali_after_clean):

    if (i + 1) % 500 == 0:

        print('Testing %d/%d...' % (i + 1, xvali_after_clean.shape[0]))

    test_vec, out_of_voc = Document2Vector(document, vocab_dict)

    if out_of_voc == 0:

        pos_smoothing, neg_smoothing = 0, 0

    else:

        pos_smoothing = np.log(out_of_voc / (pos_total_word + num_words))

        neg_smoothing = np.log(out_of_voc / (neg_total_word + num_words))

        

    output = predict(test_vec, p_pos, p_pos_word_vector, p_neg, p_neg_word_vector, pos_smoothing, neg_smoothing)

    pred_vali.append(output)
# evaluate model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



print('Accuracy score: %s' % (accuracy_score(yvali, pred_vali)))

print('Classificatin report: ')

print(classification_report(yvali, pred_vali))

print('Confusion matrix: ')

print(confusion_matrix(yvali, pred_vali))