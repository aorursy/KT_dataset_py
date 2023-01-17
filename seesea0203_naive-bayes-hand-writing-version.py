import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../input/spam.csv', encoding='latin-1')
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.info())
df.columns
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(df.v2,
                                                                 df.v1,
                                                                 test_size=0.2,
                                                                 random_state=0)
print(data_train.head(), label_train.head())
def GetVocabulary(data):
    voc_set = set()
    for email in data:
        words = email.split()
        for word in words:
            voc_set.add(word)
    return list(voc_set)
vocab_list = GetVocabulary(data_train)
print('Total number of unique words: ', str(len(vocab_list)))
def Document2Vector(vocab_list, data):
    word_vectors = []
    for document in data:
        word_vector = np.zeros(len(vocab_list))
        words = document.split()
        for word in words:
            if word in vocab_list:
                word_vector[vocab_list.index(word)] += 1
        word_vectors.append(word_vector)
    return word_vectors
data_train_vectors = Document2Vector(vocab_list, data_train.values)
print(len(data_train_vectors), len(data_train_vectors[0]))
def NaiveBayes_train(word_vectors, label_train):
    num_docs = len(word_vectors)
    num_words = len(word_vectors[0])
    
    ham_vector_cnt = np.ones(num_words)
    spam_vector_cnt = np.ones(num_words)
    ham_total_cnt = num_words
    spam_total_cnt = num_words # Laplacian Smoothing -- Improve algorithm (avoid the situation that the probability is 0)
    
    ham_count = 0
    spam_count = 0
    
    for i in range(num_docs):
        if i % 500 == 0:
            print('Train on the document ID: ', str(i))
        
        if label_train[i] == 'ham':
            ham_vector_cnt += word_vectors[i]
            ham_total_cnt += word_vectors[i].sum()
            ham_count += 1
        else:
            spam_vector_cnt += word_vectors[i]
            spam_total_cnt += word_vectors[i].sum()
            spam_count += 1
    print(ham_count, spam_count)
    p_ham_vector = np.log(ham_vector_cnt/ham_total_cnt)
    p_spam_vector = np.log(spam_vector_cnt/spam_total_cnt)
    
    p_ham = np.log(ham_count/num_docs)
    p_spam = np.log(spam_count/num_docs)
    
    return p_ham_vector, p_ham, p_spam_vector, p_spam

p_ham_vector, p_ham, p_spam_vector, p_spam = NaiveBayes_train(data_train_vectors, label_train.values)           
data_test.values.shape
def Predict(test_word_vector, p_ham_vector, p_ham, p_spam_vector, p_spam):
    spam = (test_word_vector * p_spam_vector).sum() + p_spam
    ham = (test_word_vector * p_ham_vector).sum() + p_ham
    
    if spam > ham:
        return 'spam'
    else:
        return 'ham'
data_test_vectors = Document2Vector(vocab_list, data_test.values)
predictions = []
for i in range(len(data_test_vectors)):
    if i % 200 == 0:
        print('Predict on the document ID: ', str(i))
    pred = Predict(data_test_vectors[i], p_ham_vector, p_ham, p_spam_vector, p_spam)
    predictions.append(pred)

print(len(predictions))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy: \n', accuracy_score(label_test, predictions), '\n')
print('Confusion Matrix: \n', confusion_matrix(label_test, predictions), '\n')
print('Classification Report: \n', classification_report(label_test, predictions))