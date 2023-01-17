import numpy as np
import pandas as pd
import matplotlib as plt
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
data_dir = '../input/'
df = pd.read_csv(data_dir + 'spam.csv', encoding='latin-1')
print(df.head()) # v1 (column 0) = labels, v2 (column 1) = email contents
print(df.shape) # (5572, 5)
from sklearn.model_selection import train_test_split

data_train, data_test, labels_train, labels_test = train_test_split(df.v2,
                                                                    df.v1,
                                                                   test_size=0.2,
                                                                   random_state=0)
print(data_train.shape)
print(data_test.shape)
print(labels_train.shape)
print(labels_test.shape)
def preprocessing_word(word):
    word = word.lower() # lowercase
#     word = re.sub("[^a-zA-Z0-9]"," ", word) # tokenization
#     word = wordnet_lemmatizer.lemmatize(word) # lemmatization
#     if word not in stopwords.words('english'): # remove stop words
#         return word
    return word

def generate_vocabulary(documents):
    vocabulary = {}
    for document in documents:
        words = document.split()
        for index, word in enumerate(words):
            word = preprocessing_word(word)
            if word not in vocabulary and word is not None:
                vocabulary[word] = index
    return vocabulary

vocabulary = generate_vocabulary(data_train)
print(len(vocabulary.keys()))
print(list(vocabulary.items())[:5])
def document_2_vector(vocabulary, document):
    document_vector = np.zeros(len(vocabulary.keys()))
    words = document.split()
#     new_words = []
    for word in words:
        word = preprocessing_word(word)
        if word in vocabulary and word is not None:
            document_vector[vocabulary[word]] += 1
#         elif word is not None:
#             print(word)
#             new_words.append(word)
#     return document_vector, new_words
    return document_vector

def documents_2_matrix(vocabulary, documents):
    train_matrix = []
#     new_words_matrix = []
    for document in documents:
#         document_vector, new_words = document_2_vector(vocabulary, document)
        document_vector = document_2_vector(vocabulary, document)
        train_matrix.append(document_vector)
    return train_matrix

# example = document_2_vector(vocabulary, "we are good good student students hfsdkjhiuhe")
# print(example)
# print(example[vocabulary['good']], example[vocabulary['student']])

train_matrix = documents_2_matrix(vocabulary, data_train.values)
print(len(train_matrix))
def naive_bayes_train(train_matrix, labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    
    # 1darray, each cell represents the occurrence of each word as spam/ham
    spam_word_counter = np.ones(num_words)
    ham_word_counter = np.ones(num_words)
    
    # total number of spam/ham words
    spam_total_count = 0
    ham_total_count = 0
    
    # num of spam/ham docs
    spam_count = 0
    ham_count = 0
    
    for i in range(num_docs):
        if labels_train[i] == 'ham':
            ham_word_counter += train_matrix[i]
            ham_total_count += sum(train_matrix[i])
            ham_count += 1
        else:
            spam_word_counter += train_matrix[i]
            spam_total_count += sum(train_matrix[i])
            spam_count += 1
    
    # spam/ham probability for each word with Laplace Smoothing
    p_spam_vector = np.log(spam_word_counter / (spam_total_count + num_words))
    p_ham_vector = np.log(ham_word_counter / (ham_total_count + num_words))
    
    p_spam = np.log(spam_count / num_docs)
    p_ham = np.log(ham_count / num_docs)
    
    return p_spam_vector, p_spam, p_ham_vector, p_ham

p_spam_vector, p_spam, p_ham_vector, p_ham = naive_bayes_train(train_matrix, labels_train.values)
def predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham):
    spam = sum(test_word_vector * p_spam_vector) + p_spam
    ham = sum(test_word_vector * p_ham_vector) + p_ham
    return 'spam' if spam > ham else 'ham'

predictions = []
for document in data_test.values:
    test_word_vector = document_2_vector(vocabulary, document)
    ans = predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)
    predictions.append(ans)

print(len(predictions))
print(accuracy_score(labels_test, predictions))
print(classification_report(labels_test, predictions))
print(confusion_matrix(labels_test, predictions))
