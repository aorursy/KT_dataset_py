from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')
print ('show what kind of data we are dealing with')
print (df.head())

# split into train and test
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0)   
print('')
print ('Now print each SMS text after train/test split')
print (data_train[:10])
print('')
print ('Now print labels of SMS texts')
print (labels_train[:10])

def GetVocabulary(data): 
    vocab_set = set([])
    for document in data:
        words = document.split()
        for word in words:
            vocab_set.add(word) 
    return list(vocab_set)

vocab_list = GetVocabulary(data_train)
print ('Number of all the unique words : ' + str(len(vocab_list)))



def Document2Vector(vocab_list, data):
    word_vector = np.zeros(len(vocab_list))
    words = data.split()
    for word in words:
        if word in vocab_list:
            word_vector[vocab_list.index(word)] += 1
    return word_vector

print (data_train[1:2,])
ans = Document2Vector(vocab_list,"the the the")
print (data_train.values[2])

train_matrix = []
for document in data_train.values:
    word_vector = Document2Vector(vocab_list, document)
    train_matrix.append(word_vector)

print (len(train_matrix))
# this is not using SKlearn model
def NaiveBayes_train(train_matrix,labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    
    spam_vector_count = np.ones(num_words);
    ham_vector_count = np.ones(num_words)  #initialize the count as 1 for each word
    spam_total_count = num_words;
    ham_total_count = num_words                  #this is Laplacian smooth
    
    spam_count = 0
    ham_count = 0
    for i in range(num_docs):
        if i % 500 == 0:
            print ('Train on the doc id:' + str(i))
            
        if labels_train[i] == 'spam':
            ham_vector_count += train_matrix[i]
            ham_total_count += sum(train_matrix[i])
            ham_count += 1
        else:
            spam_vector_count += train_matrix[i]
            spam_total_count += sum(train_matrix[i])
            spam_count += 1
    
    print (ham_count)
    print (spam_count)
    
    p_spam_vector = np.log(ham_vector_count/ham_total_count)#return probability vector
    p_ham_vector = np.log(spam_vector_count/spam_total_count)#return a priori probabiligy
    return p_spam_vector, np.log(spam_count/num_docs), p_ham_vector, np.log(ham_count/num_docs)

    
p_spam_vector, p_spam, p_ham_vector, p_ham = NaiveBayes_train(train_matrix, labels_train.values)
data_test.values.shape

    
def Predict(test_word_vector,p_spam_vector, p_spam, p_ham_vector, p_ham):
    
    spam = sum(test_word_vector * p_spam_vector) + p_spam
    ham = sum(test_word_vector * p_ham_vector) + p_ham
    if spam > ham:
        return 'spam'
    else:
        return 'ham'

predictions = []
i = 0
for document in data_test.values:
    if i % 200 == 0:
        print ('Test on the doc id:' + str(i))
    i += 1    
    test_word_vector = Document2Vector(vocab_list, document)
    ans = Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)
    predictions.append(ans)

print (len(predictions))
# now, evaluate the model

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


print (accuracy_score(labels_test, predictions))
print (classification_report(labels_test, predictions))
print (confusion_matrix(labels_test, predictions))
