from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_dir = "../input/"

df  = pd.read_csv(data_dir + "spam.csv", encoding = 'latin-1')

df.shape
data_train, data_test, label_train, label_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state = 0)

train_test_split( train, label )
print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)

def GetVocabulary(data):
    vocab_set = set([])
    
    for document in data: # for 文章
        words = document.split()
        for word in words:
            vocab_set.add(word)
            
    return list(vocab_set)
    
vocab_list = GetVocabulary(data_train) 

# vocab_list = GetVocabulary(["I love you", "i hate you", "you bad bad"]) 
# print(vocab_list)
print(len(vocab_list))
def Document2Vector(vocab_list, data):
    word_vector = np.zeros(len(vocab_list))
    words = data.split()
    for word in words:
        if word in vocab_list:
            word_vector[vocab_list.index(word)] += 1

    return word_vector
    
x = Document2Vector(vocab_list, "I love you")

vocab_list.index('father')
train_matrix = []
for document in data_train.values:
    word_vector = Document2Vector(vocab_list, document)
    train_matrix.append(word_vector)
len(train_matrix) # n行m列 , n是文章数目， m个单词个数
label_train
def NaiveBayes_train(train_matrix, label_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    # 统计每一个单词的概率
    
    spam_vector_count = np.ones(num_words)
    ham_vector_count = np.ones(num_words)
    spam_total_count = num_words
    ham_total_count = num_words
    
    spam_count = 1
    ham_count = 1
    for i in range(num_docs):
#         如果当前这一句话是spam邮件
        if label_train[i] == 'spam':
            spam_vector_count += train_matrix[i]
            spam_total_count += sum(train_matrix[i])
            spam_count += 1
        else:
#             如果当前这句话是ham邮件
            ham_vector_count += train_matrix[i]
            ham_total_count += sum(train_matrix[i])
            ham_count += 1
            
    
    p_spam_vector = spam_vector_count / spam_total_count
    p_ham_vector = ham_vector_count / ham_total_count
    
    
    # 统计每一个类别的概率
    p_spam = spam_count / (num_docs+1)
    p_ham = ham_count / (num_docs+1)
    
    return np.log(p_spam_vector), np.log(p_ham_vector), np.log(p_spam), np.log(p_ham)


p_spam_vector, p_ham_vector, p_spam, p_ham = NaiveBayes_train(train_matrix, label_train.values)
print (np.ones(5))
print ( np.zeros(5))
# 如何做测试
def predict(test_word_vector, p_spam_vector, p_ham_vector, p_spam, p_ham ):
    spam = sum(test_word_vector * p_spam_vector) + p_spam # spam的概率
    ham = sum(test_word_vector * p_ham_vector) + p_ham # ham的概率
    
    if spam > ham:
        return 'spam'
    else:
        return 'ham'
    
predictions = []
for document in data_test.values:
    test_word_vector = Document2Vector(vocab_list, document) # 拿到test每句话的词向量
    pred = predict(test_word_vector, p_spam_vector, p_ham_vector, p_spam, p_ham)
    predictions.append(pred)
    
# print(predictions)
data_test.values

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(label_test, predictions))

print(classification_report(label_test, predictions))

print(confusion_matrix(label_test, predictions))