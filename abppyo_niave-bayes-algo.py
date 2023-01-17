from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np



#读取数据

data_dir = "../input"

df = pd.read_csv(data_dir + '/spam.csv', encoding = 'latin-1')



#拆分数据为训练集与测试集

data_train, data_test, labels_train, labels_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state=0)



#print('拆分过后的每个邮件内容')

print(data_train[:10])

#print("拆分过后每个邮件是否是垃圾邮件")

print(labels_train[0:10])

#用一个dictionary保存词汇，给每个词汇赋唯一id



def GetVocabulary(data):

    vocab_dict = {}

    wid = 0

    for document in data:

        words = document.split()

        for word in words:

            word = word.lower()

            if word not in vocab_dict:

                vocab_dict[word] = wid

                wid += 1

    return vocab_dict



#用训练集建立词汇表

vocab_dict = GetVocabulary(data_train)

print('Number of all the unique words:' + str(len(vocab_dict.keys())))
#把文本变成向量的表示形式，以便计算



def Document2Vector(vocab_dict, data):

    word_vector = np.zeros(len(vocab_dict.keys()))

    words = data.split()

    for word in words:

        word = word.lower()

        if word in vocab_dict:

            word_vector[vocab_dict[word]] += 1 

    return word_vector



#解释向量输出例子

example = Document2Vector(vocab_dict, "we are good good")

print(example)

print(example[vocab_dict['we']],example[vocab_dict['are']],example[vocab_dict['good']])

#把训练集df变成向量形式

train_matrix = []

for document in data_train.values:

    word_vector = Document2Vector(vocab_dict, document)

    train_matrix.append(word_vector)



print(len(train_matrix))

print(train_matrix[:10])

print(np.array(train_matrix).shape)
#训练计算两个概率：

#1.词在每个分类下的概率 P（email/spam)

#2. 每个分类的概率 P（spam)



def NaiveBayes_train(train_matrix, labels_train):

    num_docs = len(train_matrix)

    num_words = len(train_matrix[0])

    spam_word_counter = np.ones(num_words)

    ham_word_counter = np.ones(num_words)

    

    ham_total_count = 0

    spam_total_count = 0

    

    spam_count = 0

    ham_count = 0

    

    for i in range(num_docs):

        if i%500 == 0:

            print("train on the doc id: " + str(i))

        

        if labels_train[i] == 'ham':

            ham_word_counter += train_matrix[i]

            ham_total_count += sum(train_matrix[i])

            ham_count += 1

        else: 

            spam_word_counter += train_matrix[i]

            spam_total_count += sum(train_matrix[i])

            spam_count += 1

            

#对概率取log



    p_spam_vector = np.log(spam_word_counter/(spam_total_count + num_words)) #注意在分母也加上平滑部分

    p_ham_vector = np.log(ham_word_counter/(ham_total_count + num_words))

    return p_spam_vector, np.log(spam_count/num_docs), p_ham_vector, np.log(ham_count/num_docs)

                         

#train



p_spam_vector, p_spam, p_ham_vector, p_ham =NaiveBayes_train(train_matrix, labels_train.values)

                      
#对测试集进行预测，计算随机emil单词两个分类下的概率，选择较大者作为分类结果



def Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham):



    spam = sum(test_word_vector * p_spam_vector) + p_spam

    ham = sum(test_word_vector * p_ham_vector) + p_ham

    if spam > ham:

        return 'spam'

    else:

        return 'ham' 



predictions = []

i = 0

for document in data_test.values:

    if i%100 == 0:

        print('test training on doc:' + str(i))

    i += 1

    test_word_vector = Document2Vector(vocab_dict, document)

    ans = Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)

    predictions.append(ans)

    

print(len(predictions))



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score



print(accuracy_score(labels_test, predictions))

print(classification_report(labels_test, predictions))

print(confusion_matrix(labels_test, predictions))
