# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np



# 读取数据

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')



# 把数据拆分成为训练集和测试集

data_train, data_test, labels_train, labels_test = train_test_split(

    df.v2,

    df.v1, 

    test_size=0.2, 

    random_state=0)  



print ('拆分过后的每个邮件内容')

print (data_train[:10])

print ('拆分过后每个邮件是否是垃圾邮件')

print (labels_train[:10])
'''

    用一个dictionary保存词汇，并给每个词汇赋予唯一的id

'''

def GetVocabulary(data): 

    vocab_dict = {}

    wid = 0

    for document in data: # document represent each line in the input spam.csv file

        words = document.split() # split each document(i.e. each line) 按空格分词 “I am a student” => ["I", "am", "a", "student"]

        for word in words:

            word = word.lower() #归一化

            if word not in vocab_dict:

                vocab_dict[word] = wid

                wid += 1

    return vocab_dict



# 用训练集建立词汇表

vocab_dict = GetVocabulary(data_train)

print ('Number of all the unique words : ' + str(len(vocab_dict.keys())))
'''

    把文本变成向量的表示形式，以便进行计算

'''

# 把上一步骤建立的好vocab_dict传入，

def Document2Vector(vocab_dict, data):

    word_vector = np.zeros(len(vocab_dict.keys()))

    words = data.split()

    out_of_voc = 0 # track没有在词汇表里出现过的单词， 这样后面好做smoothing

    for word in words:

        word = word.lower()

        if word in vocab_dict:

            word_vector[vocab_dict[word]] += 1

        else:

            out_of_voc += 1

    return word_vector, out_of_voc



# 下面是一个例子，解释向量长什么样

example, oov = Document2Vector(vocab_dict,"we are good good student")

print(example)

print(example[vocab_dict['we']], example[vocab_dict['are']], example[vocab_dict['good']], example[vocab_dict['student']])

# 每个单词是一个维度，如果单词没有出现过，对应那一维为0，否则为出现的次数.

data_train.values # data_train.values 不包含第一列line number
# 把训练集的句子全部变成向量形式

train_matrix = []

for document in data_train.values: # document represent each line in the input file

    word_vector, _ = Document2Vector(vocab_dict, document) # _ 表示第二个返回值 用不到

    train_matrix.append(word_vector)



print (len(train_matrix))

train_matrix[:8]


'''

    在训练集计算两种概率：

        1. 词在每个分类下的概率，比如P('email'|Spam)

        2. 每个分类的概率，比如P(Spam)

        

    这里的计算实现巧妙利用了numpy的array结构：

        1. 在每个分类下创建一个与词汇量大小相等的vector(即 numpy array), 即spam_word_counter 和 ham_word_counter

        2. 在遍历每一个句子的时候，直接与句子对应的vector相加，累积每个单词出现的次数

        3. 在遍历完所有句子之后，再除以总词汇量，得到每个单词的概率

'''

def NaiveBayes_train(train_matrix,labels_train):

    # train_matrix => (10，1000)

    num_docs = len(train_matrix) # 多少行 就有多少个doc 

    num_words = len(train_matrix[0]) #对第一个样本取一下vector的长度. 每一行的长度就是word dict的长度，也就表示有多少个word 在dict里

    

    spam_word_counter = np.ones(num_words) # np.ones 其实默认做了 +1 smoothing

    ham_word_counter = np.ones(num_words)  #计算频数初始化为1，即使用拉普拉斯平滑，详单于在最开始就给每个单词加了一个平滑因子



    ham_total_count = 0;

    spam_total_count = 0;

    

    spam_count = 0

    ham_count = 0

    # 一个文件一个文件处理，也就是输入文件的每一行

    for i in range(num_docs):

        if i % 500 == 0:

            print ('Train on the doc id:' + str(i))

            

        if labels_train[i] == 'ham':

            ham_word_counter += train_matrix[i] # 数组对应位置相加

            ham_total_count += sum(train_matrix[i]) # 数字相加

            ham_count += 1

        else:

            spam_word_counter += train_matrix[i]

            spam_total_count += sum(train_matrix[i])

            spam_count += 1

    

    #spam_word_counter => 每个词的计数， 是array

    #spam_total_count => Spam的总词数,是一个数字

    #spam_count => Spam邮件计数， 是一个数字

    

    # 注意，这里对所有的概率都取了log

    p_spam_vector = np.log(spam_word_counter/(spam_total_count + num_words)) #注意在分母也加上平滑部分

    p_ham_vector = np.log(ham_word_counter/(ham_total_count + num_words))  #注意在分母也加上平滑部分

    

    return p_spam_vector, np.log(spam_count/num_docs), spam_total_count, p_ham_vector, np.log(ham_count/num_docs), ham_total_count



# p_spam_vector/p_ham_vector 的每一维分别是一个单词在spam/ham分类下的概率

# p_spam / p_ham 分别是两个分类的概率

p_spam_vector, p_spam, spam_total_count, p_ham_vector, p_ham, ham_total_count = NaiveBayes_train(train_matrix, labels_train.values)
'''

    对测试集进行预测，按照公式计算例子在两个分类下的概率，选择概率较大者作为预测结果

'''

def Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham, spam_smoothing, ham_smoothing):

    

    # 注意: 如果单词没出现过，则test_word_vector对应的维度为0

    # 所以: test_word_vector * p_spam_vector 不为0的维度正好是句子中每个词的概率

    # [2, 0, 1] * [0.3, 0.2, 0.4] = sum([0.6, 0, 0.4]) 

    # 一个单词可能出现多余一次，哟与test_word_vector 里存放的即使每个单词出现的次数，所以 sum(test_word_vector * p_spam_vector) 正好达到公式的要求 i.e. 

    # 原始公式 2log(P(A|Spam))+ log(P(B|Spam))  对照Week3.Session2.Naive_ Bayers_1_V3.0.pdf 中 ”朴素贝叶斯的工程技巧 – 概率处理技巧“ 那一页 好理解

    spam = sum(test_word_vector * p_spam_vector) + p_spam + spam_smoothing

    ham = sum(test_word_vector * p_ham_vector) + p_ham + ham_smoothing

    if spam > ham:

        return 'spam'

    else:

        return 'ham'



predictions = []

num_words = len(vocab_dict.keys())

i = 0

for document in data_test.values: # data_test.values 是array

    if i % 200 == 0:

        print ('Test on the doc id:' + str(i))

    i += 1    

    test_word_vector, out_of_voc = Document2Vector(vocab_dict, document)

    spam_smoothing = 0

    ham_smoothing = 0

    if out_of_voc != 0:

        spam_smoothing = out_of_voc * np.log(1/(spam_total_count + num_words))

        ham_smoothing = out_of_voc * np.log(1/(ham_total_count + num_words))

    ans = Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham, spam_smoothing, ham_smoothing)

    predictions.append(ans)



print (len(predictions))
# 检测模型



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score





print (accuracy_score(labels_test, predictions))

print (classification_report(labels_test, predictions))

print (confusion_matrix(labels_test, predictions))