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
import pandas as pd
import numpy as np 

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')  
# 编码相关阅读http://blog.csdn.net/robertcpp/article/details/7837712 

# 查看数据
df.head()
# 查看v2的样本
df.v2.head()
# 查看v1的样本
df.v1.head()
# 查看数据的纬度
df.shape
from sklearn.model_selection import train_test_split

# 把数据拆分成训练集和测试集
# train_test_split (X, Y, test_size=0.2, random_state=0)
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0) 

# 查看训练集样本
print (data_train.head())
# 查看训练集标注
print (labels_train.head())
# 查看训练集的样本个数
print(data_train.shape)
# 查看测试机的样本个数
print(data_test.shape)
from sklearn.feature_extraction.text import CountVectorizer
# 调用库来构造分类器所需的输入数据
vectorizer = CountVectorizer()

#fit_transform一共完成了两件事. fit: build dict (i.e. word->wordID)  transform: convert document (i.e. each line in the file) to word vector 
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform
#fit: 统计单词的总个数，建成一个表，每个单词给一个标号 (这个库内部实现有一个缺陷，会把长度为1的单词给过滤掉了)
#transform:统计每句话每个单词出现的次数

# 用训练集的单词来建立词库，因为测试集的数据在现实场景中属于未知数据，且把训练集每句话词(也就是input doc中的每一行)变成向量形态
data_train_count = vectorizer.fit_transform(data_train)
# 把测试集每句话变成向量形态
data_test_count  = vectorizer.transform(data_test)
# 训练数据纬度 
# 7612 也就是fit过程中建立的词汇表的size
print (data_train_count.shape)
# 测试数据纬度
print (data_test_count.shape)
# 看看这些数据长什么样
# 词汇表 （太长了，我这里注释掉）
print (len(vectorizer.vocabulary_))
#print (vectorizer.vocabulary_)
# print first 3 lines. 
# each line: represent the word verctor for a setence/email content which is just the each line in the input file i.e. spam.csv
# each line: e.g. [0,0,1, 2.....0,0,0] 1表示词典中index 为2的单词，在这一份doc/邮件，中出现1次，2表示典中index 3的单词，在这一份doc/邮件中出现2次
print(data_train_count.toarray()[0:4])
import matplotlib.pyplot as plt # 画图常用库

# 我们来看看单词的分布. 统计每个单词出现的次数
# 因为matrix每一行就代表一个句子里的单词分布, 每个位置上（i.e. column）的数值即表示，dict里对应的index的单词出现的次数. 所以矩阵按列求和就可以求出每个单词在总的spam.csv中出现的次数
occurrence = data_train_count.toarray().sum(axis=0) #把矩阵按列求和
plt.plot(occurrence)
plt.show() # 显示图形， x 轴表示单词的index, y轴表示，dict中对应index的单词出现的次数

# 按照每个词出现的次数从高到低进行排序. get_feature_names 即是dict里面的word.
# build dataframe
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrence':occurrence})
word_freq_df_sort = word_freq_df.sort_values(by=['occurrence'], ascending=False)
word_freq_df_sort.head()
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(data_train_count, labels_train)
predictions = clf.predict(data_test_count)
print(predictions)
from sklearn.metrics import accuracy_score

print (accuracy_score(labels_test, predictions))
from sklearn.metrics import classification_report,confusion_matrix
print (confusion_matrix(labels_test, predictions))
print (classification_report(labels_test, predictions))
from sklearn.model_selection import cross_val_score
# 从df获得全部邮件内容和标注
data_content = df.v2
data_label = df.v1
vect = CountVectorizer()
# 在整体数据集上构建词汇表以及转化成计数格式 Note: 这里不需要train, validation split. cross_val_score will handle this split
data_count = vect.fit_transform(data_content)
# 交叉验证 clf = MultinomialNB()
# cross_val_score(model, X, Y, cv=20, scoring="accuracy")
cross_val = cross_val_score(clf, data_count, data_label, cv=20, scoring='accuracy')
# 打印每组实验测试集的准确率
print (cross_val)
# 求平均值
print (np.mean(cross_val))