import pandas as pd

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt # 画图常用库



data_dir = "../input/"



df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')  

# 编码相关阅读http://blog.csdn.net/robertcpp/article/details/7837712 



# 查看数据

print (df.head())

df.shape
from sklearn.model_selection import train_test_split



# split into train and test

data_train, data_test, labels_train, labels_test = train_test_split(

    df.v2,

    df.v1, 

    test_size=0.2, 

    random_state=0) 



print ('拆分过后的每个邮件内容')



print (data_train.head())

print (labels_train.head())
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()



data_train_count = vectorizer.fit_transform(data_train)

data_test_count  = vectorizer.transform(data_test)

# print ('统计每个单词出现的频率')

# print (data_train_count.shape)

# print (vectorizer.vocabulary_)

# print (data_test_count.shape)



import pandas as pd 

import matplotlib.pyplot as plt # 画图常用库

vect = CountVectorizer()

example = ['I love you, good bad bad', 'you are soo good']



result = vect.fit_transform(example)

print(result)

print (vect.vocabulary_)





word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':data_train_count.toarray().sum(axis=0)})

word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])

plt.plot(word_freq_df.occurrences)

plt.show() # 显示图形





word_freq_df_sort = word_freq_df.sort_values(by=['occurrences'], ascending=False)

word_freq_df_sort.head()



# 查看发现train 和 test 的feature不一致



print (data_train_count.shape, labels_train.shape, data_test_count.shape)


from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



clf = MultinomialNB()

clf.fit(data_train_count, labels_train)

predictions = clf.predict(data_test_count)

print(predictions)

# 检测模型



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score





print (accuracy_score(labels_test, predictions))

print (classification_report(labels_test, predictions))

print (confusion_matrix(labels_test, predictions))





cross_val = cross_val_score(clf, data_train_count, labels_train, cv=20, scoring='accuracy')

print (cross_val)

print (np.mean(cross_val))