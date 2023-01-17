import os

from time import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from pprint import pprint

import matplotlib.pyplot as plt

import matplotlib as mpl

print(os.listdir('../input'))
%%time

# 打印20个类别名称

data_train = fetch_20newsgroups(subset='train', data_home='../input/')

categories = data_train.target_names

pprint(categories)
print(len(data_train.data))
data_train = fetch_20newsgroups(subset='test', data_home='../input/')

print(len(data_train.data))
# choose five categories

# 在这里，我们选择5个类别来训练模型

categories = ('alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'rec.motorcycles')

# 可以选择删除‘headers’, ‘footers’, ‘quotes’,我们选择删除 'headers'

remove = ('headers')

# load train data

data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0, remove=remove,

                                data_home='../input/')

# 加载同样类别的测试数据

data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=0, remove=remove,

                                data_home='../input/')
# 输出训练数据和测试数据数量

print('data_train nums:{}  data_test nums:{}'.format(len(data_train.data), len(data_test.data)))
categories = data_train.target_names

# 打印 类别名称 与 其对应的 label

for i,v in enumerate(categories):

    print('label值：{}   类别名称：{}'.format(i, v))
# 获取训练集 label

y_train = data_train.target

# 获取测试集 label

y_test = data_test.target
# 使用 tfidf 进行词袋模型转换

# 实例化一个 vectorizer(向量化工具)

vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.6, sublinear_tf=True)

# 将训练集转化为词袋模型向量

x_train = vectorizer.fit_transform(data_train.data)

# 将训练集的 idf 应用到测试集

x_test = vectorizer.transform(data_test.data)

print('train_set shape{}    test_set shape{}'.format(x_train.shape, x_test.shape))
# 打印停止词

print(vectorizer.get_stop_words())
print(vectorizer.get_feature_names)
clfs = (BernoulliNB(), # 伯努利朴素贝叶斯

        MultinomialNB(),

        LogisticRegression(random_state=0, solver='lbfgs')

       )

def test_clf(clf):

    print('classifier:', clf)

    alpha = np.logspace(-4, 5, 6)

    num = alpha.size

    # 使用GridSearchCV,5折交叉验证选取最佳参数

    model = GridSearchCV(clf, param_grid={}, cv=5)

    if hasattr(clf, 'alpha'): # 朴素贝叶斯

        model.set_params(param_grid={'alpha': alpha})

    elif hasattr(clf, 'tol'): # logistic regression

        model.set_params(param_grid={'tol': alpha})

    

    start_time = time()

    model.fit(x_train, y_train)

    end_time = time()

    train_time = (end_time - start_time) / (5*num)

    print('best params:{}', model.best_params_)

    

    start_time = time()

    y_hat = model.predict(x_test)

    end_time = time()

    test_time = (end_time - start_time)

    acc_score = metrics.accuracy_score(y_test, y_hat)

    print(acc_score)

    return train_time, test_time, acc_score, str(clf).split('(')[0]
result = []

for clf in clfs:

    r = test_clf(clf)

    result.append(r)

    print('\n')

result = np.array(result)

time_train, time_test, acc_score, names = result.T

time_train = time_train.astype(np.float)

time_test = time_test.astype(np.float)

acc_score = acc_score.astype(np.float)

x = np.arange(len(time_train))

print('x=',x)

#mpl.rcParams['font.sans-serif'] = [u'simHei']

#mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 7), facecolor='w')

ax = plt.axes()

b1 = ax.bar(x, acc_score, width=0.25, color='#77E0A0')

ax_t = ax.twinx()

b2 = ax_t.bar(x+0.25, time_train, width=0.25, color='#C108ED')

b3 = ax_t.bar(x+0.5, time_test, width=0.25, color='#FF8080')

plt.xticks(x+0.5, names)

plt.legend([b1[0], b2[0], b3[0]], ('acc_acore', 'train_time', 'test_time'), loc='center left', shadow=True)

plt.title('Comparisons among different classifiers of 20news', fontsize=18)

plt.xlabel('name of classifier')

plt.grid(True)

plt.tight_layout(2)

plt.show()
