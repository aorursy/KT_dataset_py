import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
# 데이터 프레임의 형태부터 파악 합니다.
df = pd.read_csv('../input/voice.csv')
print(df.columns, len(df.columns) )
print("총 {:,} 개의 데이터".format(len(df)))       # 3,168개 데이터
df.head(2)
df.tail(2)
# 데이터 20갸, 레이블 1개 (21)번째 칼럼 = male ... female
# 데이터 값의 분포를 봅니다.
# 마지막 '라벨'값이 숫자가 아니라 '스트링' (Male/Female) 입니다. 숫자 1,0 으로 바꿔줍니다.
# '라벨'값을 숫자로 바꾼다, ---> male = 1.0, female = 0.0
if 'female' in set(df['label']) or 'male' in set(df['label']):
    print('[변경전] %s' % set(df['label']), end='')
    df['label'].replace(['male', 'female'], [1., 0.], inplace=True)
    print('  .... [변경후] %s' % set(df['label']))
else: 
    print('이미, 값을 변경 하였습니다. 라벨은 ... %s' % set(df['label']))

df.tail(2)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

print(list(df['label']).count(1.0) == list(df['label']).count(0.0))
print("(남성):{:,} 개 + (여성):{:,}게 = 총 {:,} 개의 데이터".format(
    list(df['label']).count(1.0), 
    list(df['label']).count(0.0), 
    len(df)))
f, ax = plt.subplots(figsize=(20,5))
sns.distplot(df['label'])
# shuffling -- 데이터를 무작위로 섞는다.
df = df.reindex(np.random.permutation(df.index))

df_train = df[:2168]      #  0 ~ 2167 개 = 학습 세트
df_test = df[2168:]       #  1000 개 = 테스트세트

print()
print("학습데이터 : {:,} 개".format(len(df_train)))
print("검증데이터 : {:,} 개".format(len(df_test)))

df_train.describe()
df_test.describe()
df_train.tail(2)
# print(test.head(2))
# print(v_data.tail(2))
# print(test.tail(2))
# v_data.size
# v_data.columns
print(df_train.describe())
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(figsize=(6,5))
sns.heatmap(df_train.corr(), vmax=.8, square=True);
def show_bigger(f1, f2):
    print('{} > {}'.format(stronger_relation(f1, f2)[0], stronger_relation(f1, f2)[1]))

def stronger_relation(f1, f2):       # 헬퍼()
    f1_corr = df_train.corr().loc[f1,'label']
    f2_corr = df_train.corr().loc[f2,'label']
    # print(f1_corr, f2_corr)
    return (f1, f2) if (f1_corr >= f2_corr) else (f2, f1)
show_bigger('meanfreq', 'median')    # median > meanfreq
show_bigger('meanfreq', 'Q25')       # meanfreq > Q25
show_bigger('meanfreq', 'Q75')       # Q75 > meanfreq
show_bigger('meanfreq', 'mode')      # mode > meanfreq
show_bigger('meanfreq', 'centroid')  # meanfreq > centroid

# Q75 > mode > median > meanfreq > centroid > Q25
#   ... 이 계열에선, Q75만 포함시키고 나머지는 드롭(Drop:제거) 한다
#   ... 유사한 영향력을 발휘하는 인자 들, 중에, Q75의 영향력이 가장 크기 때문이다

show_bigger('sd', 'IQR')
show_bigger('sd', 'sfm')

show_bigger('median', 'Q25')
show_bigger('median', 'Q75')
show_bigger('median', 'mode')
show_bigger('median', 'centroid')

show_bigger('Q25', 'centroid')
show_bigger('Q75', 'centroid')
show_bigger('mode', 'centroid')

show_bigger('skew', 'kurt')
show_bigger('sp.ent', 'sfm')

show_bigger('meandom', 'maxdom')
show_bigger('meandom', 'dfrange')

show_bigger('maxdom', 'dfrange')

show_bigger('mode', 'Q75')
show_bigger('IQR', 'sp.ent')

df_train = df_train.drop(['sp.ent', 'mode', 'meanfreq', 'centroid', 'median', 'Q25', 'sd', 'sfm', 'skew', 'sfm', 'dfrange', 'maxdom'], axis=1)
df_test = df_test.drop(['sp.ent', 'mode', 'meanfreq', 'centroid', 'median', 'Q25', 'sd', 'sfm', 'skew', 'sfm', 'dfrange', 'maxdom'], axis=1)
print(len(df_train.columns), df_train.columns)
# 11개를 제거하고 10개 '컬럼'만 남았음..
f, ax = plt.subplots(figsize=(5,4))
sns.heatmap(df_train.corr(), vmax=.8, square=True)
%matplotlib inline
import matplotlib.pyplot as plt

# I think this graph is more elegant than pandas.hist()
# train['SalePrice'].hist(bins=100)
sns.distplot(df_train['kurt'])
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 6, figsize=(15, 7), sharey=True)
for col, a in zip(df_train.columns, axes.flatten()):
    if col == 'label':   
        a.set_title(col)
        a.scatter(df['label'], df['label'])
    else:
        df = df_train[['label', col]].dropna()
        a.set_title(col)
        a.scatter(df[col], df['label'])
# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(743)  # for reproducibility

# collect data
x_data = df_train.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_data = df_train.loc[:,['label']].values
print(x_data[0],y_data[0])
#len(x_data)
type(x_data)
#y_data
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, cost_val)

x_test = df_test.loc[:,['Q75','IQR','kurt','sp.ent']].values
y_test = df_test.loc[:,['label']].values
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test, Y: y_test})
print("Accuracy: ", a)
print(c[0:8], y_test[0:8])