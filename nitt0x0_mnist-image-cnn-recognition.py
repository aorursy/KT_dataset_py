# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import random
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/train.csv")      # pandas.core.frame.DataFrame
df2 = pd.read_csv("../input/test.csv")       # pandas.core.frame.DataFrame
TRAIN_EPOCH = 15
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
df1_len = len(df1)  
df2_len = len(df2)                       
print("1. train set : {:,} each".format(df1_len))
print("2. test set  : {:,} each".format(df2_len))
# 데이터셋과 트레인셋이 같은지 비교해 보니 'False'
print(list(df1.columns) == list(df2.columns))
print(df1.columns)   # 785 컬럼
print(df2.columns)   # 784 컬럼 (레이블 데이터 삭제 됨)
df1.describe()       # 42,000 개
df2.describe()       # 28,000 개
appear = []                        # 숫자의 출현횟수 카운트
for i in range(10):
    count = list(df1.loc[:,['label']].values.reshape(42000,)).count(i)
    appear.append(count)
    print("{} = {:,}".format(i,count))

print("\nmax: {} 번 = {:,} 회 \nmin: {} 번 = {:,} 회 \nmax-min: {} 개 차이".format(
    appear.index(max(appear)), max(appear), 
    appear.index(min(appear)),min(appear), 
    max(appear) - min(appear)))

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


f, ax = plt.subplots(figsize=(20,5))        # 아 뭐야? 비슷한데, even 하지는 않네.. 데이터 'flaw 가 있는건가?
sns.distplot(df1['label'])
# 판다스 데이터프레임을 넘파이 어레이로 바꾼다
df1_arr = df1.values
print(df1.shape, df1_arr.shape, "  ...  둘 다 어레이 갯수는 동일하다, 저장형식만 다를 뿐...")

# 1개의 '라벨'값 42,000을 슬라이싱 한다
Y_train = df1_arr[:,0:1]
print("라벨값 :", Y_train.shape)

# 동일하게 784개 '픽셀'데이터를 42,000개 슬라이싱하고,
X_train = df1_arr[:,1:]
print("학습값 :", X_train.shape)

# ---------------------------------------------------------------------------------------------
# 이것을 다시, [42000, 28, 28] 로 차원을 바꾼다   ... 아니, 이것은 하지 않는다 (필요 할 때만 바꾼다.)
# X_train = X_train.reshape(-1, 28, 28)
# print(X_train.shape)
# 42,000개의 [28,28]데이터중 10 개를 랜덤으로 뽑아서 '라벨과 비교해 본다'
# 먼저, 그룹이미지의 외곽 프레임을 [13 x 6] 로 사이징 고정한다.   ...   이 안에 이미지 10개를 우겨넣는 방식
fig = plt.figure(figsize=(13, 6))

for i in range(10):
    rand_num = random.randint(0, 4200-1)            # 0 ~ 4199 개의 데이터 중 '랜덤' 셀렉트!    ... 어짜피, 인생은 랜덤 셀렉트!
    
    # 폭에 맞추어서, [2,5]열로 배치 한다
    fig.add_subplot(2, 5, i+1)
    plt.title("%s / Label %s"% (rand_num, Y_train[rand_num]) )
    plt.imshow(X_train[rand_num].reshape(28,28), cmap='Greys', interpolation='nearest')
    # (!) 여기서, X_train array는 (?,784)  --> (?, 28, 28)로 바꿔야, imshow()가 가능하다.
    
plt.show()    # 배치가 다 끝났으면, 보여준다.
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# csv의 784 개 데이터 [?,784]를 컨볼루션을 위한 매트릭스로 변형  .. [?,28,28,1]
X_img = tf.reshape(X, [-1, 28, 28, 1])   

print(X)       # [?, 783] ... [?, 28, 28]
print(X_img)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) 

# 손실 없이 32개의 필터로 밀었을 때, [?, 28,28,32]
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
print(L1)

# '소프트맥스' 대신 '렐루' 활성화 함수()를 이용하여, 스코어(0~254)를 Probablity 값(0~1)으로 변환  .. [?, 28,28,32]
L1 = tf.nn.relu(L1)
print(L1)

# '맥스풀링'을 통해서, 피처드 벨류로 [14,14,32] 로 압축함.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L1)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) #  

L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
print(L2)
L2 = tf.nn.relu(L2)
print(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L2)

L2r = tf.reshape(L2, [-1, 7 * 7 * 64])               # Tensor("Reshape_5:0", shape=(?, 3136), dtype=float32)
print(L2r)

W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L2r, W3) + b
# ---- 최초이미지(X)(그레이 스케일로 reshape 한 것) --------
print(X_img) # shape=(?, 28, 28, 1)

# ---- 첫번째 레이어 #1 -------------------------------------
print(W1)    # shape=(3, 3, 1, 32)
print(L1)    # shape=(?, 14, 14, 32)

# ---- 두번째 레이어 #2 -------------------------------------
print(W2)    # shape=(3, 3, 32, 64)
print(L2)    # shape=(?, 7, 7, 64)
print(L2r)   # shape=(?, 3136)  ...  FC 레이어에 맞는 '압축' 필요(Flatten)

# ---- FC 레이어 (DENSE 레이어):피쳐드맵을 넣어서 결과를 판단
print(W3)    # shape=(3136, 10)
print(b)     # shape=(10,)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
seq = [[0.11], [0.2], [0.3], [0.44], [0.51]]
res = tf.argmax(seq, axis=1)
with tf.Session() as sess:
    res_val = sess.run(res)
    print(res_val)
print("학습 입력값(X_train)의 디멘젼", X_train.shape)
print("학습 라벨값(Y_train)의 디멘젼", Y_train.shape)
# (42_000, 1 )을 (42_000, 10) 으로 바꾸는 함수
y = np.arange(42_000*10).reshape((42_000,10))
y = np.zeros_like(y)

for i in range(42_000):
    #print(int(Y_train[i]))
    y[i][int(Y_train[i])] = 1

print(y.shape)        # (42000, 10)
y
# 다른 방법으로 간단하게 '원핫'을 쓴다
Y_train.shape                                   # (42000, 1)

res = tf.one_hot(Y_train, depth=10, axis=1)     # (42000, 1)
with tf.Session() as sess:
    res_val = sess.run(res)
    res_val1 = res_val.reshape((-1,10))         # (42000, 10, 1)  --> (42000, 10) 
    # print(res_val[:10,:,:].reshape(10,10))
    print("res_val1 =", res_val1.shape)
sess = tf.Session()

INIT = tf.global_variables_initializer()
# Initialize TensorFlow variables @ Hyper Parameter Definition 'ABOVE'
sess.run(INIT)

# Training cycle
total_batch = int(len(X_train) / BATCH_SIZE)            # 420번

for epoch in range(TRAIN_EPOCH):
    avg_cost = 0
    batch_start = 0
    
    for i in range(total_batch):
        batch_end = batch_start + (BATCH_SIZE - 1)     # 0 ~ 99 / 100 ~ 199 / 200 ~ 299 ...
        batch_xs = X_train[batch_start:batch_end+1,:]
        #batch_ys = Y_train[batch_start:batch_end+1,:]
        batch_ys = y[batch_start:batch_end+1,:]

        # for batch_x, batch_y in zip(batch_xs, batch_ys):
        cost_val, _, hypo_val, accu_val = sess.run([cost, train, hypothesis, accuracy ], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += cost_val / total_batch
        # batch_start = batch_end + 1

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    # print("batch_start =", batch_start)
    # print("", total_batch)

print("   ...... Learning finished ......\n\n")
print("Hypothesis = [%s,%s]" % hypo_val.shape)      # 마지막 저장된 배치(100개)의 답 [100,10] = 스코어로 나오기 때문에 ONE-HOT으로 1개를 추출해야 함.
print("Accuracy = %s" % accu_val)                   # 아큐러시는 보나마나, 1이군..
print("Final cost = %.9f %%" % (avg_cost*100))
# 판다스 데이터프레임을 넘파이 어레이로 바꾼다  -- '분해는 조립의 역순이다'.   .. 음.. 좀 이상하다..
# 기억 날랑가? (첫번째 줄, 판다스 데이터 프레임)    df2 = pd.read_csv("../input/test.csv")
df2_arr = df2.values
print(df2.shape, df2_arr.shape, "  ...  둘 다 어레이 갯수는 동일하다, 저장형식만 다를 뿐...")

# 테스트 데이터 셋에는 '라벨'이 없으므로, 전체가 검증데이터 이다 [28k,784]
# 동일하게 784개 '픽셀'데이터를 28,000개 슬라이싱하고,
X_test = df2_arr[:,:]
print("검증값 :", X_test.shape, "가 만들어 졌습니다... 시험 보러 갑시다~")
indices = [0, 2, -1, 1]
depth = 3
res = tf.one_hot(indices, depth, on_value=5.0, off_value=0.0, axis=-1)
with tf.Session() as sess:
    print(sess.run(res))
indices = [0, 1, 2]
depth = 3
res = tf.one_hot(indices, depth)
with tf.Session() as sess:
    print(sess.run(res))
res = tf.one_hot(indices=[random.randint(0,9) for i in range(10)], depth=10)
with tf.Session() as sess:
    print(sess.run(res))