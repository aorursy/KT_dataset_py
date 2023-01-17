# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/0/0"))
f = open("../input/0/0/train.json")
text = [line for line in f]
label = np.array(pd.read_csv("../input/0/0/train.csv")['pred'])
# Any results you write to the current directory are saved as output.
print(label)
tot = len(text)
V = tot // 10
T = tot - V
#tdata, vdata = data[:-V], data[-V,:]
labels = []
for i in range(tot):
    if label[i] == 1:
        labels.append([0, 1])
    else:
        labels.append([1, 0])
label = np.reshape(labels, [tot, 2])

print(np.sum(label))
print(len(label))
dic = {}
for i in range(T):
    doc = text[i].encode('latin-1').decode('unicode_escape')
    for j in doc:
        try:
            dic[j] = dic[j] + 1
        except:
            dic[j] = 1
print(len(dic))
import operator
F = 1000
ind = {}
sorted_dic = sorted(dic.items(),key=operator.itemgetter(1))[-F:]
for i, word in enumerate(sorted_dic):
    ind[word[0]] = i

data = np.zeros([tot, F])
for i in range(tot):
    doc = text[i].encode('latin-1').decode('unicode_escape')
    for word in doc:
        try:
            index = ind[word]
            data[i][index] = data[i][index] + 1
        except:
            pass
    for j in range(F):
        data[i][j] = data[i][j] / len(doc)
    if i % 1000 == 0:
        print(i)
        #print(data[i])
idf = np.zeros([F])
for i in range(tot):
    doc = text[i].encode('latin-1').decode('unicode_escape')
    flag = 0
    subdic = {}
    for word in doc:
        if not subdic.__contains__(word):
            try:
                index = ind[word]
                idf[index] = idf[index] + 1
            except:
                pass
            subdic[word] = 1
    if i % 10000 == 0:
        print(i)
idf = np.log(tot / idf)
for j in range(F):
    data[:, j] = data[:, j] * idf[j]
import tensorflow as tf
H = 70
LR = 0.01
ph = tf.placeholder
tn = lambda x: tf.Variable(tf.truncated_normal(x, stddev = 0.1))
tc = lambda x, y: tf.Variable(tf.constant(x, shape=y))
mm = tf.matmul
sce = tf.nn.softmax_cross_entropy_with_logits
rm = tf.reduce_mean
relu = tf.nn.relu
inp = ph(tf.float32, shape = (None, F))
lab = ph(tf.float32, shape = (None, 2))
w1, w2 = tn((F, H)), tn((H, 2))
b1, b2 = tc(0.0, [H]), tc(0.0, [2])
def logits(x):
    return mm(relu(mm(x, w1) + b1), w2) + b2
pred = tf.nn.softmax(logits(inp))
loss = rm(sce(logits=logits(inp), labels=lab))
acc = rm(tf.to_float(tf.equal(tf.argmax(pred, 1), tf.argmax(lab, 1)))) * 100
opt = tf.train.RMSPropOptimizer(LR)
step = opt.minimize(loss)
slabel = []
sdata = []
Q = 20000
cnt1 = 0
cnt2 = 0
for i in range(tot):
    if label[i][0] == 0 and cnt1 < cnt2:
        cnt1 = cnt1 + 1
        sdata.append(data[i])
        slabel.append(label[i])
        if cnt1 == Q: break
    if label[i][0] == 1 and cnt2 <= cnt1:
        cnt2 = cnt2 + 1
        sdata.append(data[i])
        slabel.append(label[i])

slabel = np.array(slabel)
sdata = np.array(sdata)
print(sdata.shape)
V = 1000
T = 2 * Q - V
tlabel, vlabel = slabel[:-V], slabel[-V:]
tdata, vdata = sdata[:-V], sdata[-V:]
init = tf.global_variables_initializer()
ss = tf.Session()
ss.run(init)
B = 1000
S = 10000
for i in range(S):
    index = np.random.randint(0, T, B)
    fd = {inp: tdata[index], lab: tlabel[index]}
    ss.run(step, feed_dict=fd)
    if i % 1000 == 0:
        fd = {inp: vdata, lab: vlabel}
        valid_acc = ss.run(acc, feed_dict=fd)
        print('Step %i: acc %f' % (i, valid_acc))