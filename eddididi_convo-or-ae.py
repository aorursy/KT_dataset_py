# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the"../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw=pd.read_csv('/kaggle/input/sudoku/sudoku.csv')
def preprocessSDK(raw):

    qlist=raw['quizzes'].tolist()

    alist=raw['solutions'].tolist()

    qArray=[]

    aArray=[]

    temp=[]

    row=[]

    cnter=0

    switcher2=0

    for q in qlist[:4096]:

        switcher=0

        for i in str(q):

            row.append(float(i))

            cnter+=1

            if cnter==9: 

                cnter=0

                

                if switcher==0:

                    temp=np.array(row)

                    switcher=1

                else:

                    temp=np.vstack((temp,np.array(row)))

                    

                row=[]

        qArray.append(temp)

        

    cnter=0

    switcher=0

    

    for a in alist[:4096]:

        temp=[]

        switcher=0

        for i in str(a):

            row.append(float(i))

            cnter+=1

            if cnter==9: 

                cnter=0

                if switcher==0:

                    temp=np.array(row)

                    switcher=1

                else:

                    temp=np.vstack((temp,np.array(row)))

                

                row=[]

        aArray.append(temp)

    return np.array(qArray),np.array(aArray)
qList,aList=preprocessSDK(raw)
question=tf.placeholder(tf.float32,shape=(None,9,9))

answer=tf.placeholder(tf.float32,shape=(None,9,9))



q_input=tf.reshape(question,shape=[tf.shape(question)[0], -1])

a_output=tf.reshape(answer,shape=[tf.shape(answer)[0], -1])



W1=tf.Variable(tf.random_normal([81, 32], stddev=0.1))

B1=tf.Variable(tf.zeros(32))



W2=tf.Variable(tf.random_normal([32, 16], stddev=0.1))

B2=tf.Variable(tf.zeros(16))



W3=tf.Variable(tf.random_normal([16, 8], stddev=0.1))

B3=tf.Variable(tf.zeros(8))



enc1=tf.matmul(q_input,W1)+B1

enc2=tf.matmul(enc1,W2)+B2

enc3=tf.matmul(enc2,W3)+B3





W3I=tf.Variable(tf.random_normal([8, 16], stddev=0.1))

B3I=tf.Variable(tf.zeros(16))



W2I=tf.Variable(tf.random_normal([16, 32], stddev=0.1))

B2I=tf.Variable(tf.zeros(32))



W1I=tf.Variable(tf.random_normal([32, 81], stddev=0.1))

B1I=tf.Variable(tf.zeros(81))



dec1=tf.matmul(enc3,W3I)+B3I

dec2=tf.matmul(dec1,W2I)+B2I

dec3=tf.matmul(dec2,W1I)+B1I





loss=tf.losses.mean_squared_error(a_output,dec3)

train=tf.train.RMSPropOptimizer(0.01).minimize(loss)





sess=tf.Session()

init=tf.global_variables_initializer()

sess.run(init)

batchSize=20

for epoches in range(1):

    for batchIdx in range(int(qList.shape[0]/batchSize)):

        questionBatch=qList[batchIdx*batchSize:batchIdx*batchSize+batchSize]

        answerBatch=aList[batchIdx*batchSize:batchIdx*batchSize+batchSize]

        losses,_=sess.run([loss,train],feed_dict={question:questionBatch, answer:answerBatch})

        print('Current batch loss: ',losses)