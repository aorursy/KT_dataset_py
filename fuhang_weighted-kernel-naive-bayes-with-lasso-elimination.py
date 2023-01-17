#Kernel NB reference: https://www.kaggle.com/jiazhuang/demonstrate-naive-bayes



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns



#fit arbitary pdf by scipy.stats.kde.gaussian_kde

from scipy.stats.kde import gaussian_kde



from sklearn.metrics import roc_auc_score as AUC

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)

target = train.target.values

train.drop('target', axis=1, inplace=True)

train.shape, target.shape, test.shape
pos_idx=(target==1)

neg_idx=(target==0)

prior_pos=pos_idx.sum()/len(target)

prior_neg=1-prior_pos



pos_kdes,neg_kdes=[],[]#kde函数的列表

for col in train.columns:

    pos_kde=gaussian_kde(train.loc[pos_idx,col])

    neg_kde=gaussian_kde(train.loc[neg_idx,col])

    pos_kdes.append(pos_kde)

    neg_kdes.append(neg_kde)



def cal_prob_KDE_col_i(df,i,num_of_bins=100):

    bins=pd.cut(df.iloc[:,i],bins=num_of_bins)

    uniq=bins.unique()

    uniq_mid=uniq.map(lambda x:(x.left+x.right)/2)

    #把每一格uniq_mid映射到kde值

    mapping=pd.DataFrame({

        'pos':pos_kdes[i](uniq_mid),

        'neg':neg_kdes[i](uniq_mid)

    },index=uniq)

    return bins.map(mapping['pos'])/bins.map(mapping['neg'])



ls=[[prior_pos/prior_neg]*len(train)]

for i in range(200):

    ls.append(cal_prob_KDE_col_i(train,i))

train_KernelNB=pd.DataFrame(np.array(ls).T,columns=['prior']+['var_'+str(i) for i in range(200)])

pred=train_KernelNB.apply(lambda x:np.prod(x),axis=1)

AUC(target,pred)
log_train_KernelNB=train_KernelNB.apply(np.log,axis=1)#the log transform  is input for our nerual net.
# it is same as temp=train_KernelNB.apply(lambda x:np.prod(x),axis=1)

temp=log_train_KernelNB.apply(lambda x:np.exp(np.sum(x)),axis=1)

AUC(target,temp)
#NB:score=exp( log(p(y=1)/p(y=0))+∑log(p(xi|y=1)/p(xi|y=0)) )

#weighted NB:score=exp( w0*log(p(y=1)/p(y=0))+ wi*∑log(p(xi|y=1)/p(xi|y=0)) ) 

import tensorflow as tf



n=201 #201=1+200: 1 for prior prob's logit, 200 for likelyhood's logit 

y=tf.placeholder(tf.float32,[None,1])

x=tf.placeholder(tf.float32,[None,n])

w=tf.Variable(tf.ones([n]))#here, we initlize weighted NB making it start as a normal NB.

w=tf.nn.relu(w)#ReLU applied on w the make sure weights are positive or sparse

tf.multiply(w,x).shape



linear_term=tf.reduce_sum(tf.multiply(w,x),axis=1,keepdims=True)#(None,1)

linear_term=tf.math.exp(linear_term)#do the exp transform to reverse log



#define lambda coef for L1-norm, a key parameter to tune.

lambda_w=tf.constant(2*1e-5,name='lambda_w')

l1_norm=tf.multiply(lambda_w,tf.reduce_sum(tf.abs(w),keepdims=True))



error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=linear_term, labels=y))

loss = error+l1_norm

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(log_train_KernelNB.values, target, test_size=0.2, random_state=42)

print(X_test[:5,:])

def batcher(X_, y_=None, batch_size=-1):

    n_samples = X_.shape[0]

    if batch_size == -1:

        batch_size = n_samples

    if batch_size < 1:

       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):

        upper_bound = min(i + batch_size, n_samples)

        ret_x = X_[i:upper_bound]

        ret_y = None

        if y_ is not None:

            ret_y = y_[i:i + batch_size]

            yield (ret_x, ret_y)



optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

N_EPOCHS=100

batch_size=5000

init=tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(N_EPOCHS):

        if epoch==0:

            for bX, bY in batcher(X_test, y_test):#all sample

                prediction_list=sess.run(linear_term, feed_dict={x: bX.reshape(-1, n)})

                print('Init w:',sess.run(w, feed_dict={x: bX.reshape(-1, n)}))

                print('Init pred:',prediction_list)

                print('Init Test AUC:',AUC(y_test,prediction_list))

               

        perm = np.random.permutation(X_train.shape[0])

        # iterate over batches

        for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):

            sess.run(optimizer, feed_dict={x: bX.reshape(-1, n), y: bY.reshape(-1, 1)})

            

        if (epoch+1)%20==0:

            print("Epoch:",epoch)

            for bX, bY in batcher(X_test, y_test):#all sample

                print('Test CrossEntropy (logloss):',sess.run(error, feed_dict={x: bX.reshape(-1, n), y: bY.reshape(-1, 1)}))

                prediction_list=sess.run(linear_term, feed_dict={x: bX.reshape(-1, n)})

                weights=sess.run(w, feed_dict={x: bX.reshape(-1, n)})

                print('w:',weights)

                print('Test pred:',prediction_list)

                print('Test AUC:',AUC(y_test,prediction_list))

            print('=======')
sparse_weights=np.where(np.abs(weights)<1e-2,0,weights)

print ('Number of eliminated feature',(sparse_weights<=0).sum())
ls=[[prior_pos/prior_neg]*len(test)]

for i in range(200):

    ls.append(cal_prob_KDE_col_i(test,i))

test_KernelNB=pd.DataFrame(np.array(ls).T,columns=['prior']+['var_'+str(i) for i in range(200)])
pred_test=test_KernelNB.loc[:,sparse_weights>0].apply(lambda x:np.prod(x),axis=1)

pd.DataFrame({

    'ID_code': test.index,

    'target': pred_test

}).to_csv('sub01_KernelNB_L1_2E-5_VAL_0.904.csv', index=False)
sparse_weights=pd.Series(sparse_weights,index=['prior']+['var_'+str(i) for i in range(200)])

sns.distplot(sparse_weights,bins=50,kde=False)
print (sparse_weights.sort_values(ascending=False))
sparse_weights.plot()
sparse_weights[['var_'+str(i) for i in [108,184,9,80,76,13,166,94,170,154,133,169,174,123,6] ]]