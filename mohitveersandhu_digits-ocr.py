import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import  PCA

import tensorflow as tf



from sklearn.preprocessing import StandardScaler

from sklearn.utils import  shuffle

%matplotlib inline
df=pd.read_csv('../input/train.csv')

df.info()
df.isnull().sum().head()
print(df['label'].value_counts())
plt.figure(figsize=(15,7))

sns.countplot(data=df,x='label',palette='Paired')

plt.xlabel('Digits')

plt.title('Frequency of Digits')
plt.figure(figsize=(15,5))

sns.kdeplot(df.iloc[:,0])
def show_digits(x,nrows,ncols):

    fig,axis=plt.subplots(nrows,ncols)

    fig.set_figheight(18)

    fig.set_figwidth(15)

    for i in range(nrows):

        for j in range(ncols):

            count=(np.random.rand(1)*100).astype('int')

            sns.heatmap(x.iloc[count,1:].values.reshape(28,28),ax=axis[i][j],xticklabels=False,yticklabels=False,cbar=False)

            axis[i,j].title.set_text(x.iloc[count,0].values[0])

            count+=1  

print('random images of given pixes in heatmap ')

show_digits(df,5,3)      

x=df.iloc[:,1:].values

y=df.iloc[:,0].values
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x=sc.fit_transform(x)
def comm_sum(x):

    total=0

    arr=np.array([])

    for i in x:

        total+=i

        arr=np.append(arr,total)

    return arr

def com_var(x,max_comp):

    pc=PCA(n_components=max_comp).fit(x)

    var=pc.explained_variance_ratio_

    var=comm_sum(var)

    return var

var=com_var(x[:42000,:],500)
fig=plt.figure(figsize=(12,7))

plt.plot(var*100,lw=5,c='red')

plt.xlabel('number of components')

plt.ylabel('comm. variance percentage')

plt.xlim(xmin=1)

plt.title('comulative variance ')
xtrain=x[:42000,:]

ytrain=y[:42000]




ytrain=ytrain.astype('int')

t_train=pd.get_dummies(ytrain).values



sc=StandardScaler()

xtrain=sc.fit_transform(xtrain)





def forward_test(X,w1,b1,w2,b2,w3,b3,w4,b4): 

    layer1=tf.add(tf.matmul(X,w1),b1)

    z1=tf.nn.relu(layer1)

    layer2=tf.add(tf.matmul(z1,w2),b2)

    z2=tf.nn.relu(layer2)

    layer3=tf.add(tf.matmul(z2,w3),b3)

    z3=tf.nn.relu(layer3)

    layer4=tf.add(tf.matmul(z3,w4),b4)

    

    return layer4



def forward(X,w1,b1,w2,b2,w3,b3,w4,b4):

    

    layer1=tf.add(tf.matmul(X,w1),b1)

    z1=tf.nn.relu(layer1)

    z1=tf.nn.dropout(x=z1,keep_prob=0.5)

    layer2=tf.add(tf.matmul(z1,w2),b2)

    z2=tf.nn.relu(layer2)

    z2=tf.nn.dropout(x=z2,keep_prob=0.5)

    layer3=tf.add(tf.matmul(z2,w3),b3)

    z3=tf.nn.relu(layer3)

    z3=tf.nn.dropout(x=z3,keep_prob=0.5)

    layer4=tf.add(tf.matmul(z3,w4),b4)

    

    return layer4







g1=tf.Graph()

with g1.as_default():

    m1=1500

    m2=1300

    m3=1100

    d=xtrain.shape[1]

    k=t_train.shape[1]

    

    w1=tf.Variable(tf.random_normal([d,m1])*tf.sqrt(2/d))

    w2=tf.Variable(tf.random_normal([m1,m2])*tf.sqrt(2/m1))

    w3=tf.Variable(tf.random_normal([m2,m3])*tf.sqrt(2/m2))

    w4=tf.Variable(tf.random_normal([m3,k]))





    b1=tf.Variable(tf.random_normal([m1]))

    b2=tf.Variable(tf.random_normal([m2]))

    b3=tf.Variable(tf.random_normal([m3]))



    b4=tf.Variable(tf.random_normal([k]))



    

    tfx=tf.placeholder(tf.float32,[None,d])

    tfy=tf.placeholder(tf.float32,[None,k])

    

    logits=forward(tfx,w1,b1,w2,b2,w3,b3,w4,b4)

    t_logits=forward_test(tfx,w1,b1,w2,b2,w3,b3,w4,b4)

    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfy,logits=logits))

    opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    predict=tf.argmax(t_logits,axis=1)

       

    

def model():

    ypred=[]

    with  tf.Session(graph=g1) as sess:

        init=tf.global_variables_initializer()

        sess.run(init)

        epoch=1

        batch_size=512

        n_batches=xtrain.shape[0]//batch_size

        x,y=xtrain,t_train

        x,y=shuffle(x,y)

        for i in range(epoch):

            for j in range(n_batches):

                xt=x[j*batch_size:(j*batch_size+batch_size)]

                yt=y[j*batch_size:(j*batch_size+batch_size)]

                sess.run(opt,feed_dict={tfx:xt,tfy:yt})

            print('accuracy-'+str(np.mean(ytrain==sess.run(predict,feed_dict={tfx:xtrain}))))

        sess.close()

    return ypred



        

ytest=model()