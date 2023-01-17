# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# Import librairies
import math
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("../input/creditcard.csv")

# Any results you write to the current directory are saved as output.
df.head()
# EDA
df.describe()
df.info()
# data dictionary: positive class is fraud, negative is not; only 0 and 1
fraud_ind=df[df.Class==1].index
nofraud_ind=df[df.Class==0].index
fraud_num=len(fraud_ind)
nofraud_num=len(nofraud_ind)
fraud_perc=round(fraud_num/(fraud_num+nofraud_num),5)*100
print("% of fraud of all transactions is ", fraud_perc, "%")
# imbalanced dataset
# fraudulent transactions make up 0.173% portion of the data set
# Feature Selection
# Correlation
xx=df.drop(['Class','Time'],axis=1)
plt.figure(figsize=(20,20))
sns.heatmap(xx.corr(),annot=True,fmt= '.1f')
plt.title('Heatmap correlation')
plt.show()
# Most features have no or low correlations
# Selected features: V1-28, Amount
# Features: Amount, Time
f, (ax_fraud, ax_not) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
bins = 30

ax_fraud.hist(df.Amount[df.Class==1],bins=bins,normed=True,color='coral')
ax_fraud.set_title('Fraud Transactions over Amount')

ax_not.hist(df.Amount[df.Class==0],bins=bins,normed=True,color='seagreen')
ax_not.set_title('Non-Fraud Transactions over Amount')

plt.xlabel('Amount')
plt.ylabel('% of Transactions')
plt.yscale('log')
plt.show()
# fraudulent transactions have a higher average amount per transaction
bins=80
plt.figure(figsize=(8,6))
plt.hist(df.Time[df.Class==1],bins=bins,normed=True,alpha=0.9,label='Fraud',color='red')
plt.hist(df.Time[df.Class==0],bins=bins,normed=True,alpha=0.9,label='Not Fraud',color='lightblue')
plt.legend(loc='upper right')
plt.xlabel('Time in Sec')
plt.ylabel('% of Transactions')
plt.title('Transactions over Time')
plt.show()
# Inverse trends between fraudulent activity and nonfraudulent activity.
# Downturn in regular transactions, fraudulent activity increase. eg: 100,000 sec
# Check features V1-V28 at once
y=df.Class
x=df.drop(['Class','Time','Amount'],axis=1)

# choose V1-V10
sub_df_a=pd.concat([y,x.iloc[:,0:10]],axis=1)
sub_df_aa=pd.melt(sub_df_a,id_vars="Class",var_name="Feature",value_name='Value')
plt.figure(figsize=(20,8))
sns.violinplot(x="Feature",y="Value",hue="Class",data=sub_df_aa, split=True)
# need to scale x
# scale x
x_scaled=(x-x.min())/(x.max()-x.min())
# Easiness to see graph: make 3 sub dfs to better v
sub_df1=pd.concat([y,x_scaled.iloc[:,0:10]],axis=1)
sub_df2=pd.concat([y,x_scaled.iloc[:,10:19]],axis=1)
sub_df3=pd.concat([y,x_scaled.iloc[:,19:28]],axis=1)
sub_df11=pd.melt(sub_df1,id_vars="Class",var_name="Feature",value_name='Value')
sub_df22=pd.melt(sub_df2,id_vars="Class",var_name="Feature",value_name='Value')
sub_df33=pd.melt(sub_df3,id_vars="Class",var_name="Feature",value_name='Value')

plt.figure(figsize=(20,8))
sns.violinplot(x="Feature",y="Value",hue="Class",data=sub_df11, split=True)
plt.figure(figsize=(20,8))
sns.violinplot(x="Feature",y="Value",hue="Class",data=sub_df22, split=True)
plt.figure(figsize=(20,8))
sns.violinplot(x="Feature",y="Value",hue="Class",data=sub_df33, split=True)
# Most of the features have difference between frauds and non-frauds
# But eg: V20 and 22 are very symmetric
# Drop all of the features that have very similar distributions between the two types of transactions.
df_features=df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8','Time','Class'],axis=1)
# Normalize Amount
df_features["Amount"]=(df_features["Amount"]-df_features["Amount"].mean())/df_features["Amount"].std()
# Create train and test datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_features,y,test_size=0.3)
# Linear classifier in tensorflow
nV01 = tf.feature_column.numeric_column('V1')
nV02 = tf.feature_column.numeric_column('V2')
nV03 = tf.feature_column.numeric_column('V3')
nV04 = tf.feature_column.numeric_column('V4')
nV05 = tf.feature_column.numeric_column('V5')
nV06 = tf.feature_column.numeric_column('V6')
nV07 = tf.feature_column.numeric_column('V7')
nV09 = tf.feature_column.numeric_column('V9')
nV10 = tf.feature_column.numeric_column('V10')
nV11 = tf.feature_column.numeric_column('V11')
nV12 = tf.feature_column.numeric_column('V12')
nV14 = tf.feature_column.numeric_column('V14')
nV16 = tf.feature_column.numeric_column('V16')
nV17 = tf.feature_column.numeric_column('V17')
nV18 = tf.feature_column.numeric_column('V18')
nV19 = tf.feature_column.numeric_column('V19')
nV21 = tf.feature_column.numeric_column('V21')
nV22 = tf.feature_column.numeric_column('V22')
nV30 = tf.feature_column.numeric_column('Amount')

features=[nV01,nV02,nV03,nV04,nV05,nV06,nV07,nV09,nV10,nV11,nV12,nV14,nV16,nV17,nV18,nV19,nV21,nV30]
# Classification
# Tensorflow: Linear Classifier
input_func=tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=100,num_epochs=1000,shuffle=True) 
model=tf.estimator.LinearClassifier(feature_columns=features,n_classes=2)
model.train(input_fn=input_func,steps=1000)

result1=model.evaluate(tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10, num_epochs=1, shuffle=False))
# Result of linear classification
print(result1)
# Test Linear classification
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,batch_size=10, num_epochs=1, shuffle=False)
results2=model.evaluate(eval_input_func)
print(results2)
# Prediction:
from sklearn import metrics
pred_input_func= tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False)
predictions = model.predict(pred_input_func)

y_pred=[d['logits'] for d in predictions]
fpr,tpr,thresholds=metrics.roc_curve(y_test, y_pred)
roc_auc=metrics.auc(fpr, tpr)

plt.figure(figsize=(10,10))
plt.title('ROC-Tensorflow')
plt.plot(fpr, tpr,'b', label='Area under curve = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# AUC=0.97 good to use the model
# Neural network: use all the attributes
x_neural=df.drop(['Class'],axis=1)
x_scaled_neural=(x_neural-x_neural.min())/(x_neural.max()-x_neural.min())
y_neural=df.Class
from sklearn.model_selection import train_test_split
x_train_neural,x_test_neural,y_train_neural,y_test_neural=train_test_split(x_scaled_neural,y_neural,test_size=0.3)
# Feed data into the network
def to_one_hot(c, depth):
    i=np.identity(depth)
    return i[c,:]

def train_batch(batch_size):
    for j in range(int(len(x_train_neural)/batch_size)):
        start=batch_size*j
        end=start+batch_size
        
        train_x_batch=x_train_neural[start:end]
        train_y_batch=y_train_neural[start:end]
        
        train_y_batch=np.apply_along_axis(lambda x:to_one_hot(x,depth=2),0,train_y_batch)
        
        yield train_x_batch, train_y_batch
        
def get_test_data():
    return x_test_neural, np.apply_along_axis(lambda x: to_one_hot(x, depth=2), 0, y_test_neural)
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_recall_curve

X= tf.placeholder(tf.float32, [None, 30]) # inputs
Y= tf.placeholder(tf.float32, [None, 2]) # targets

def forward_propogation(X): # model
    num_neural= 10
    weights={"lvl_1": tf.Variable(tf.random_normal([30,num_neural])), 
               "output": tf.Variable(tf.random_normal([num_neural,2]))}
    
    biases={"lvl_1": tf.Variable(tf.random_normal([num_neural])), 
            "output": tf.Variable(tf.random_normal([2]))}
    
    h1=tf.add(tf.matmul(X, weights["lvl_1"]), biases["lvl_1"])
    h1=tf.nn.relu(h1)
    
    h2=tf.add(tf.matmul(h1, weights["output"]), biases["output"])
    output=tf.nn.relu(h2)
    
    return output

# Minimize loss
logits=forward_propogation(X)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer=tf.train.AdamOptimizer(0.01).minimize(loss) 

# Confusion matrix accuracy
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 

# Precision-recall curve
decision_variable = tf.nn.softmax(logits) 

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    
    # train phase
    batch_size = 128
    n_epochs = 3
    for epoch in range(n_epochs):
        epoch_loss = 0
        batch_generator = train_batch(batch_size)
        for batch in batch_generator:
            batch_x, batch_t = batch 
            _, curr_loss = sess.run([optimizer, loss], feed_dict={Y: batch_t, X: batch_x})
            epoch_loss += curr_loss
            
        print("Epoch " + str(epoch+1) + " loss: " + str(epoch_loss))
        
    # test phase
    test_x, test_t = get_test_data()
    test_y = sess.run(decision_variable, feed_dict={X: test_x})
    
    auprc = average_precision_score(test_t[:,0], test_y[:,0])
    precision, recall, _ = precision_recall_curve(test_t[:,0], test_y[:,0])

    # plot pr curve
    plt.plot(recall, precision)
    plt.grid()
    plt.show()
    
    # print average precision
    print(auprc)
