# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from imblearn.pipeline import make_pipeline

from imblearn.over_sampling import SMOTE

import tensorflow as tf



import sklearn.pipeline 

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dftrain=pd.read_csv("../input/train.csv")

dftest=pd.read_csv("../input/test.csv")

dftrain.head()
dftrain.dtypes
df_Insincere=dftrain[dftrain.target==0]

print("Total Samples : {}". format(dftrain.shape[0]))

print("No Of Insincere Samples : {}". format(df_Insincere.shape[0]))

print("No Of sincere Samples : {}". format(dftrain.shape[0]- df_Insincere.shape[0]))
dftrain.dropna(axis=1)

dftrain.drop_duplicates(inplace=True)
dftrain.shape
# Check data is Balance 

dfgroup=dftrain.groupby(['target']).agg(['count'])

dfgroup.columns=['COUNT_PER_CLASS', 'COUNT_PER_CLASS_TEXT']

dfgroup['COUNT_PER_CLASS_%']=dfgroup['COUNT_PER_CLASS'].map(lambda x: (x/dftrain.shape[0])*100)

dfgroup
target_visual={1:'YES',0:'NO'}

dftrain_visual=dftrain

dftrain_visual['target']=dftrain_visual['target'].map( lambda x : 'YES' if x>0 else 'NO') 
font={'size':16}

fig, ax=plt.subplots(figsize=(10,5))

# Sample Per class

df_sample_count=dftrain_visual['target'].groupby(dftrain_visual['target']).count()

x=df_sample_count.index.values

ax.bar(x,df_sample_count,align='center', label=['On-Time', 'Delayed Flight']) 

ax.set_ylabel('Number of Samples')

ax.set_xlabel('Types of Class')

ax.set_xticks(x)

ax.set_xticklabels(x, rotation = 45) 

plt.show()
countV=CountVectorizer(stop_words='english')

tfIdf=TfidfTransformer() 
X=dftrain.question_text

X=X.str.lower().str.strip()

Y= dftrain.target

Y=pd.get_dummies(Y)
X=countV.fit_transform(X)

X=tfIdf.fit_transform(X)
x_train,x_test, y_train,y_test=train_test_split(X,Y, test_size=0.25,stratify=Y)
def nn_layers(df,weights,biases, keep_prob):

    l1=tf.add(tf.matmul(df,weights['h1']), biases['b1'])

    l1=tf.nn.relu(l1)

    l1=tf.nn.dropout(l1,keep_prob)

    l_out=tf.add(tf.matmul(l1,weights['out']), biases['out'])

    return l_out
n_hidden_1=3000

n_input=x_train.shape[1]

n_classes=y_train.shape[1]
# Weight and Biases for every layer

weights={

    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),

    'out':tf.Variable(tf.random_normal([n_hidden_1, n_classes]))

}

biases={

    'b1':tf.Variable(tf.random_normal([ n_hidden_1])),

    'out':tf.Variable(tf.random_normal([ n_classes]))

}

keep_prob = tf.placeholder(tf.float32)

training_epochs = 5

display_step = 1000

batch_size = 100000

x=tf.placeholder(tf.float32, [None,n_input])

y=tf.placeholder(tf.float32, [None,n_classes])
predictions=nn_layers(df=x,weights=weights,biases=biases,keep_prob=keep_prob)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))

lr_rate=0.001

optimizer=tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(cost)
with tf.Session() as sess:

    initializer = tf.global_variables_initializer()

    sess.run(initializer)

    for epoch in range(training_epochs):

        avg_cost = 0.0

        total_batches = int(len(x_train) / batch_size)

        x_batches = np.array_split(x_train, total_batches)

        y_batches = np.array_split(y_train_NN, total_batches)

        for i in range(total_batches):

            batch_x, batch_y = x_batches[i], y_batches[i]

            print(batch_x.shape)

            print(batch_y.shape)

            _,co=sess.run([optimizer, cost], feed_dict={x:x_batch, y:y_batch, keep_prob:0.50})

            avg_cost += c / total_batches

        if epoch % display_step:

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print('Execution Finished')

    correct_prediction=tf.equal(tf.argmax(predictions,1), tf.argmaxa(y_train,1))

    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    print("Test Accuracy {}".format(accuracy.eval({x:x_test,y:y_test,keep_prob:1.0})))

    print("Train Accuracy {}".format(accuracy.eval({x:x_train,y:y_train,keep_prob :1.0})))