import numpy as np 

import pandas as pd

path='/kaggle/input/safety-parameters-in-indian-sites/After_Accident_Data.csv'

data=pd.read_csv(path,sep=',')

data.head()

import matplotlib.pyplot as plt

a=data.groupby(['Month', 'Injury_risk'])

month=a.size().unstack().fillna(0)



fig,az=plt.subplots(nrows=3,figsize=(12,5))

az [ 0  ].plot(month.index,month.High)

az[0].legend(loc="HIGH")



az[1].plot(month.index,month.Medium,color='red')

az[1].legend(loc="MEDIUM")



az[2].plot(month.index,month.Low,color='green')

az[2].legend(loc="LOW")

plt.show()
a=data.groupby(['Day', 'Injury_risk'])

day=a.size().unstack().fillna(0)

fig,az=plt.subplots(nrows=3,figsize=(12,5))

az [ 0  ].plot(day.index,day.High)

az[0].legend(loc="HIGH")



az[1].plot(day.index,day.Medium,color='red')

az[1].legend(loc="MEDIUM")



az[2].plot(day.index,day.Low,color='green')

az[2].legend(loc="LOW")



plt.show()
data_y=data['Injury_risk']

del data['Injury_risk']
from sklearn import preprocessing

import sklearn as sk



d=list(range(12))

data_1=data.iloc[0:,d].values

str_t=sk.preprocessing.LabelEncoder()

str_t.fit(data_1.reshape(22764))

data_1=str_t.transform(data_1.reshape(22764))

data_1=data_1.reshape(1897,12)
str_y=sk.preprocessing.LabelEncoder()

str_y.fit(data_y[1:])

data_y=str_y.transform(data_y)

np.unique(data_y)
from sklearn.ensemble import RandomForestClassifier

classifier_inform= RandomForestClassifier(criterion='entropy',n_estimators=40)

classifier_inform.fit(data_1,data_y)

importances=classifier_inform.feature_importances_
plt.bar(range(data_1.shape[1]),importances,color='blue',align='center')

plt.show()
data_inform=data_1[0:,[0,2,3,4,8,10,11]]
from sklearn.preprocessing import StandardScaler

stand=StandardScaler()

stand.fit(data_inform)

data_inform=stand.transform(data_inform)



data_inform.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data_inform,data_y,train_size=0.7,random_state=100)

y_train.shape

x_train.shape
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

x=tf.placeholder(tf.float32,(None,x_train.shape[1]),name='x')

y=tf.placeholder(tf.int32,(None),name='y')
hidden_layer_1=tf.layers.dense(x,32,activation=tf.nn.elu)



output_layer=tf.layers.dense(hidden_layer_1,3,activation=tf.nn.softmax)
entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output_layer))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(entropy)

correct=tf.nn.in_top_k(output_layer,y,1)

accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
init=tf.global_variables_initializer()

save=tf.train.Saver()

with tf.Session() as sess:

    init.run()

    for epoch in range(200):

        for i in range(len(x_train)):

            x1=x_train[i].reshape(1,7)

            y1=y_train[i].reshape(1)

            sess.run(optimizer,feed_dict={x:x1,y:y1})

    save_path=save.save(sess,'/kaggle/input/tensor_indi.ckpt')
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

with tf.Session() as sess:

    metrix_class=[]

    save.restore(sess,"/kaggle/input/tensor_indi.ckpt")

    for i in range(len(x_test)):

            x1=x_test[i].reshape(1,7)

            y1=y_test[i].reshape(1)

            metrix=tf.nn.softmax(output_layer).eval({x:x1})

            metrix_class.append(np.argmax(metrix))

    targets=['class-0','class-1','class-2']

   

    print('\n',classification_report(y_test,metrix_class,target_names=targets))

    print('accurcy',accuracy_score(metrix_class, y_test))
classifier= RandomForestClassifier(criterion='entropy',n_estimators=30,n_jobs=2)



classifier.fit(x_train,y_train)



pred=classifier.predict(x_test)

print('accurcy',accuracy_score(pred,y_test))