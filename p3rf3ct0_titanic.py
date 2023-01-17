# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_gender_submission = pd.read_csv('../input/gender_submission.csv')
categorizador = {'female':0, 'male':1,'C':0, 'Q':1, 'S':2}
data_train = data_train.fillna(data_train.mean()).fillna('None').replace(categorizador)
data_test = data_test.fillna(data_train.mean()).fillna('None').replace(categorizador)
data_gender_submission = data_gender_submission.fillna(data_train.mean()).fillna('None')
data_train.corr()
values = ['Pclass','Sex','Fare', 'Age']
clf = LinearDiscriminantAnalysis()
clf.fit(data_train[values],data_train.Survived)
prediccion_discriminat_analis=pd.DataFrame(data={'Survived':clf.predict(data_test[values]), 'PassengerId':data_test['PassengerId'].tolist()})
prediccion_discriminat_analis.to_csv('submission_lineal.csv',header=True, index=False)
(prediccion_discriminat_analis.Survived == data_gender_submission.Survived).sum() - 418

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
classifier=DecisionTreeClassifier()
classifier = classifier.fit(data_train[values],data_train.Survived)
prediccion_arbol_decision=pd.DataFrame(data={'Survived':classifier.predict(data_test[values]), 'PassengerId':data_test['PassengerId'].tolist()})
prediccion_arbol_decision.to_csv('submission_arbol.csv',header=True, index=False)
(prediccion_arbol_decision.Survived == data_gender_submission.Survived).sum() - 418
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier = classifier.fit(data_train[values],data_train.Survived)
predictions = classifier.predict(data_test[values])
sklearn.metrics.confusion_matrix(data_gender_submission.Survived,predictions)
sklearn.metrics.accuracy_score(data_gender_submission.Survived, predictions)
prediccion_random_forest=pd.DataFrame(data={'Survived':predictions, 'PassengerId':data_test['PassengerId'].tolist()})
prediccion_random_forest.to_csv('submission_random_forest.csv',header=True, index=False)
model = ExtraTreesClassifier()
model.fit(data_test[values],data_gender_submission.Survived)
print(model.feature_importances_)
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = data_train
test_data = data_test
test_passenger_id=test_data["PassengerId"]
def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

not_concerned_columns = ["PassengerId","Name", "Fare", "Ticket", "Cabin", "Embarked"]
train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass"]
train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)
from sklearn.preprocessing import MinMaxScaler
def normalize_column(data, column):
    scaler = MinMaxScaler()
    data[column] = scaler.fit_transform(data[column].values.reshape(-1,1))
    return data
train_data = normalize_column(train_data,'Age')
#train_data = normalize_column(train_data,'Fare')
test_data = normalize_column(test_data,'Age')
#test_data = normalize_column(test_data,'Fare')
train_data.head()
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_valid_test_data(data, fraction=(1 - 0.95)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

print("valid_x:{}".format(valid_x.shape))
print("valid_y:{}".format(valid_y.shape))
# Build Neural Network
from collections import namedtuple

def build_neural_network(hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

model = build_neural_network()
def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y
epochs = 2000
train_collect = 50
train_print=train_collect*2

learning_rate_value = 0.00001
batch_size=16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration=0
    for e in range(epochs):
        for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
            iteration+=1
            feed = {model.inputs: train_x,
                    model.labels: train_y,
                    model.learning_rate: learning_rate_value,
                    model.is_training:True
                   }

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
            
            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print==0:
                     print("Epoch: {}/{}".format(e + 1, epochs),
                      "Train Loss: {:.4f}".format(train_loss),
                      "Train Acc: {:.4f}".format(train_acc))
                        
                feed = {model.inputs: valid_x,
                        model.labels: valid_y,
                        model.is_training:False
                       }
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)
                
                if iteration % train_print==0:
                    print("Epoch: {}/{}".format(e + 1, epochs),
                      "Validation Loss: {:.4f}".format(val_loss),
                      "Validation Acc: {:.4f}".format(val_acc))
                

    saver.save(sess, "./titanic.ckpt")
plt.plot(x_collect, train_loss_collect, "r--")
plt.plot(x_collect, valid_loss_collect, "g^")
plt.show()
plt.plot(x_collect, train_acc_collect, "r--")
plt.plot(x_collect, valid_acc_collect, "g^")
plt.show()
model=build_neural_network()
restorer=tf.train.Saver()
with tf.Session() as sess:
    restorer.restore(sess,"./titanic.ckpt")
    feed={
        model.inputs:test_data,
        model.is_training:False
    }
    test_predict=sess.run(model.predicted,feed_dict=feed)
    
test_predict[:10]
from sklearn.preprocessing import Binarizer
binarizer=Binarizer(0.5)
test_predict_result=binarizer.fit_transform(test_predict)
test_predict_result=test_predict_result.astype(np.int32)
test_predict_result[:10]
passenger_id=test_passenger_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=test_predict_result
evaluation[:10]
evaluation.to_csv("evaluation_submission.csv",index=False)
data_1 = prediccion_discriminat_analis
data_2 = prediccion_arbol_decision
data_3 = prediccion_random_forest
data_4 = evaluation
data = (data_1+data_2+data_3+data_4)/4
data.head()

data['Survived'] = np.where(data['Survived']>=0.5, 1, 0)
data['PassengerId'] = data['PassengerId'].astype('int')
data.to_csv('submission_tonto.csv',header=True, index=False)
