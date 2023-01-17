# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
print(os.listdir("../input"))


#tf.reset_default_graph()

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

def scale(data,scale_range = [1,0.1]):
    max_range,min_range = scale_range
    max_data = max(data)
    min_data = min(data)
    data = np.array(list(data))
    scale_data = min_range + (data - min_data)*\
                (max_range - min_range)/(max_data - min_data)
    return [scale_data,max_data,min_data]            


def rescale(scale_data,max_min,scale_range = [1,0.1]):
    max_range,min_range = scale_range
    max_data,min_data = max_min
    data = min_data + (scale_data -min_range)*\
    (max_data - min_data)/(max_range - min_range)
    return data
    


# Any results you write to the current directory are saved as output.
class dnn_regressor:
    
    def __init__(self,hidden_units,
                 learning_rate = 0.01,
                 training_epochs = 1000,
                 beta = 0.1,
                 batch_size = 1000,
                 name = 'model_saver',
                 activation = [tf.nn.tanh]):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs 
        self.beta = beta 
        self.batch_size = batch_size
        self.name = name
        self.activation = activation
        
    # Network Parameters        
    def MLP(self,hidden_units,x):
        activation = self.activation
        n_features = x.shape[1].value
        weights = []
        biases = []
        temp_u = n_features
        for i in range(len(hidden_units)):
            weights.append(tf.Variable(\
                           tf.random_normal([temp_u, hidden_units[i]], 0, 0.1)))
            biases.append(tf.Variable(\
                        tf.random_normal([hidden_units[i]], 0, 0.1)))
            temp_u = hidden_units[i]
        weights.append(tf.Variable(\
                       tf.random_normal([hidden_units[-1],1], 0, 0.1)))
        biases.append(tf.Variable(\
                    tf.random_normal([1], 0, 0.1)))    
            
        layers = [x]   
        if len(activation) == 1:
            activation_f = activation[0]
        
        for i in range(len(hidden_units)):
            layer = tf.add(tf.matmul(layers[-1], weights[i]), biases[i])
            if  len(activation) == len(hidden_units):
                activation_f = activation[i]
            layer = activation_f(layer)
            layers.append(layer)
        out_layer = tf.matmul(layers[-1], weights[-1]) + biases[-1]
        
        regularizer = 0
        for w,b in zip(weights,biases):
            regularizer = regularizer + tf.nn.l2_loss(w) 
            #no bias here works better,why?
    
        return [out_layer,regularizer]

    def train(self,x_train,y_train,evaluate = False,
              x_test = [],y_test = [],
              optimizer = tf.train.AdamOptimizer,
              display_step = 100):
        n_features = x_train.shape[1]
        n_samples = x_train.shape[0]
        x = tf.placeholder("float", [None, n_features])
        y = tf.placeholder("float", [None,1])
        pred,regularizer = self.MLP(self.hidden_units,x)
        self.pred = pred
        self.x = x
        
        cost = tf.reduce_mean(tf.square(pred-y)) 
        cost = cost + self.beta * regularizer
        optimizer = optimizer(self.learning_rate).minimize(cost)

        
        sess =  tf.Session()
        sess.run(tf.initialize_all_variables())
        # Training cycle
        self.cost_curve_train = []
        self.cost_curve_test = []
        for epoch in range(self.training_epochs):
            total_batch = int(n_samples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x_train[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = y_train[i*self.batch_size:(i+1)*self.batch_size]
                _, c, p = sess.run([optimizer, cost, pred], 
                                   feed_dict={x: batch_x, y: batch_y})

            train_pred = sess.run(pred, feed_dict={x: x_train})
            train_cost = np.mean(np.square(train_pred - y_train))      
            self.cost_curve_train.append([epoch,train_cost])
            
            if evaluate:
                test_pred = sess.run(pred, feed_dict={x: x_test})
                test_cost = np.mean(np.square(test_pred - y_test))      
                self.cost_curve_test.append([epoch,test_cost])                
            self.sess = sess
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "train cost=", \
                    "{:.9f}".format(train_cost))
                if evaluate:
                    print ("Epoch:", '%04d' % (epoch+1), "evaluate cost=", \
                           "{:.9f}".format(test_cost))
                print ("[*]----------------------------")       
        #sess.close()
        print ("Optimization Finished!")
    def predict(self,x_predict):
        sess = self.sess
        test_pred = sess.run(self.pred, feed_dict={self.x: x_predict}) 
        return test_pred
    def close(self):
        self.sess.close()
        print("session is closed")
        
    



for c in list(df_train.columns)[1:-1]:
    column_train = df_train[c]
    column_test = df_test[c]
    len_c1 = column_train.shape[0]
    len_c2 = column_test.shape[0]
    column = pd.concat((column_train,column_test))
    if column.dtype.kind == 'f' or column.dtype == 'i':
        column = column.fillna(0) 
        s_column,max_column,min_column = scale(column)
        df_train[c] = s_column[:len_c1]
        df_test[c] = s_column[-len_c2:]
    else:
        if column.isnull().values.any():#
            column_na = column.fillna('None')
            column_c = column.astype('category')
#            print('$$$$$')
#            print(column_c.cat.categories)  
            cat_l = column_c.cat.categories.tolist()
            new_cat_l = [0.8*t/(len(cat_l) - 1) + 0.1 \
                         for t in list(range(len(cat_l)))]
            dic = dict(zip(cat_l,new_cat_l))
            dic['None'] = 0
            new_column = column_na.map(dic)
            df_train[c] = new_column[:len_c1]
            df_test[c] = new_column[-len_c2:]           
            
        else:
            column_c = column.astype('category')
#            print(column_c.cat.categories)     
            cat_l = column_c.cat.categories.tolist()
            new_cat_l = [0.8*t/(len(cat_l) - 1) + 0.1 \
                         for t in list(range(len(cat_l)))]
            dic = dict(zip(cat_l,new_cat_l))
            new_column = column_c.map(dic)
            df_train[c] = new_column[:len_c1]
            df_test[c] = new_column[-len_c2:]           

X = df_train.iloc[:,1:-1].values
y = df_train['SalePrice'].values.astype('float')
s_y,max_y,min_y = scale(y)

y = s_y.reshape(len(y),1)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)                        



regressor = dnn_regressor
r_ = regressor([40,15,10,5,5,5], batch_size = X_train.shape[0],
                   training_epochs = 10000,beta = 0.00005)
r_.train(X_train,y_train,evaluate= True,x_test = X_test,y_test = y_test)

y_pred = rescale(r_.predict(X_test),[max_y,min_y])
y_test = rescale(y_test,[max_y,min_y])
num = len(y_test)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
p1, = ax.plot(range(num),y_test)
p2, = ax.plot(range(num),y_pred)
plt.show()

df_test_id = df_test['Id']
df_test_value = df_test.iloc[:,1:]
y_pred = r_.predict(df_test_value)
df_test_y = rescale(y_pred,[max_y,min_y])
r_.close()

my_submission = pd.DataFrame({'Id': df_test_id.values, 'SalePrice': df_test_y.ravel()})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


