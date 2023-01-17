# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from glob import glob
import sklearn
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/A_DeviceMotion_data/A_DeviceMotion_data"))

# Any results you write to the current directory are saved as output.
# Activety types dict:
Activety_Types = {'dws':1,'jog':2,'sit':3,'std':4,'ups':5,'wlk':6}        
listDict = list(Activety_Types.keys())

# Data Folders:
Folders = glob('../input/A_DeviceMotion_data/A_DeviceMotion_data/*_*')
Folders = [s for s in Folders if "csv" not in s]

Df_all_list = []
Exp = 0
# Segment the data to 400 sampels frames , each one will be a different Expirament
Segment_Size = 400

# Load All data:
for j  in Folders:
    Csv = glob(j + '/*' )


    for i in Csv:
        df = pd.read_csv(i)
        # Add Activety label, Subject name and Experiment number
        df['Activity'] = Activety_Types[j[49:52]]
        df['Sub_Num'] = i[len(j)+5:-4]
        df['Exp_num'] = 1
        ExpNum = np.zeros((df.shape[0])) 
        for i in range(0,df.shape[0]-Segment_Size,Segment_Size):
            ExpNum[range(i,i+Segment_Size)] = i/Segment_Size +Exp*100 
        df['Exp_num'] = ExpNum
        #Df_all = pd.concat([Df_all,df])
        Df_all_list.append(df)
        Exp += 1        

Df_all = pd.concat(Df_all_list,axis=0)  
# let's see the data
Df_all.head()
plt.plot([1,2,3])
# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
#plt.subplot(2,1,1)
#plt.plot(range(12))
for i in range(6):
    D = Df_all[Df_all['Activity']==i+1]
    plt.subplot(3,2,i+1)
    plt.plot(D['userAcceleration.z'][:200])
    plt.title(listDict[i])
    plt.ylim([-1, 1])
plt.tight_layout()

#  Calculate features
df_sum = Df_all.groupby('Exp_num', axis=0).mean().reset_index()
df_sum.columns = df_sum.columns.str.replace('.','_sum_')

df_sum_SS = np.power(Df_all.astype(float),2).groupby('Exp_num', axis=0).median().reset_index() 
df_sum_SS.columns = df_sum_SS.columns.str.replace('.','_sumSS_')

df_max = Df_all.groupby('Exp_num', axis=0).max().reset_index()
df_max.columns = df_max.columns.str.replace('.','_max_')

df_min = Df_all.groupby('Exp_num', axis=0).min().reset_index()
df_min.columns = df_min.columns.str.replace('.','_min_')

df_skew = Df_all.groupby('Exp_num', axis=0).skew().reset_index()
df_skew.columns = df_skew.columns.str.replace('.','_skew_')

df_std = Df_all.groupby('Exp_num', axis=0).std().reset_index()
df_std.columns = df_std.columns.str.replace('.','_std_')

# Concat features and labels vector into one Data Frame:
Df_Features = pd.concat([ df_max , df_sum[df_sum.columns[2:-2]], 
                         df_min[df_min.columns[2:-2]], df_sum_SS[df_sum_SS.columns[2:-2]], 
                         df_std[df_std.columns[2:-2]], df_skew[df_skew.columns[2:-2]]], axis=1)
# Features
Df_Features_1 = Df_Features.drop(['Exp_num','Unnamed: 0','Activity','Sub_Num'],axis=1)
Labels = Df_Features['Activity']
# Train test split (this can be done also by user, to makeit a more realistic case)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Df_Features_1, Labels, test_size=0.25, random_state=42)

# Define placeholders for X and Y
X_shape = Df_Features_1.shape[1]
Cluss_Num = len(Activety_Types)
X = tf.placeholder(dtype=tf.float32,shape=[None,X_shape])
y = tf.placeholder(dtype=tf.float32,shape = [None,Cluss_Num])
Hold_prob = tf.placeholder(tf.float32)
# Helper functions for DNN design
def Init_Wightes(shape1):
    W = tf.truncated_normal(shape1,stddev=0.1)
    return tf.Variable(W)

def Init_bias(shape1):
    b =  tf.constant(0.1,shape=shape1)
    return tf.Variable(b)

def FC_layer(input1,shape1):
    
    W = Init_Wightes(shape1)
    B = Init_bias([shape1[1]])
    Wx = tf.matmul(input1,W)
    Wx_b = tf.add(Wx,B)
    return tf.nn.relu(Wx_b)

    
# Define model parmeters and net design
H1_size = 54
H2_size = 24

H1 = FC_layer(X, [X_shape,H1_size])
#H2 = FC_layer(H1,[H1_size,H2_size])
H1_drop = tf.nn.dropout(H1,keep_prob=Hold_prob)
y_pred = FC_layer(H1_drop,[H1_size,len(Activety_Types)])
# Set TensorFlow Error and train function
Err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_pred))
Optimaizer = tf.train.AdamOptimizer()
Train = Optimaizer.minimize(Err)
Init = tf.global_variables_initializer()



BatchSize = 64
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(Cluss_Num))

# convarte to numpy and devide into batches:

X_train_Np = np.array(X_train,dtype=np.float32)
Y_train_Np = np.array(y_train,dtype=np.float32)
X_test_Np = np.array(X_test,dtype=np.float32)
Y_test_Np = np.array(y_test,dtype=np.float32)
Y_test_OH  = label_binarizer.transform(Y_test_Np)
Y_test_OH = np.array(Y_test_OH,dtype=np.float32)
Batches = np.array(range(0,X_train_Np.shape[0]-BatchSize,BatchSize))

# Run net
steps = 5000

with tf.Session() as sess:
    
    sess.run(Init)
    
    for i in range(steps):
        
        BastchNum = np.mod(i,len(Batches)-1)
        #print(BastchNum)
        batch_x = X_train_Np[Batches[BastchNum] : Batches[BastchNum+1] ,:]
        batch_y = Y_train_Np[Batches[BastchNum] : Batches[BastchNum+1] ]
        batch_y_OH  = label_binarizer.transform(batch_y)
        batch_y_OH = np.array(batch_y_OH,dtype=np.float32)
        sess.run(Train,feed_dict={X:batch_x,y:batch_y_OH,Hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={X:X_test_Np,y:Y_test_OH,Hold_prob:1.0}))
            print('\n')
            Conf = tf.confusion_matrix(tf.arg_max(y_pred,1),tf.arg_max(y,1))
            C1 = sess.run(Conf,feed_dict={X:X_test_Np,y:Y_test_OH,Hold_prob:1.0})
# Plot Confusion matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
Conf = tf.confusion_matrix(tf.arg_max(y_pred,1),tf.arg_max(y,1))
plot_confusion_matrix(C1,target_names=[*Activety_Types])




