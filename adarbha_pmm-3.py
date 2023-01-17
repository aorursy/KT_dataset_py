# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_X_train = pd.read_csv("../input/X_train.csv")

df_X_test = pd.read_csv("../input/X_test.csv")

df_y_train = pd.read_csv("../input/y_train.csv")



def check_for_nas(df):

    return sum(df.isnull().sum())



print("Number of na values in X_train: {}".format(check_for_nas(df_X_train)))

print("Number of na values in X_test: {}".format(check_for_nas(df_X_test)))

print("Number of na values in y_train: {}".format(check_for_nas(df_y_train)))
df_y_train['surface'].value_counts()
def convert_q_e(list_):

    '''Convert quaternion to euler angles.

    list_ has 4 elements in the order mentioned below

    or_x - X orientation

    or_y - Y orientation

    or_z - Z orientation

    or_w - W orientation

    Returns a tuple of (roll,pitch,yaw)'''

    or_x = list_[0]

    or_y = list_[1] 

    or_z = list_[2] 

    or_w = list_[3] 

    

    sinr_cosp = 2 * ((or_w * or_x) + (or_y * or_z))

    cosr_cosp = 1 - 2 * ((or_x * or_x) + (or_y * or_y))

    roll = np.arctan2(sinr_cosp, cosr_cosp)

    

    pitch = None

    sinp = 2 * ((or_w * or_y) - (or_z * or_x))

    if abs(sinp) >= 1:

        pitch = np.copysign(np.pi/2,sinp)

    else:

        pitch = np.arcsin(sinp)

    

    siny_cosp = 2 * ((or_w * or_z) + (or_x * or_y))

    cosy_cosp = 1 - 2 * ((or_y * or_y) + (or_z * or_z))

    yaw = np.arctan2(siny_cosp, cosy_cosp)

    

    return(roll, pitch, yaw)



# Using apply to convert orientation to euler angles

orientation_col_names = ['orientation_X','orientation_Y','orientation_Z','orientation_W']

df_X_train_euler = df_X_train.loc[:,orientation_col_names].apply(convert_q_e,axis=1,result_type='expand')

df_X_train_euler.rename(columns={0:'roll',1:'pitch',2:'yaw'},inplace = True)



df_X_test_euler = df_X_test.loc[:,orientation_col_names].apply(convert_q_e,axis=1,result_type='expand')

df_X_test_euler.rename(columns={0:'roll',1:'pitch',2:'yaw'}, inplace=True)

    

# df_X_train_processed = df_X_train.drop(orientation_col_names,axis=1)

df_X_train_processed = pd.concat([df_X_train,df_X_train_euler], axis = 1)

# Need to remove row_id and measurement_number as they are redundant

redundant_cols = ['row_id','measurement_number']

df_X_train_processed = df_X_train_processed.drop(redundant_cols,axis=1)



# df_X_test_processed = df_X_test.drop(orientation_col_names,axis=1)

df_X_test_processed = pd.concat([df_X_test,df_X_test_euler],axis = 1)

df_X_test_processed = df_X_test_processed.drop(redundant_cols, axis = 1)

   
df_X_test_processed.head()
# Function to address 1

integration = lambda x: 0.001 * (x.diff().sum())



# Function to address 2

mean_abs = lambda x:x.abs().mean()



# Function to address 3

sum_abs = lambda x:x.abs().sum()



# Function to address 4

mean_diff = lambda x:x.mean()



# Function to address 11

def modified_mean_diff(x):

    res = [0]

    res.extend(list(np.diff(x)))

    return np.mean(res)

# Function to create aggregates

def create_aggregates(df,function,appendage,cols_affected = None):

    '''Returns an aggregate dataset based on group_by dataset. Appendage modifies colnames by adding this string to the end of the colname

    df - Pandas Dataframe Object on which group by is performed

    cols_affected - list of columns on which aggregation needs to be performed

    function - function to aggregate groupby df on

    appendage - string to modify colnames'''

    if cols_affected:

        cols_affected.append('series_id')

        df_sub = df.loc[:,cols_affected].copy()

        group_by_df = df_sub.groupby(by=['series_id'])

    else:

        group_by_df = df.groupby(by=['series_id'])

    

    df_ = group_by_df.agg(function).rename(mapper = lambda x:x + "_" + appendage, axis = 'columns').copy()

    return df_

    
# Create feature datasets based on bullet points above. Merge them later as a final step

#Step - 1

#Train

df_X_train_step_1 = create_aggregates(df_X_train_processed,function=integration,appendage='integral')

#Test

df_X_test_step_1 = create_aggregates(df_X_test_processed,function=integration,appendage='integral')



#Step - 2

#Train

df_X_train_step_2 = create_aggregates(df_X_train_processed,function=mean_abs,appendage='mean_abs')

#Test

df_X_test_step_2 = create_aggregates(df_X_test_processed,function=mean_abs,appendage='mean_abs')



#Step - 3

#Train

df_X_train_step_3 = create_aggregates(df_X_train_processed,function=sum_abs,appendage='sum_abs')

#Test

df_X_test_step_3 = create_aggregates(df_X_test_processed,function=sum_abs,appendage='sum_abs')



#Step - 4

#Train

df_X_train_step_4 = create_aggregates(df_X_train_processed,function=mean_diff,appendage='mean_diff')

#Test

df_X_test_step_4 = create_aggregates(df_X_test_processed,function=mean_diff,appendage='mean_diff')



#Step - 5

#Train

df_X_train_step_5 = create_aggregates(df_X_train_processed,function=np.mean,appendage='mean')

#Test

df_X_test_step_5 = create_aggregates(df_X_test_processed,function=np.mean,appendage='mean')



#Step - 6

#Train

df_X_train_step_6 = create_aggregates(df_X_train_processed,function=np.std,appendage='std')

#Test

df_X_test_step_6 = create_aggregates(df_X_test_processed,function=np.std,appendage='std')



#Step - 7 

#Train

df_X_train_step_7 = create_aggregates(df_X_train_processed,function=np.max,appendage='max')

#Test

df_X_test_step_7 = create_aggregates(df_X_test_processed,function=np.max,appendage='max')



#Step - 8

#Trainm

df_X_train_step_8 = create_aggregates(df_X_train_processed,function=np.min,appendage='min')

#Test

df_X_test_step_8 = create_aggregates(df_X_test_processed,function=np.min,appendage='min')



# Step-9

#Train

df_X_train_step_9 = create_aggregates(df_X_train_processed,function=np.ptp,appendage='ptp')

#Test

df_X_test_step_9 = create_aggregates(df_X_test_processed,function=np.ptp,appendage='ptp')



# Step - 10

#Train

df_X_train_step_10 = create_aggregates(df_X_train_processed,function=np.median,appendage='median')

#Test

df_X_test_step_10 = create_aggregates(df_X_test_processed,function=np.median,appendage='median')



# Step - 11

#Train

df_X_train_step_11 = create_aggregates(df_X_train_processed,function=modified_mean_diff,appendage='mod_diff')

#Test

df_X_test_step_11 = create_aggregates(df_X_test_processed,function=modified_mean_diff,appendage='mod_diff')
X_train_master = pd.concat([df_X_train_step_1,df_X_train_step_2,df_X_train_step_3,df_X_train_step_4,df_X_train_step_5,df_X_train_step_6,df_X_train_step_7,df_X_train_step_8,df_X_train_step_9,df_X_train_step_10,df_X_train_step_11],axis=1,join='inner')

X_test_master = pd.concat([df_X_test_step_1,df_X_test_step_2,df_X_test_step_3,df_X_test_step_4,df_X_test_step_5,df_X_test_step_6,df_X_test_step_7,df_X_test_step_8,df_X_test_step_9,df_X_test_step_10,df_X_test_step_11],axis=1,join='inner')
X_train_master.head()
# Scale the datasets before proceeding to classification tasks

from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = StandardScaler()

X_train_master_scaled = scaler.fit_transform(X_train_master)

X_test_master_scaled = scaler.fit_transform(X_test_master)

# Y_train is not numeric yet - convert using LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_labels_encoded = le.fit_transform(df_y_train['surface'])
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier



# param_grid = {'n_estimators':[500],'max_depth':[5],'max_features':['log2']}



# param_grid = {'n_estimators':[500]}

gdbt = GradientBoostingClassifier()



# gscv = GridSearchCV(estimator=gdbt,param_grid=param_grid,cv=5)

gscv = GradientBoostingClassifier(n_estimators=500,max_depth=5,max_features='log2')

gscv.fit(X_train_master_scaled,y_labels_encoded)
# print("Best score: {}".format(gscv.best_score_))

# print("Best Classifier: {}".format(gscv.best_estimator_))
import csv

def write_to_csv(test_array,classifier,encoder,output_file_name="predictions.csv"):

    '''Writes a sample csv

    test_array - numpy array; this needs to be sorted on series_id

    classifier - result from cross vaidation analysis

    encoder - label encoder used for labels'''

    predictions = classifier.predict(test_array)

    transformed_predictions = list((encoder.inverse_transform(predictions)))

    series_id = list(range(len(transformed_predictions)))

    df_ = pd.DataFrame({'series_id':series_id,'surface':transformed_predictions})

    df_.to_csv(output_file_name,index=False)

    return df_

    

# write_to_csv(X_test_master_scaled,gscv.best_estimator_,le)

write_to_csv(X_test_master_scaled,gscv,le)
# from sklearn.preprocessing import OneHotEncoder

# surface_labels = np.array(df_y_train['surface']).reshape(-1,1)

# ohe = OneHotEncoder(handle_unknown='ignore')

# surface_labels_ohe = ohe.fit_transform(surface_labels).toarray()
# import tensorflow as tf



# x = tf.placeholder(tf.float32,[None,X_train_master.shape[1]])



# W1 = tf.Variable(tf.random.normal([X_train_master.shape[1],256]))

# b1 = tf.Variable(tf.zeros([1,256]))

# Z1 = tf.matmul(x,W1) + b1

# sigma1 = tf.nn.relu(Z1)



# W2 = tf.Variable(tf.random.normal([256,9]))

# b2 = tf.Variable(tf.zeros([1,9]))

# Z2 = tf.matmul(sigma1, W2) + b2

# sigma2 = tf.nn.sigmoid(Z2)



# y = tf.nn.softmax(sigma2)

# y_ = tf.placeholder(tf.float32,[None,surface_labels_ohe.shape[1]])







# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cross_entropy)





# init = tf.global_variables_initializer()



# max_iter = 10000



# with tf.Session() as sess:

#     sess.run(init)

#     iters = 0

#     while iters < max_iter:

# #         for i in range(X_train_master_scaled.shape[0]):

# #         print(sess.run(W1,feed_dict={x:X_train_master_scaled[i,:].reshape(1,93),y_:surface_labels_ohe[i,:].reshape(1,9)}))

#         sess.run(train_step,feed_dict={x:X_train_master_scaled.reshape(-1,93),y_:surface_labels_ohe.reshape(-1,9)})

# #         print(sess.run(cross_entropy,feed_dict={x:X_train_master_scaled.reshape(-1,93),y_:surface_labels_ohe.reshape(-1,9)}))

#         iters = iters + 1

    

#     prediction = tf.argmax(y,1)

#     predicted_labels = prediction.eval(feed_dict={x: X_train_master_scaled.reshape(-1,93)})

#     correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)),tf.float32)) / 3810

#     print(correct_prediction.eval(feed_dict={x:X_train_master_scaled.reshape(-1,93),y_:surface_labels_ohe.reshape(-1,9)}))

    

#     print(prediction.eval(feed_dict={x: X_train_master_scaled.reshape(-1,93)}))

    

# y = tf.nn.softmax(tf.matmul(x,W) + b)



# y_ = tf.placeholder(tf.float32,[None,surface_labels_ohe.shape[1]])



# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



# sess = tf.Session()

# tf.global_variables_initializer().run(session=sess)



# train_features = X_train_master[10000:]

# train_labels = surface_labels_ohe[10000:]



# for _ in range(1000):

#   batch_xs, batch_ys = train_features, train_labels

#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



    

# sess.close()

# # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

prediction = tf.argmax(y,1)

predicted_labels = prediction.eval(feed_dict={x: train_features})

# print(prediction.eval(feed_dict={x: train_features}))

import numpy as np

import pandas as pd

import tensorflow as tf
w = tf.Variable([0],dtype=tf.float32)

x = tf.placeholder(tf.float32,[3,1])

cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]

coeff = np.array([[1],[-10],[25]])

W1 = tf.Variable(tf.random.normal([X_train_master.shape[1],128]))



train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)



init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    print(sess.run(W1))