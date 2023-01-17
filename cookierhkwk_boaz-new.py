# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# from sklearn.preprocessing import MinMaxScaler

# from sklearn.model_selection import KFold, train_test_split

# import seaborn as sns

# import tensorflow as tf

# from sklearn.utils import shuffle

# from imblearn.under_sampling import RandomUnderSampler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cat = pd.read_csv("/kaggle/input/submisson/submisson_cat.csv")

nn = pd.read_csv('/kaggle/input/submisson/submission_nn.csv')

xg = pd.read_csv('/kaggle/input/submisson/submission_xg.csv')

rf = pd.read_csv('/kaggle/input/submisson/submisson_Rf.csv')

lg = pd.read_csv('/kaggle/input/submisson/submission_lgbm.csv')
cat.head()
lg.head()
xg.head()
nn.head()
rf.head()
#ex = 0.5*lg + 0.5 * cat

#ex.head()
submission = 0.5*lg +0.4*cat + 0.005*rf + 0.005* xg
submission['id'] = cat['id']
submission.head()
submission.to_csv('final_sum.csv', index = False)
# train = pd.read_csv('/kaggle/input/train-nova/train_nona.csv', index_col = 0)

# test = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
# for col in train.columns:

#     if 'calc' in col:

#         train = train.drop(col, axis=1)

        

# for col in test.columns:

#     if 'calc' in col:

#         test = test.drop(col, axis=1)
# test = test.drop(['ps_car_03_cat','ps_car_05_cat'], axis=1)

# for col in ['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat','ps_car_02_cat','ps_car_07_cat','ps_car_09_cat']:

#     test[col] = test[col].replace(-1,int(test[col].mode()))
# for col in train.columns:

#     if 'cat' in col:

#         train[col] = train[col].astype(str)



# for col in test.columns:

#     if 'cat' in col:

#         test[col] = test[col].astype(str)
# train = pd.get_dummies(train)

# test = pd.get_dummies(test)
# mmscaler = MinMaxScaler()

# for col in train.drop(['id','target'],axis=1).columns:

#     mmscaler.fit(np.array(train[col]).reshape(-1,1))

#     train[col] = mmscaler.transform(np.array(train[col]).reshape(-1,1))



# for col in test.drop(['id'],axis=1).columns:

#     mmscaler.fit(np.array(test[col]).reshape(-1,1))

#     test[col] = mmscaler.transform(np.array(test[col]).reshape(-1,1))
# target = train['target']

# train = train.drop(['id','target'], axis = 1)

# test = test.drop(['id'], axis = 1)
# print(train.shape)

# print(test.shape)
# X, y = RandomUnderSampler(sampling_strategy= 0.4, random_state=42).fit_sample(train, target)

# X = pd.DataFrame(X, columns = train.columns)
# tf.test.is_gpu_available()
# gpu_options = tf.GPUOptions(visible_device_list="0")
# def to_tensor(data, target, batch_size):



#     ind = pd.DataFrame()

#     reg = pd.DataFrame()

#     car = pd.DataFrame()

#     for col in data.columns:

#         if 'ind' in col:

#             ind = pd.concat([ind, train[col]], axis =1)

#         elif 'reg' in col:

#             reg = pd.concat([reg, train[col]], axis =1)

#         else:

#             car = pd.concat([car, train[col]], axis =1)

#     y = target

    

#     shuffle(ind, random_state =42)

#     shuffle(reg, random_state =42)

#     shuffle(car, random_state =42)

#     shuffle(y, random_state =42)

    

#     length = len(ind)

#     max_batch = length // batch_size +1

#     index = 0

#     i = 0

    

#     while index<length:

#         try:

#             batch_ind = ind.iloc[index : index + batch_size]

#             batch_reg = reg.iloc[index : index + batch_size]

#             batch_car = car.iloc[index : index + batch_size]

#             batch_y = y[index : index + batch_size]

#         except IndexError:

#             batch_ind = ind.iloc[index:]

#             batch_reg = reg.iloc[index:]

#             batch_car = car.iloc[index:]

#             batch_y = y[index:]

    

#         batch_ind = np.array(batch_ind)

#         batch_reg = np.array(batch_reg)

#         batch_car = np.array(batch_car)



#         batch_y = np.array(batch_y).astype(int)

#         batch_y = np.eye(2)[batch_y]



#         index += batch_size

#         left = length - index

#         i += 1



#         yield i, batch_ind, batch_reg, batch_car, batch_y, left, max_batch
# X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=42)
# def early_stopping_and_save_model(sess, saver, input_vali_loss, early_stopping_val_loss_list):



#     if len(early_stopping_val_loss_list) != early_stopping_patience:

#         early_stopping_val_loss_list = [99.99 for _ in range(early_stopping_patience)]



#     early_stopping_val_loss_list.append(input_vali_loss)

#     if input_vali_loss < min(early_stopping_val_loss_list[:-1]):

#         saver.save(sess, os.getcwd()+'/model.ckpt')

#         early_stopping_val_loss_list.pop(0)



#         return True, early_stopping_val_loss_list



#     elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list):

#         return False, early_stopping_val_loss_list



#     else:

#         return True, early_stopping_val_loss_list
# dropout_rate = 0.3

# epoch_num = 20000

# batch_size = 4096

# learning_rate = 0.001

# early_stopping_patience = 30
# tf.reset_default_graph()

# ind_data = tf.placeholder(tf.float32, [None, 28], name='ind_data')

# reg_data = tf.placeholder(tf.float32, [None, 3], name='reg_data')

# car_data = tf.placeholder(tf.float32, [None, 163], name='car_data')

# y_data = tf.placeholder(tf.float32, [None, 2], name='y_data')

# initializer = tf.contrib.layers.xavier_initializer()

# # initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)



# ind_w_1 = tf.get_variable('ind_data', [28,64], initializer = initializer)

# ind_b_1 = tf.Variable(tf.zeros([64]), name='ind_b_1')

# # fc_bn_1 = tf.layers.batch_normalization(tf.matmul(x_data, fc_w_1) + fc_b_1, name = 'bn_1')

# # fc_z_1 = tf.nn.sigmoid(fc_bn_1, name='fc_z_1')

# ind_z_1 = tf.nn.sigmoid(tf.matmul(ind_data, ind_w_1) + ind_b_1, name='ind_z_1')

# ind_d_1 = tf.nn.dropout(ind_z_1, dropout_rate, name='ind_d_1')



# reg_w_1 = tf.get_variable('reg_data', [3,64], initializer = initializer)

# reg_b_1 = tf.Variable(tf.zeros([64]), name='reg_b_1')

# # fc_bn_1 = tf.layers.batch_normalization(tf.matmul(x_data, fc_w_1) + fc_b_1, name = 'bn_1')

# # fc_z_1 = tf.nn.sigmoid(fc_bn_1, name='fc_z_1')

# reg_z_1 = tf.nn.sigmoid(tf.matmul(reg_data, reg_w_1) + reg_b_1, name='reg_z_1')

# reg_d_1 = tf.nn.dropout(reg_z_1, dropout_rate, name='fc_d_1')



# car_w_1 = tf.get_variable('car_data', [163,64], initializer = initializer)

# car_b_1 = tf.Variable(tf.zeros([64]), name='fc_b_1')

# # fc_bn_1 = tf.layers.batch_normalization(tf.matmul(x_data, fc_w_1) + fc_b_1, name = 'bn_1')

# # fc_z_1 = tf.nn.sigmoid(fc_bn_1, name='fc_z_1')

# car_z_1 = tf.nn.sigmoid(tf.matmul(car_data, car_w_1) + car_b_1, name='car_z_1')

# car_d_1 = tf.nn.dropout(car_z_1, dropout_rate, name='fc_d_1')



# combine = tf.concat([ind_d_1, reg_d_1], axis=1, name='combine')

# combine = tf.concat([car_d_1, combine], axis=1, name='combine')



# fc_w_2 = tf.get_variable('combine', [192, 256], initializer = initializer)

# fc_b_2 = tf.Variable(tf.zeros([256]), name='fc_b_2')

# # fc_bn_2 = tf.layers.batch_normalization(tf.matmul(fc_z_1, fc_w_2) + fc_b_2, name = 'bn_2')

# # fc_z_2 = tf.nn.sigmoid(fc_bn_2, name='fc_z_2')

# fc_z_2 = tf.nn.sigmoid(tf.matmul(combine, fc_w_2) + fc_b_2, name='fc_z_2')

# fc_d_2 = tf.nn.dropout(fc_z_2, dropout_rate, name='fc_d_2')



# fc_w_3 = tf.get_variable('fc_d_2', [256, 2], initializer = initializer)

# fc_u_3 = tf.matmul(fc_d_2, fc_w_3, name='fc_u_3')



# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_u_3, labels=y_data), name='loss')

# optimizer = tf.train.AdamOptimizer(learning_rate)

# training = optimizer.minimize(loss)



# pred_y = tf.nn.softmax(fc_u_3, name='pred_y')

# pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_data, 1), name='pred')

# acc = tf.reduce_mean(tf.cast(pred, tf.float32), name='acc')
# batch_index_list = list(range(0, X_train.shape[0], batch_size))

# vali_batch_list = list(range(0, X_vali.shape[0], batch_size))

# train_loss_list, vali_loss_list = [], []

# train_acc_list, vali_acc_list = [], []

# saver = tf.train.Saver()

# early_stopping_val_loss_list = []
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

#     sess.run(tf.global_variables_initializer())

#     for epoch in range(epoch_num):

#         total_loss, total_acc, vali_loss, vali_acc = 0, 0, 0, 0

#         train_gen = to_tensor(X_train, y_train, batch_size)

#         vali_gen = to_tensor(X_vali, y_vali, batch_size)



#         for i, batch_ind, batch_reg, batch_car, batch_y, left, tot in train_gen:

#             sess.run(training, feed_dict={ind_data: batch_ind, reg_data:batch_reg, car_data:batch_car, y_data: batch_y})

#             loss_val, acc_val = sess.run([loss, acc], feed_dict={ind_data: batch_ind, reg_data:batch_reg, car_data:batch_car, y_data: batch_y})    

#             total_loss += loss_val

#             total_acc += acc_val

        

#         for i, vali_ind, batch_reg, batch_car, vali_y, left, tot in vali_gen:

#             tmp_vali_loss, tmp_vali_acc, vali_pred_y = sess.run([loss, acc, pred_y],

#                                                                         feed_dict={ind_data: batch_ind, reg_data:batch_reg, car_data:batch_car, y_data : vali_y})

#             vali_loss += tmp_vali_loss

#             vali_acc += tmp_vali_acc

        

#         train_loss_list.append(total_loss/len(batch_index_list))

#         train_acc_list.append(total_acc/len(batch_index_list))

#         vali_loss_list.append(vali_loss)

#         vali_acc_list.append(vali_acc)

        

#         print('{} / {}'.format(np.argmax(vali_pred_y, axis=1).sum(), len(vali_pred_y)))

#         print('\n#%4d/%d' % (epoch + 1, epoch_num), end='  |  ')

#         print('Avg_loss={:.4f} / Avg_acc={:.4f}'.format(total_loss/len(batch_index_list), total_acc/len(batch_index_list)), end='  |  ')

#         print('vali_loss={:.4f} / vali_acc={:.4f}'.format(vali_loss/len(vali_batch_list), vali_acc/len(vali_batch_list)), end='  |  ')

#         print('-'*100)

        

#         bool_continue, early_stopping_val_loss_list = early_stopping_and_save_model(sess, saver, vali_loss_list[-1], early_stopping_val_loss_list)

#         if not bool_continue:

#             print('{0}\nstop epoch : {1}\n{0}'.format('-' * 100, epoch - early_stopping_patience + 1))

#             break
# X_test = test

# y_test = np.zeros(len(test))
# tf.reset_default_graph()

# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

#     sess.run(tf.global_variables_initializer())

#     saver = tf.train.import_meta_graph(os.getcwd()+'/model.ckpt'+ '.meta')

#     saver.restore(sess, os.getcwd()+'/model.ckpt')



#     test_gen = to_tensor(X_test, y_test, batch_size)



#     ind_data = tf.get_default_graph().get_tensor_by_name('ind_data:0')

#     reg_data = tf.get_default_graph().get_tensor_by_name('reg_data:0')

#     car_data = tf.get_default_graph().get_tensor_by_name('car_data:0')

#     y_data = tf.get_default_graph().get_tensor_by_name('y_data:0')

#     acc = tf.get_default_graph().get_tensor_by_name('acc:0')

#     loss = tf.get_default_graph().get_tensor_by_name('loss:0')

#     pred_y = tf.get_default_graph().get_tensor_by_name('pred_y:0')

#     pred = np.array([])

#     prob = np.array([])

#     for i, test_ind, test_reg, test_car, test_y, left, tot in test_gen:

#         test_loss, test_acc, test_pred_y, test_true_y = sess.run([loss, acc, pred_y, y_data],

#                                                                  feed_dict={ind_data: test_ind, reg_data:test_reg, car_data:test_car,, y_data: test_y})



#         y_prob = test_pred_y[:,1]

# #         y_pred = np.argmax(test_pred_y, axis=1)

# #         pred = np.append(pred, y_pred)

#         prob = np.append(prob, y_prob)

#     final_y = pd.DataFrame({'target' : prob})
# sub = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv')
# sub['target'] = prob
# sub.to_csv('submission.csv', index=False)