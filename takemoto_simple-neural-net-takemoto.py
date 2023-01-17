# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import types

from sklearn.linear_model import LinearRegression

from scipy.sparse import coo_matrix, hstack

from scipy import io

import tensorflow as tf

import math

import sys, csv, h5py



from scipy.sparse import coo_matrix, hstack

from scipy import io

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# pandasのデータフレームを返す

# train_or_testには'train'か'test'を入れる

def load_data(path,train_or_test,brand_threshold = 100,category_threshold = 50,frequent_brands=None,frequent_categories=None):

    data_pd = pd.read_csv(path, error_bad_lines=False, encoding='utf-8', header=0, delimiter='\t')

    #ブランド名がないものを'NO_BRAND'とする

    data_pd['brand_name'] = data_pd['brand_name'].fillna('NO_BRAND')

    data_pd=data_pd.fillna("")



    if train_or_test == 'train':

        frequent_brands = data_pd['brand_name'].value_counts()[data_pd['brand_name'].value_counts()>brand_threshold].index

        frequent_categories = data_pd['category_name'].value_counts()[data_pd['category_name'].value_counts()>category_threshold].index

    elif train_or_test != 'test':

        print('Error : Please input "train" or "test" in train_or_test')

        return

    

    if type(frequent_brands)==type(None) or type(frequent_categories)==type(None):

        print('Error : Please load train data first')

        return

    else:

        data_pd.loc[~data_pd['brand_name'].isin(frequent_brands),'brand_name']= 'SOME_BRAND'

        data_pd.loc[~data_pd['category_name'].isin(frequent_categories),'category_name'] = 'SOME_CATEGORY'

        

    return data_pd,frequent_brands,frequent_categories
csv_train_path = u'../input/train.tsv'

csv_test_path = u'../input/test.tsv'

train_data_pd, frequent_brands, frequent_categories = load_data(csv_train_path,'train',brand_threshold=100,category_threshold=50)

test_data_pd, _, _ = load_data(csv_test_path,'test',frequent_brands=frequent_brands,frequent_categories=frequent_categories)

print('loading data completed')
use_cols = ['item_condition_id','brand_name','shipping','category_name']

train_num = len(train_data_pd)

test_num = len(test_data_pd)
prices = np.array(train_data_pd['price'])

prices_log = np.log(prices+1)
# scipyのsparse matrix(coo_matrix)X_transform と 変数のリストvariables を返す

# save_pathに何も指定しない場合ファイルを保存しない 指定した場合指定したディレクトリ内に保存する

def make_onehot(use_cols,data_pd,train_or_test,save_path=None):

    variables = []

    flag = 0

    for use_col in use_cols:

        dummy_pd = pd.get_dummies(data_pd[use_col]).astype(np.uint8)

        if flag==0:

            X_transform = coo_matrix(dummy_pd.values)

            flag=1

        else:

            X_transform = hstack([X_transform,coo_matrix(dummy_pd.values)])

        

        variables.extend( list( dummy_pd.columns ) )

        

        if save_path is not None:

            if train_or_test != 'test' and train_or_test != 'train':

                print('Error : Please input "train" or "test" in train_or_test')

                return

            save_path_ = '{}/{}_{}.csv'.format(save_path,use_col,train_or_test)

            dummy_pd.to_csv(save_path_,index=False,encoding="utf8")

            

    if save_path is not None:

        # sparse matrixの保存

        io.savemat("{}/X_transform_{}".format(save_path,train_or_test), {"X_transform":X_transform})

        print('sparse matrixを保存しました。次回からはsparse matrixを読み込んで学習に利用してください')



    return X_transform,np.array(variables)
X_transform_train,variables = make_onehot(use_cols,train_data_pd,'train',save_path=None)

X_transform_test,variables_ = make_onehot(use_cols,test_data_pd,'test',save_path=None)

print('converting data completed')
# NNの学習

MAX_EPOCH = 5

BATCH_SIZE = 1000

UNIT_NOS = [500,50]

features = X_transform_test.shape[1]

data_path = "working_dir"

patience = 2



max_size = 100000000

hidden_no = len(UNIT_NOS)

x = tf.placeholder(tf.float32, [None, features])

W_list = []

b_list = []



old_unit_no = features

z = x

for i, unit_no in enumerate(UNIT_NOS):

    W = tf.Variable(tf.random_normal([old_unit_no, unit_no], mean=0.0, stddev=0.05))

    b = tf.Variable(tf.constant(0.1, shape=[unit_no]))

    W_list.append(W)

    b_list.append(b)

    z = tf.nn.relu(tf.matmul(z, W) + b)

    old_unit_no = unit_no

W_last = tf.Variable(tf.random_normal([UNIT_NOS[-1], 1], mean=0.0, stddev=0.05))

b_last = tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.05))

y = tf.matmul(z, W_last) + b_last



y_ = tf.placeholder(tf.float32, [None, 1])

mse = tf.reduce_mean((y - y_) * (y - y_))



train_step = tf.train.AdamOptimizer(1e-2).minimize(mse)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)



#学習の保存

saver = tf.train.Saver(max_to_keep=1)

ckpt = tf.train.get_checkpoint_state(data_path)

if ckpt and ckpt.model_checkpoint_path:

    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)

    saver.restore(sess, ckpt.model_checkpoint_path)

# データ取得

# # sparse matrixの読み込み

# X_transform = io.loadmat("../../../onehots/X_transform_train")["X_transform"]

# y = np.load('../../../onehots/y_log.npy')

# X_transform_test = io.loadmat("../../../onehots/X_transform_test")["X_transform"]



train_X, test_X, train_y, test_y = train_test_split(X_transform_train, prices_log, test_size=0.1, random_state=42)

train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)



print('data_length:'+str(train_X.shape[0]))

batch_no = int( (train_X.shape[0] - 1) / BATCH_SIZE + 1)

print('batch_no:'+str(batch_no))

count = 0
min = np.inf

for i in range(MAX_EPOCH):

    print("epoch:"+str(i+1))



    # SGDを実装している

    for j in range(batch_no):

        batch_xs = (train_X.toarray())[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

        batch_ys = train_y[j * BATCH_SIZE:(j + 1) * BATCH_SIZE].reshape(-1, 1)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#         if j%100==1:

#             print('   batch:'+str(j))

    train_cost = sess.run(mse, feed_dict={x: (train_X.toarray())[:BATCH_SIZE], y_: train_y[:BATCH_SIZE].reshape(-1, 1)})

    valid_cost = sess.run(mse, feed_dict={x: valid_X.toarray(), y_: valid_y.reshape(-1, 1)})

    print (str(i + 1) + "epoch:train cost(rmse)=" + str(math.sqrt(train_cost)) + ", rmse=" + str(math.sqrt(valid_cost)) )



    if valid_cost < min:

        count = 0

        # for i, W, b in zip(range(hidden_no), W_list, b_list):

        #     np.savetxt(path + "W" + str(i + 1) + ".csv", sess.run(W), delimiter=",")

        #     np.savetxt(path + "b" + str(i + 1) + ".csv", sess.run(b), delimiter=",")

        # np.savetxt(path + "W.csv", sess.run(W_last), delimiter=",")

        # np.savetxt(path + "b.csv", sess.run(b_last), delimiter=",")

        min = valid_cost

    else:

        count += 1





    # 改善されなかった回数がpatience回以上で学習終了

    if count >= patience:

        break

print ("test rmse=" + str(math.sqrt(sess.run(mse, feed_dict={x: test_X.toarray(), y_: test_y[:].reshape(-1, 1)}))) )

prediction_log = sess.run(y,feed_dict={x:X_transform_test.toarray()}).reshape(-1)

prediction = np.exp(prediction_log)-1

test_id = np.arange(prediction.shape[0])
submission = pd.DataFrame([])

submission['test_id'] = test_id

submission['price'] = pd.DataFrame(prediction)

submission.to_csv('submission.csv',index=None)

print(submission.iloc[:10])