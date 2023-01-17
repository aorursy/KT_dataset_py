import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error

import gc

import tensorflow as tf
df_train_X = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/x_train.csv')

df_train_Y = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/y_train.csv')

df_test_X = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/x_test.csv')

df_test_Y = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/y_test.csv')

df_Submission_X = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/submissionX.csv')
itemsDF = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/items_reindex.csv')
def NWRMSLE(y, pred, w):

    return mean_squared_error(y, pred, sample_weight=w)**0.5
df_train_X.drop(['Unnamed: 0'], inplace=True,axis=1)

df_test_X.drop(['Unnamed: 0'], inplace=True,axis=1)

df_train_Y.drop(['Unnamed: 0'], inplace=True,axis=1)

df_test_Y.drop(['Unnamed: 0'], inplace=True,axis=1)

df_Submission_X.drop(['Unnamed: 0'], inplace=True,axis=1)
numFeatures = df_train_X.shape[1]

numLabels = 1

hiddenUnit = 20

learningRate = 0.01

numEpochs = 1000
x = tf.placeholder(tf.float64, [None, numFeatures],name="X_placeholder")

y_ = tf.placeholder(tf.float64, [None, numLabels],name="Y_placeholder")
weights = tf.Variable(tf.random_normal([numFeatures,hiddenUnit],stddev=0.1,name="weights", dtype=tf.float64))

weights2 = tf.Variable(tf.random_normal([hiddenUnit,1],name="weights2", dtype=tf.float64))
bias = tf.Variable(tf.random_normal([1,hiddenUnit],stddev=0.1,name="bias", dtype=tf.float64))

bias2 = tf.Variable(tf.random_normal([1,1],stddev=0.1,name="bias2", dtype=tf.float64))
weightsNWR = tf.placeholder(tf.float32, [None, 1],name="weightsNWR")
itemWeightsTrain = pd.concat([itemsDF["perishable"]] * 6) * 0.25 + 1

itemWeightsTrain = np.reshape(itemWeightsTrain,(itemWeightsTrain.shape[0], 1))
itemWeightsTest = itemsDF["perishable"]* 0.25 + 1

itemWeightsTest = np.reshape(itemWeightsTest,(itemWeightsTest.shape[0], 1))
y = tf.matmul(x,weights) + bias
y = tf.nn.relu(y)
epsilon = 1e-3

batch_mean2, batch_var2 = tf.nn.moments(y,[0])

scale2 = tf.Variable(tf.ones([hiddenUnit],dtype=tf.float64),dtype=tf.float64)

beta2 = tf.Variable(tf.zeros([hiddenUnit],dtype=tf.float64),dtype=tf.float64)

y = tf.nn.batch_normalization(y,batch_mean2,batch_var2,beta2,scale2,epsilon)
dropout_placeholder = tf.placeholder(tf.float64,name="dropout_placeholder")

y=tf.nn.dropout(y,dropout_placeholder)
#create 1 more hidden layer

y = tf.matmul(y,weights2)+bias2
y = tf.nn.relu(y)
loss = tf.losses.mean_squared_error(predictions=y,labels=y_,weights=weightsNWR)

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
val_pred_nn = []

test_pred_nn = []

cate_vars_nn = []

submit_pred_nn=[]



trainingLoss=[]

validationLoss=[]





#step through all the dates(16)

for i in range(16):

    print("Step %d" % (i+1))

    

    trainY_NN = np.reshape(df_train_Y.iloc[:,i],(df_train_Y.shape[0], 1))

    testY_NN = np.reshape(df_test_Y.iloc[:,i],(df_test_Y.shape[0], 1))

    

    for epoch in range(numEpochs):

        _,loss = sess.run([optimizer,cost], feed_dict={x: df_train_X, y_: trainY_NN,weightsNWR:itemWeightsTrain,dropout_placeholder:0.6})



        if epoch%100 == 0:

            print('Epoch', epoch, 'completed out of',numEpochs,'loss:',loss)

            #trainingLoss.append(loss)

            #check against test dataset

            test_pred = sess.run(cost, feed_dict={x:df_test_X,y_: testY_NN,weightsNWR:itemWeightsTest,dropout_placeholder:1.0})

            print('Acc for test dataset ',test_pred)

            #validationLoss.append(test_pred)

    

    tf_pred = sess.run(y,feed_dict={x:df_test_X,weightsNWR:itemWeightsTest,dropout_placeholder:1.0})

    tf_predY = np.reshape(tf_pred,(tf_pred.shape[0],))

    test_pred_nn.append(tf_predY)

    print('score for step',(i+1))

    print("Validation mse:", mean_squared_error(df_test_Y.iloc[:,i], tf_predY))

    print('NWRMSLE:',NWRMSLE(df_test_Y.iloc[:,i], tf_predY,itemsDF["perishable"]*0.25+1))



    #predict for submission set

    nn_submit_predY = sess.run(y,feed_dict={x:df_Submission_X,dropout_placeholder:1.0})

    nn_submit_predY = np.reshape(nn_submit_predY,(nn_submit_predY.shape[0],))

    submit_pred_nn.append(nn_submit_predY)

    

    gc.collect()

    sess.run(tf.global_variables_initializer())
nnTrainY= np.array(test_pred_nn).transpose()

pd.DataFrame(nnTrainY).to_csv('nnTrainY.csv')

nnSubmitY= np.array(submit_pred_nn).transpose()

pd.DataFrame(nnSubmitY).to_csv('nnSubmitY.csv')
print('NWRMSLE:',NWRMSLE(df_test_Y,nnTrainY,itemsDF["perishable"]* 0.25 + 1))
#to reproduce the testing IDs

df_train = pd.read_csv(

    '../input/favorita-grocery-sales-forecasting/train.csv', usecols=[1, 2, 3, 4, 5],

    dtype={'onpromotion': bool},

    converters={'unit_sales': lambda u: np.log1p(

        float(u)) if float(u) > 0 else 0},

    parse_dates=["date"],

    skiprows=range(1, 66458909)  # 2016-01-01

)



df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]

del df_train



df_2017 = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(

        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)
#submitDF = pd.read_csv('../input/testforsubmit/testForSubmit.csv',index_col=False)

df_test = pd.read_csv(

    "../input/favorita-grocery-sales-forecasting/test.csv", usecols=[0, 1, 2, 3, 4],

    dtype={'onpromotion': bool},

    parse_dates=["date"]  # , date_parser=parser

).set_index(

    ['store_nbr', 'item_nbr', 'date']

)
print("Making submission...")

combinedSubmitPredY = nnSubmitY

df_preds = pd.DataFrame(

    combinedSubmitPredY, index=df_2017.index,

    columns=pd.date_range("2017-08-16", periods=16)

).stack().to_frame("unit_sales")

df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission[['id','unit_sales']].to_csv('submit_nn.csv',index=None)