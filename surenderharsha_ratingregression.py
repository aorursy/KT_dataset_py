
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tflearn as tf
from sklearn import preprocessing

dataframe=pd.read_csv('../input/games.csv')
dataframe
df= dataframe.drop(["id","type","name","yearpublished","minplaytime","maxplaytime","total_weights","bayes_average_rating"],axis=1)
#Changing Column order
cols = df.columns.tolist()
s=cols[5]
cols=cols[:5]+cols[7:]

cols.append(s)
df=df[cols]

#Hyper Parameters
training_to_test_ratio = 0.8
size=df.shape
epochs=10
batch=24

seperation = int(size[0]*training_to_test_ratio)
df = df.sample(frac=1).reset_index(drop=True)
df=df.fillna(0)
#inputs
training = df[:seperation]
test = df[seperation:]

#df.columns.tolist()
df.isnull().any()
inputX = training.loc[:, df.columns != 'average_rating'].as_matrix()
inputY = training.loc[: , ['average_rating']].as_matrix()
for i in range(len(inputY)):
    inputY[i]=inputY[i]/10.0

inputX

net = tf.input_data(shape=[None, 10])
net= tf.batch_normalization(net)
net = tf.fully_connected(net, 16)
net = tf.dropout(net,0.6)
net = tf.fully_connected(net, 16)
net = tf.dropout(net,0.6)
net = tf.fully_connected(net,1 , activation='sigmoid')
net = tf.regression(net,loss='mean_square')
model = tf.DNN(net)

model.fit(inputX, inputY, n_epoch=epochs, batch_size=batch, show_metric=True)
model.save('AverageRatingAI')
testX = test.loc[:, df.columns != 'average_rating'].as_matrix()
testY = test.loc[: , ['average_rating']].as_matrix()
for i in range(len(testY)):
    testY[i]=testY[i]/10.0
model.evaluate(testX,testY,batch_size=16)
tmp_data=test[4:5]
tmpX = tmp_data.loc[:, df.columns != 'average_rating'].as_matrix()
tmpY = tmp_data.loc[: , ['average_rating']].as_matrix()
for i in range(len(tmpY)):
    tmpY[i]=tmpY[i]/10.0
outY=model.predict(tmpX)
tmp_data
print(outY,tmpY)
