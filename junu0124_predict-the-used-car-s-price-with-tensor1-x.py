import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import pandas as pd

import numpy as np
# load data 

#data = pd.read_csv('../input/used-cars-database/autos.csv',encoding = 'cp1252')

data = pd.read_csv('../input/used-cars-database/autos.csv',encoding = 'cp1252')

original = data.copy()



# information of numerical features 

data.describe()
# drop out some irrelvant features

col = data.columns

data.drop(['dateCrawled','monthOfRegistration','dateCreated','nrOfPictures','postalCode','lastSeen']

         ,axis = 'columns',inplace = True)



# check whether data has correctly changed

data.sample(3)
# find out severe outliner

# print(data.loc[data.yearOfRegistration > 2017].count()['yearOfRegistration'])

# print(data.loc[data.yearOfRegistration < 1990].count()['yearOfRegistration'])

# print(data.loc[data.price < 100].count()['price'])

# print(data.loc[data.price > 50000].count()['price'])

# print(data.loc[data.powerPS < 10].count()['powerPS'])

# print(data.loc[data.powerPS > 2000].count()['powerPS'])
# remove outliner

data = data[(data.yearOfRegistration <= 2017) & (data.yearOfRegistration >= 1990)

          & (data.price>=100) & (data.price <= 50000) & (data.powerPS >= 10) & (data.powerPS <= 2000)]



# print the ratio of remaining data

print(data.count()['name'] / original.count()['name'])
# check null values in each feature

print(data.isnull().sum())



# change null values into the value which shows the highest frequency

feature=[col for col in data.columns if data[col].isnull().sum()>0]

for col in feature:

    value=data[col].value_counts().idxmax()

    data[col]=data[col].fillna(value)



# check null values in each feature 

print(data.isnull().sum())
# find out unique values of each features

print(data.seller.unique())

print(data.offerType.unique())

print(data.abtest.unique())

print(data.brand.unique())
# Use LabelEncoder to change literal values into numerical values

from sklearn.preprocessing import LabelEncoder

feature = ['name','seller','offerType','abtest','vehicleType','gearbox','model','fuelType','brand','notRepairedDamage']

for cols in feature:

     data[cols] = LabelEncoder().fit_transform(data[cols])
# to minimize the difference between the values of each feature

for col in data.columns:

    value=data[col]

    data[col]=np.log1p(value)



# data after preprocessing

data.describe()
label = data[['price','yearOfRegistration','powerPS','kilometer'] + [x for x in feature]]

label.corr().loc[:,'price'].abs().sort_values(ascending = False)[1:]
data.drop(['notRepairedDamage','brand','name','vehicleType','offerType','seller','model','abtest','gearbox'], 

         axis = 'columns',inplace = True)
x_data = data.drop("price",axis = 1).values

y_data = data['price'].values

y_data = y_data.reshape(len(y_data),1)
DIV_RATE = 0.1
TRAIN_RATE = 1 - DIV_RATE

DATA_LEN = len(x_data)

TRAIN_LEN = (int)(DIV_RATE * DATA_LEN)



x_train = []

y_train = []



i = 0

while i < DATA_LEN:

    st_idx = i

    end_idx = i + TRAIN_LEN

    

    print(st_idx, end_idx)

    

    if end_idx >= DATA_LEN: break

    

    x_train.append(x_data[st_idx : end_idx])

    y_train.append(y_data[st_idx : end_idx])

    

    i = end_idx - 1
COL_CNT = len(x_data[0])



X = tf.placeholder(tf.float32, shape = [None, COL_CNT])

Y = tf.placeholder(tf.float32, shape = [None, 1])



W1 = tf.Variable(tf.random_normal([COL_CNT, COL_CNT * 2]), name='weight1')

b1 = tf.Variable(tf.random_normal([COL_CNT * 2]), name='bias1')

layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)



W2 = tf.Variable(tf.random_normal([COL_CNT * 2, COL_CNT]), name='weight2')

b2 = tf.Variable(tf.random_normal([COL_CNT]), name='bias2')

layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)



W3 = tf.Variable(tf.random_normal([COL_CNT, 1]), name='weight3')

b3 = tf.Variable(tf.random_normal([1]), name='bias3')

hypothesis = tf.matmul(layer2, W3) + b3
#Setting Variables

LEARN_RATE = 1e-2

Y_GAP = 1
cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.AdamOptimizer(learning_rate = LEARN_RATE).minimize(cost)



predicted = tf.cast((hypothesis-Y) * (hypothesis-Y) < Y_GAP * Y_GAP, dtype = tf.float32)

accuracy = tf.reduce_mean(predicted)
#Setting Variables

EPOCH = 10001



COST_GAP = 1e-4

BREAK_CNT = 2000
def k_fold(test_idx):

    global x_train_real

    global y_train_real



    x_train_real = x_train[0 : test_idx]

    x_train_real = x_train_real + x_train[test_idx + 1 : len(x_train)]

    x_train_real = np.array(x_train_real)

    x_train_real = x_train_real.reshape(len(x_train[i]) * (len(x_train) - 1), 4)

        

    y_train_real = y_train[0 : test_idx]

    y_train_real = y_train_real + y_train[test_idx + 1 : len(x_train)]

    y_train_real = np.array(y_train_real)

    y_train_real = y_train_real.reshape(len(x_train[i]) * (len(x_train) - 1), 1)
with tf.Session() as sess:

    Accuracy = []

    test_idx = 0

    

    for i in range(len(x_train)):

        sess.run(tf.global_variables_initializer())

        bfcost = 0

        numcnt = 0



        k_fold(test_idx)

        

        print("========= SET %d =========" % (i + 1))

        for step in range(EPOCH):

            cost_val, W_val, _, h, yy = sess.run(

                [cost, W1, optimizer, hypothesis, Y], feed_dict = {X: x_train_real, Y: y_train_real})

            if step % 100 == 0:

                print("[%d]" % (i + 1),end = ' ')

                print("STEP: %5d" % step, end = ' ')

                print("COST: %.5f" % cost_val)

            

            if (bfcost-cost_val) * (bfcost-cost_val) < COST_GAP * COST_GAP :

                numcnt += 1

            else:

                numcnt = 0

            if numcnt == BREAK_CNT:

                print("[%d]" % (i + 1), end = ' ')

                print("STEP: %5d" % step, end = ' ')

                print("COST: %.5f" % cost_val)

                break

            bfcost = cost_val

            

        c = sess.run([accuracy], feed_dict = {X: x_train[test_idx], Y: y_train[test_idx]})

        print("[%d]" % (i + 1), end = ' ')

        print("Accuracy: ",c,"\n")

        

        Accuracy.append(c)

        

        test_idx += 1

        

    print("\nAll Accuracy: ", np.mean(Accuracy))