# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 300)

#Read CSV
training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")

def fillna(attribute, value):
    training[attribute] = training[attribute].fillna(value)
    testing[attribute] = testing[attribute].fillna(value)

#Filling in values
training['LotFrontage'] = training['LotFrontage'].fillna(training['LotFrontage'].mean())
testing['LotFrontage'] = testing['LotFrontage'].fillna(testing['LotFrontage'].mean())
fillna("Alley", "None")
fillna("MasVnrType", "None")
fillna("MasVnrArea", 0)
fillna("BsmtQual", "None")
fillna("BsmtCond", "None")
fillna("BsmtExposure", "None")
fillna("BsmtFinType1", "None")
fillna("BsmtFinType2", "None")
fillna("Electrical", "SBrkr")
fillna("FireplaceQu", "None")
fillna("GarageType", "None")
fillna("GarageYrBlt", 0)
fillna("GarageFinish", "None")
fillna("GarageQual", "None")
fillna("GarageCond", "None")
fillna("PoolQC", "None")
fillna("Fence", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
for e in testing.columns:
    testing[e] = testing[e].fillna(testing[e].mode().iloc[0])

#training[training.columns[training.isnull().any()].tolist()].isnull().sum()
#testing[testing.columns[testing.isnull().any()].tolist()].isnull().sum()

#Individual datasets
train_cat = training.iloc[:, np.r_[1:3, 5, 7:9, 10:17, 21:25, 27:33, 35, 39:42, 53, 55, 57:58, 60, 63:65, 73:75, 76, 78:79] ]
train_num = training.iloc[:, np.r_[3:4, 17:20, 26, 34, 36:38, 43:52, 54, 56, 59, 61:62, 66:72, 75, 77] ]
train_result = training.iloc[:, 80].values.reshape(-1, 1)

test_cat = testing.iloc[:, np.r_[1:3, 5, 7:9, 10:17, 21:25, 27:33, 35, 39:42, 53, 55, 57:58, 60, 63:65, 73:75, 76, 78:79] ]
test_num = testing.iloc[:, np.r_[3:4, 17:20, 26, 34, 36:38, 43:52, 54, 56, 59, 61:62, 66:72, 75, 77] ]

cat = np.concatenate([train_cat, test_cat], axis=0)
num = np.concatenate([train_num, test_num], axis=0)

#Create Label Encoder and Onehot encoder
le = LabelEncoder()
enc = OneHotEncoder()

"""train_enc = train_cat.astype(str).apply(le.fit_transform)
enc.fit(train_enc)
train_onehot = enc.transform(train_enc).toarray()

test_enc = test_cat.astype(str).apply(le.fit_transform)
enc.fit(test_enc)
test_onehot = enc.transform(test_enc).toarray()"""

cat = pd.DataFrame(cat)
#One hot encoding of All Categories
overall_enc = cat.astype(str).apply(le.fit_transform)
enc.fit(overall_enc)
onehot = enc.transform(overall_enc).toarray()

#Split Categories into train and test
train_onehot = onehot[:1460]
test_onehot = onehot[1460:]

train = np.concatenate([train_num, train_onehot], axis=1)
test = np.concatenate([test_num, test_onehot], axis=1)

validate = train[1200:]
train_sub = train[:1200]

validate_result = train_result[1200:]
train_sub_result = train_result[:1200]
first_layer = 1
"""beta = 0.0001
keep_prob = tf.placeholder(tf.float32)  """# DROP-OUT here

x = tf.placeholder(tf.float32, [None, train_sub.shape[1]])

w = tf.Variable( tf.random_normal( [train_sub.shape[1], first_layer] ) )
b = tf.Variable( tf.random_normal([first_layer]))
y = tf.matmul(x, w) + b

"""h = tf.nn.relu(y)

drop_out = tf.nn.dropout(h, keep_prob)

w1 = tf.Variable( tf.random_normal( [first_layer, 1] ) )
b1 = tf.Variable( tf.random_normal([1]))
y1 = tf.matmul(drop_out, w1) + b1"""

y_ = tf.placeholder(tf.float32, [None, 1])
#regularizer = tf.nn.l2_loss(w) + tf.nn.l2_loss(w1)

mse_loss = tf.reduce_mean(tf.square( tf.log(y) - tf.log(y_) ))
#mse_loss = mse_loss + regularizer * beta

train_step = tf.train.AdamOptimizer(0.2).minimize(mse_loss)

def random_batch(batch_size):
    rand = random.randint(0, len(train_sub)-1)
    t = train_sub[rand]
    r = train_sub_result[rand]
    for _ in range(batch_size-1):
        rand = random.randint(0, len(train_sub)-1)
        t = np.vstack([t, train_sub[rand]])
        r = np.vstack([r, train_sub_result[rand]])
    return t, r

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

zt_=[]
zv_=[]

for _ in range(4000):
    batch_xs, batch_ys = random_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    a = sess.run(mse_loss, feed_dict={x: validate, y_: validate_result})
    if(np.isnan(w.eval().reshape(-1)).sum() > 100 or a!=a):
    #if(np.isnan(w.eval().reshape(-1)).sum() > 100 or np.isnan(w1.eval().reshape(-1)).sum() > 100 or a!=a):
        print("Reset weights")
        w = tf.Variable( tf.random_normal( [train_sub.shape[1], first_layer] ) )
        #w1 = tf.Variable( tf.random_normal( [first_layer, 1]))
        tf.global_variables_initializer().run()
    
    if( _ % 1000 == 0):
        print("Epoch" + str(_) )
        ww = sess.run(w, feed_dict={x: train_sub, y_: train_sub_result})
        #ww1 = sess.run(w1, feed_dict={x: train_sub, y_: train_sub_result, keep_prob: 1.0})
        #print(sess.run(y_, feed_dict={x: train_sub, y_: train_result}))
        
        #print(sess.run(regularizer * beta, feed_dict ={x: validate, y_: validate_result, keep_prob: 1.0}))
        print(sess.run(mse_loss, feed_dict={x: validate, y_: validate_result}))
        #print(sess.run(y, feed_dict={x: train, y_: train_result}))
        
        zt = sess.run(mse_loss, feed_dict={x: train, y_: train_result})
        zv = sess.run(mse_loss, feed_dict={x: validate, y_: validate_result})
        zt_.append(zt)
        zv_.append(zv)
        
        tffeature = sess.run(y, feed_dict={x: train})
        tfresult = sess.run(y, feed_dict={x: test} )
        tfresult = tfresult.reshape(-1)
    
sess.close()


x_list = range(100)
#plt.plot(x_list, zt_)
#plt.plot(x_list, zv_)
#plt.show()

diff_ = [None] * 100
x_list = range(100)
for x in x_list:
    diff_[x] = zt_[x] - zv_[x]
    
plt.plot(x_list, diff_)
"""
------------------------------------------------ SKLEARN -------------------------------
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
"""
lasso = make_pipeline(RobustScaler(), Lasso(alpha =1, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, train_result.reshape(-1), scoring="neg_mean_squared_log_error", cv = kf))
    return(rmse)

class AverageModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, x, y):
        self.models_ = [clone(z) for z in self.models]
        
        for model in self.models_:
            model.fit(x, y)
            
        return self
    
    def predict(self, x):
        predictions = np.column_stack([model.predict(x) for model in self.models_])
        return np.mean(predictions, axis = 1)
    
average_model = AverageModel(models = (GBoost, model_xgb, model_lgb))
average_model.fit(train, train_result)
score = rmsle_cv(average_model)
print("Average Score: {:f} with std of {:f}".format(score.mean(), score.std()))

skfeature = average_model.predict(train).reshape(-1, 1)
skresult = average_model.predict(test)
"""lasso.fit(train, train_result)
ENet.fit(train, train_result)
GBoost.fit(train, train_result)
meta_train = np.vstack([lasso.predict(train), ENet.predict(train), GBoost.predict(train)]).T
meta_validate = meta_train[1200:]
meta_train_sub = meta_train[:1200]

def meta_random_batch(batch_size):
    rand = random.randint(0, len(meta_train_sub)-1)
    t = meta_train_sub[rand]
    r = train_result[rand]
    for _ in range(batch_size-1):
        rand = random.randint(0, len(meta_train_sub)-1)
        t = np.vstack([t, meta_train_sub[rand]])
        r = np.vstack([r, train_result[rand]])
    return t, r

x = tf.placeholder(tf.float32, [None, meta_train.shape[1]])
w = tf.Variable( tf.random_normal([meta_train.shape[1], 1]) )
b = tf.Variable( tf.random_normal([1]))
y = tf.matmul(x, w) + b
y_ = tf.placeholder(tf.float32, [None, 1])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#z = tf.scalar_mul(weight[0], models[0].predict(x))
#y = tf.scalar_mul(weight[0], models[0].predict(x)) + tf.scalar.mul(weight[1], models[1].predict(x)) + tf.scalar.mul(weight[2], models[2].predict(x))
overall_loss = tf.reduce_mean( tf.square( tf.log(y) - tf.log(y_) ) )

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(overall_loss)

zt_=[]
zv_=[]

for _ in range(10000):
    batch_xs, batch_ys = meta_random_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    a = sess.run(overall_loss, feed_dict={x: batch_xs, y_: batch_ys})
    if(np.isnan(w.eval().reshape(-1)).sum() > 100 or a!=a):
    #if(np.isnan(w.eval().reshape(-1)).sum() > 100 or np.isnan(w1.eval().reshape(-1)).sum() > 100 or a!=a):
        print("Reset weights")
        w = tf.Variable( tf.random_normal( [meta_train.shape[1], 1] ) )
        tf.global_variables_initializer().run()
    
    if( _ % 1000 == 0):
        print("Epoch" + str(_) )
        zt = sess.run(overall_loss, feed_dict={x: meta_train, y_: train_result})
        zv = sess.run(overall_loss, feed_dict={x: meta_validate, y_: validate_result})
        
        zt_.append(zt)
        zv_.append(zv)
        
        print(sess.run(overall_loss, feed_dict={x: meta_validate, y_: validate_result}))
        #print(sess.run(b, feed_dict={x: batch_xs, y_: batch_ys}))
        #print(sess.run(w, feed_dict={x: batch_xs, y_: batch_ys}))
        #print(sess.run(y, feed_dict={x: batch_xs, y_: batch_ys}))
        #print(sess.run(y_, feed_dict={x: batch_xs, y_: batch_ys}))
    
    #result = sess.run(y, feed_dict={x: meta_test} )
    #result = tfresult.reshape(-1)
        
sess.close()"""
#Meta Model on top of TF + Average_XGL
#meta_train = np.hstack([tffeature, skfeature])
#meta_test = np.hstack([tfresult.reshape(-1, 1), skresult.reshape(-1,1)])

#lasso.fit(meta_train, train_result)
#score = rmsle_cv(lasso)
#print("Lasso Score: {:f} with std of {:f}".format(score.mean(), score.std()))

#npresult = lasso.predict(meta_test)
npresult = (tfresult + skresult) /2
nparange = np.arange(1461, 2920)
pd.DataFrame({'Id': nparange, 'SalePrice': result}).to_csv('2018-07-12.csv', index =False)
"""
tz = pd.read_csv("../input/train.csv")
#tzz = tz.melt()
#tzzz = pd.crosstab(index=tzz['value'], columns = tzz['variable'])
distribution = tz.apply(pd.Series.value_counts).sum()
distribution.plot(x='soe', y='SalePrice', ylim=(0,1500))

var = 'BedroomAbvGr'
data = pd.concat([training['SalePrice'], training[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
"""