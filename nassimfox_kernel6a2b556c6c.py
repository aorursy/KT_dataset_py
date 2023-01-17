

import numpy as np

import pandas as pd 

from matplotlib import pyplot as p

import seaborn as sns

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape
train.isna().sum()
s = train.isna().sum()

s = s.sort_values(ascending=False)



s_dataframe = pd.DataFrame()



for i in s.index : 

    if s[i]>0 : 

        dataframe = pd.DataFrame([s[i]],[i])

        s_dataframe = s_dataframe.append(dataframe)

        

s_dataframe

    
train = train.drop(["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"],axis=1)

test = test.drop(["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"],axis=1)
train.info()
data = [train,test]

for dataset in data : 

    cols = dataset.select_dtypes(include="object").columns

    for col in cols : 

         dataset[col] = dataset[col].astype("category").cat.codes+1

    
train.head()
for i in train.columns : 

    if train[i].isna().sum() > 0 : 

        print(i)

        print(train[i].isna().sum())



for dataset in data : 

    dataset["LotFrontage"].fillna(dataset.groupby("Neighborhood")["LotFrontage"].transform("median"),inplace=True)

    dataset["MasVnrArea"].fillna(dataset.groupby("Neighborhood")["MasVnrArea"].transform("median"),inplace=True)

    #fill with must commun year

    dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].value_counts().index[0],inplace=True)
for i in test.columns : 

    if test[i].isna().sum() > 0 : 

        print(i)

        print(test[i].isna().sum())

        test[i].fillna(0,inplace=True)
id_test = test["Id"]



train = train.drop("Id",axis=1)

test = test.drop("Id",axis=1)

train


from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



x = train.drop("SalePrice",axis=1).copy()

y = train["SalePrice"]



features = SelectKBest(score_func=chi2,k=20)

prediction = features.fit(x,y)

dataframe_col = pd.DataFrame(x.columns)

dataframe_score = pd.DataFrame(prediction.scores_)



dataframe = pd.concat([dataframe_col,dataframe_score],axis=1)



dataframe.columns = ['Index','Scores']

dataframe.nsmallest(40,'Scores')
# small = 1 , medium = 2 , large = 3 , very large = 4 



#train.loc[train["LotArea"]<=5000,"LotArea"] = 1 

#train.loc[(train["LotArea"]>5000)&(train["LotArea"]<=10000),"LotArea"] = 2 

#train.loc[(train["LotArea"]>10000)&(train["LotArea"]<=20000),"LotArea"] = 3 

#train.loc[train["LotArea"]>20000,"LotArea"] = 4

# small = 1 , medium = 2 , large = 3 , very large = 4 



#train.loc[train["BsmtFinSF1"]<=400,"BsmtFinSF1"] = 1 

#train.loc[(train["BsmtFinSF1"]>400)&(train["BsmtFinSF1"]<=800),"BsmtFinSF1"] = 2 

#train.loc[(train["BsmtFinSF1"]>800)&(train["BsmtFinSF1"]<=1200),"BsmtFinSF1"] = 3 

#train.loc[(train["BsmtFinSF1"]>1200)&(train["BsmtFinSF1"]<=1600),"BsmtFinSF1"] = 4

#train.loc[train["BsmtFinSF1"]>1600,"BsmtFinSF1"] = 5

#train = train.drop("BsmtFinSF2",axis=1)

train = train.drop("LowQualFinSF",axis=1)

train = train.drop("PoolArea",axis=1)

#train = train.drop("2ndFlrSF",axis=1)

#train = train.drop("MasVnrArea",axis=1)

train = train.drop("MiscVal",axis=1)

#train = train.drop("BsmtUnfSF",axis=1)



#test = test.drop("BsmtFinSF2",axis=1)

test = test.drop("LowQualFinSF",axis=1)

test = test.drop("PoolArea",axis=1)

#test = test.drop("2ndFlrSF",axis=1)

#test = test.drop("MasVnrArea",axis=1)

test = test.drop("MiscVal",axis=1)

#test = test.drop("BsmtUnfSF",axis=1)

#train.SalePrice = np.log(train.SalePrice)

train.head()


from scipy import stats 



f , ( a1 , a2 ) = p.subplots(1,2,figsize=(18,5))

p.sca(a1)

p.hist(train.SalePrice)

p.sca(a2)

sns.distplot(train.SalePrice,fit=stats.norm)


f , ( a1 , a2 ) = p.subplots(1,2,figsize=(18,5))

p.sca(a1)

p.hist(np.log(train.SalePrice))

p.sca(a2)

sns.distplot(np.log(train.SalePrice),fit=stats.norm)


for col in dataframe.nsmallest(28,"Scores").Index : 

        train = train.drop(col,axis=1)

        test = test.drop(col,axis=1)



train.head()

        
m = train.corr()

#p.imshow(m,cmap="hot",interpolation="nearest")

f, ax = p.subplots(figsize=(12, 9))

sns.heatmap(m, vmax=.8, square=True)

p.yticks(rotation=0)

p.xticks(rotation=90)

cols = m.nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,

                 xticklabels=cols.values)

p.yticks(rotation=0)

p.xticks(rotation=90)
f , ( ax1 , ax2 , ax3 ) = p.subplots(1,3,figsize=(20,5))

dataframe1 = pd.concat([train["SalePrice"],train["TotalBsmtSF"]],axis=1)

dataframe2 = pd.concat([train["SalePrice"],train["1stFlrSF"]],axis=1)

dataframe3 = pd.concat([train["SalePrice"],train["2ndFlrSF"]],axis=1)



dataframe1.plot.scatter(x="TotalBsmtSF",y="SalePrice",ylim=(0, 800000), ax=ax1)

dataframe2.plot.scatter(x="1stFlrSF",y="SalePrice",ylim=(0, 800000), ax=ax2)

dataframe3.plot.scatter(x="2ndFlrSF",y="SalePrice",ylim=(0, 800000), ax=ax3)

train = train.drop(["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"],axis=1)

train["TotalSF"] = train["TotalBsmtSF"]+train["1stFlrSF"]+train["2ndFlrSF"]

train = train.drop(["TotalBsmtSF","1stFlrSF","2ndFlrSF"],axis=1)

train = train.drop("GarageArea",axis=1)



test = test.drop(["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"],axis=1)

test["TotalSF"] = test["TotalBsmtSF"]+test["1stFlrSF"]+test["2ndFlrSF"]

test = test.drop(["TotalBsmtSF","1stFlrSF","2ndFlrSF"],axis=1)

test = test.drop("GarageArea",axis=1)
dataframe = pd.concat([train["SalePrice"],train["TotalSF"]],axis=1)

dataframe.plot.scatter(x="TotalSF",y="SalePrice")
from matplotlib import pyplot as p 



def plot_precis(y,prediction,i) :

    f1 = p.figure(i)

    p.plot(y,y)

    p.scatter(y,prediction)

def plot_prediction(y,prediction,i) : 

    s = np.arange(0,y.shape[0])



    f3 = p.figure(i)



    f3 , (a1 , a2 ) = p.subplots(1,2,figsize=(10,5))

    p.sca(a1)

    p.plot(s,y)

    p.plot(s,prediction)

    p.sca(a2)

    p.plot(s,np.sort(y))

    p.plot(s,np.sort(prediction))


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



regression = RandomForestRegressor()



x = train.drop("SalePrice",axis=1).copy()

x , x_test = train_test_split(x,test_size=0.01)

y = train["SalePrice"]

y , y_test = train_test_split(y,test_size=0.01)



y_train = np.log(y)

y_test_v = np.log(y_test)



regression.fit(x,y_train)



prediction = regression.predict(x)

test_prediction = regression.predict(x_test)



cst = mean_squared_error(y_train,prediction)



print(cst)


plot_precis(y_train,prediction,1)

plot_precis(y_test_v,test_prediction,2)



plot_prediction(y_train,prediction,3)

plot_prediction(y_test_v,test_prediction,4)





from sklearn.linear_model import LinearRegression 



regression = LinearRegression()





regression.fit(x,y_train)



prediction = regression.predict(x)

prediction_test = regression.predict(x_test)

plot_precis(y_train,prediction,1)

plot_precis(y_test_v,prediction_test,2)



plot_prediction(y_train,prediction,3)

plot_prediction(y_test_v,prediction_test,4)

print('R square is: {}'.format(regression.score(x_test, y_test)))


from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler 





regression = MLPRegressor(activation='identity', alpha=1e-05, batch_size='auto',

              #epsilon=1e-08,

              hidden_layer_sizes=(3, 4),

              learning_rate='constant', learning_rate_init=0.0005,

              max_iter=200, 

              solver="sgd",

              #momentum=0.9, n_iter_no_change=10,

              validation_fraction=0.1, verbose=True, warm_start=False)



#x = train.drop("SalePrice",axis=1).copy()

#x , x_test = train_test_split(x,test_size=0.2)

#y = train["SalePrice"]

#y , y_test = train_test_split(y,test_size=0.2)



x_scaled = x.copy()

x_test_scaled = x_test.copy()



for col in x.columns : 

    x_scaled[col] = (x[col]-np.min(x[col])) / (np.max(x[col])-np.min(x[col]))

    x_test_scaled[col] = (x_test[col]-np.min(x_test[col])) / (np.max(x_test[col])-np.min(x_test[col]))

    



y_train = np.log(y)

y_test_v = np.log(y_test)



y_scaled = (y_train-np.min(y_train)) / (np.max(y_train)-np.min(y_train))

#y_test_scaled = (y_test_v-np.min(y_test_v)) / (np.max(y_test_v)-np.min(y_test_v))

y_test_scaled = (y_test_v-np.min(y_train)) / (np.max(y_train)-np.min(y_train))





y_train_nn = y_scaled



regression.fit(x_scaled,y_train_nn)



prediction = regression.predict(x_scaled)

test_prediction = regression.predict(x_test_scaled)



cst = mean_squared_error(y_train_nn,prediction)



print(cst)



plot_precis(y_train_nn,prediction,1)

plot_precis(y_test_scaled,test_prediction,2)



plot_prediction(y_train_nn,prediction,3)

plot_prediction(y_test_scaled,test_prediction,4)





test_scaled = test.copy()



for col in test.columns : 

    test_scaled[col] = (test[col]-np.min(test[col])) / (np.max(test[col])-np.min(test[col]))

    

  

 

subpred = regression.predict(test_scaled)



subpred = subpred * (np.max(y_train)-np.min(y_train)) + np.min(y_train)



subpred = np.exp(subpred)



dataframe_s = pd.DataFrame(subpred)

dataframe = pd.concat([id_test,dataframe_s],axis=1)



dataframe.columns = {"SalePrice","Id"}



dataframe

submission = dataframe.to_csv("submission.csv",index=False)



#subpred.max()
print(np.max(y_train))

print(np.max(y_test))

print(np.max(prediction))
import tensorflow as tf
'''x_scaled = np.asarray(x_scaled)

y_scaled = np.asarray(y_scaled)

x_test_scaled = np.asarray(x_test_scaled)

y_test_scaled = np.asarray(y_test_scaled).reshape(y_test_scaled.shape[0],1)



tf.reset_default_graph()



x_t = tf.placeholder(tf.float32,shape=(None,x_scaled.shape[1]),name="x_t")



y_log = y_scaled #np.log(y)

y_test_log = y_test_scaled #np.log(y_test)



input = x_scaled.shape[1]

layer1_nodes = 10

layer2_nodes = 10

layer3_nodes = 20

layer31_nodes = 20

layer32_nodes = 20

output = 1 



epoch = 1000 

learning_rate = 0.0005



with tf.variable_scope("l1_3",reuse=tf.AUTO_REUSE) : 

    w1 = tf.get_variable("w1",shape=(input,layer1_nodes),initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable("b1",shape=(1,layer1_nodes),initializer=tf.zeros_initializer())

    l1 = tf.nn.relu(tf.matmul(x_t,w1)+b1)

with tf.variable_scope("l2_3",reuse=tf.AUTO_REUSE) : 

    w2 = tf.get_variable("w2",shape=(layer1_nodes,layer2_nodes),initializer=tf.contrib.layers.xavier_initializer())

    b2 = tf.get_variable("b2",shape=(1,layer2_nodes),initializer=tf.zeros_initializer())

    l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)

with tf.variable_scope("l3_3",reuse=tf.AUTO_REUSE) : 

    w3 = tf.get_variable("w3",shape=(layer2_nodes,layer3_nodes),initializer=tf.contrib.layers.xavier_initializer())

    b3 = tf.get_variable("b3",shape=(1,layer3_nodes),initializer=tf.zeros_initializer())

    l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)

with tf.variable_scope("l3.1_3",reuse=tf.AUTO_REUSE) : 

    w31 = tf.get_variable("w3.1",shape=(layer3_nodes,layer31_nodes),initializer=tf.contrib.layers.xavier_initializer())

    b31 = tf.get_variable("b3.1",shape=(1,layer31_nodes),initializer=tf.zeros_initializer())

    l31 = tf.nn.relu(tf.matmul(l3,w31)+b31)

with tf.variable_scope("l3.2_3",reuse=tf.AUTO_REUSE) : 

    w32 = tf.get_variable("w3.2",shape=(layer31_nodes,layer32_nodes),initializer=tf.contrib.layers.xavier_initializer())

    b32 = tf.get_variable("b3.2",shape=(1,layer32_nodes),initializer=tf.zeros_initializer())

    l32 = tf.nn.relu(tf.matmul(l31,w32)+b32)

with tf.variable_scope("output_3",reuse=tf.AUTO_REUSE) : 

    w4 = tf.get_variable("w4",shape=(layer32_nodes,output),initializer=tf.contrib.layers.xavier_initializer())

    b4 = tf.get_variable("b4",shape=(1,output),initializer=tf.zeros_initializer())

    l4 = tf.nn.relu(tf.matmul(l32,w4)+b4)

with tf.variable_scope("cost_3",reuse=tf.AUTO_REUSE) : 

    y_t = tf.placeholder(tf.float32,shape=y.shape,name="y_t")

    cost = tf.reduce_mean(tf.squared_difference(y_t,l4))

with tf.variable_scope("train",reuse=tf.AUTO_REUSE) : 

    trains = tf.train.AdamOptimizer(learning_rate).minimize(cost)



    

init = tf.global_variables_initializer() 



cst = np.zeros([epoch,1])



with tf.Session() as session : 

    

    session.run(init)

    

    for i in range(epoch) : 

        

        session.run(trains,feed_dict={x_t:x_scaled,y_t:y_log})

        

        cst[i] = session.run(cost,feed_dict={x_t:x_scaled,y_t:y_log})

        

        if i % 10 == 0 : 

            print("Epoch {0} cost {1}".format(i,cst[i]))

        

        

    prediction = session.run(l4,feed_dict={x_t:x_scaled})

    

    prediction_test = session.run(l4,feed_dict={x_t:x_test_scaled})     '''

    




'''plot_precis(y_log,prediction,1)

plot_precis(y_test_log,test_prediction,2)





f3 = p.figure(3)

p.plot(cst)





f4 = p.figure(4)



f4 , ( a1 , a2 ) = p.subplots(1,2,figsize=(10,5))



p.sca(a1)



s = np.arange(0,y.shape[0])



p.plot(s,y_log)

p.plot(s,prediction)



p.sca(a2)



p.plot(s,np.sort(y_log))

p.plot(s,np.sort(prediction))

'''
'''prediction_unscaled = prediction_test * (np.max(y_test)-np.min(y_test)) + np.min(y_test)

prediction_unscaled



f2 = p.figure(2)

p.plot(y_test,y_test)

p.scatter(y_test,prediction_unscaled)'''

x_scaled.T[0].min()