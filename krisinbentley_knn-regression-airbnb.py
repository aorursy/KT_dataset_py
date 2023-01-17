import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# read data

df=pd.read_csv('../input/airbnb.csv')

# go over all variables

#print(df.columns.values)

# 根据个人经验选择可能有用的variables,drop columns such as url...

v_pool=['city','state','accommodates','latitude','longitude','room_type',\

        'bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','cancellation_policy',\

        'number_of_reviews','reviews_per_month','review_scores_rating']

df=df[v_pool]

df.head()
# data cleaning - transform price(str) into price(float)

df['price']=df['price'].str.replace('$','')

df['price']=pd.to_numeric(df['price'],errors='coerce')
def knnreg(c,feature_col,k=5):

    data=df.copy()

    data=data.dropna()

    data['distance']=np.abs(df[feature_col]-c)

    data=data.sort_values('distance')

    knn=data.iloc[:k]

    price_predict=knn['price'].mean()

    return price_predict

k10=knnreg(c=3,feature_col='bedrooms',k=10)

k5=knnreg(c=6,feature_col='bedrooms',k=5)

print('the number of the bedroom of my house is 3, K=10,在只考虑房间数量的情况下，the predicted price for my hosue is $%.2f' %k10)

print('the number of the bedroom of my house is 6, K=5,在只考虑房间数量的情况下，the predicted price for my hosue is $%.2f' %k5)
#考虑其他连续单变量

for col in ['accommodates','bedrooms','bathrooms','number_of_reviews']:

    data=df[:100].copy()

    data['predicted_price'] = data[col].apply(knnreg,feature_col=col,k=5)

    data['squared_error']=(data['price']-data['predicted_price'])**2

    mse=data['squared_error'].mean()

    rmse=mse**0.5

    print("RMSE for the {} column: {}".format(col,rmse))
from sklearn import preprocessing #https://scikit-learn.org/stable/

cols = ['accommodates','bedrooms','bathrooms','beds','latitude','longitude','review_scores_rating','price']

data=df[cols].copy()

data=data.dropna()



# 把data打乱，再分成3组

data_shuffle=data.sample(frac=1, random_state=0)

n1=int(len(data)*(0.8))

n2=int(len(data)*(0.9))



data_train=data_shuffle[:n1]

data_test1=data_shuffle[n1:n2]

data_test2=data_shuffle[n2:]



# 对training data, test1, test2标准化处理

scaler = preprocessing.StandardScaler().fit(data_train)



data_norm_train=pd.DataFrame(scaler.transform(data_train),

                             columns=cols)

data_norm_test1=pd.DataFrame(scaler.transform(data_test1),

                             columns=cols)

data_norm_test2=pd.DataFrame(scaler.transform(data_test2),

                             columns=cols)
from sklearn.neighbors import KNeighborsRegressor

factors=['accommodates','bedrooms','bathrooms','beds','latitude','longitude','review_scores_rating']

target=['price']

knn_reg=KNeighborsRegressor(n_neighbors=5)            

knn_reg.fit(data_norm_train[factors],data_norm_train[target]) #将data_norm_train作为样本



data_norm_train['predict_price']=knn_reg.predict(data_norm_train[factors]) #把预测的price存储在新column里面

data_norm_test2['predict_price']=knn_reg.predict(data_norm_test2[factors])

data_norm_test1['predict_price']=knn_reg.predict(data_norm_test1[factors])



from sklearn.metrics import mean_squared_error

rmse1=mean_squared_error(data_norm_train[target],data_norm_train['predict_price'])**0.5

print('RMSE of the training data: %.2f' %rmse1)

rmse2=mean_squared_error(data_norm_test1[target],data_norm_test1['predict_price'])**0.5

print('RMSE of the test data1: %.2f' %rmse2)

rmse3=mean_squared_error(data_norm_test2[target],data_norm_test2['predict_price'])**0.5

print('RMSE of the test data2: %.2f' %rmse3) 
#测试不同的k，查看RMSE





    

lst_rmse_train=[]

lst_rmse_test1=[]

lst_rmse_test2=[]



for i in range(2,40):

    knn_reg=KNeighborsRegressor(n_neighbors=i)

    knn_reg.fit(data_norm_train[factors],data_norm_train[target])

    data_norm_train['predict_price']=knn_reg.predict(data_norm_train[factors]) #把预测的price存储在新column里面

    data_norm_test1['predict_price']=knn_reg.predict(data_norm_test1[factors])

    data_norm_test2['predict_price']=knn_reg.predict(data_norm_test2[factors])

    

    rmse1=mean_squared_error(data_norm_train[target],data_norm_train['predict_price'])**0.5

    rmse2=mean_squared_error(data_norm_test1[target],data_norm_test1['predict_price'])**0.5

    rmse3=mean_squared_error(data_norm_test2[target],data_norm_test2['predict_price'])**0.5

    lst_rmse_train.append(rmse1)

    lst_rmse_test1.append(rmse2)

    lst_rmse_test2.append(rmse3)

import matplotlib.style as psl

psl.use('bmh')

plt.plot(lst_rmse_train,

        linestyle='-',

        marker='d')   

plt.plot(lst_rmse_test1,

        linestyle='--',

        marker='x',alpha=0.4)

plt.plot(lst_rmse_test2,

        linestyle='--',

        marker='x',

        alpha=1)

plt.title("how RMSE changes with k")

plt.xlabel("K")

plt.ylabel("RMSE")
#假设我的房源信息已知，预测租金

my=np.array([3,2,2.5,3,38.953960,-77.028639,100,9999]).reshape(1,-1) #array.reshape(1, -1) if it contains a single sample,租金母鸡，所以写9999

a=pd.DataFrame(scaler.transform(my),columns=cols)

a=a.drop('price',axis=1)

predicted_price_scaled=knn_reg.predict(a)



mu=data_train['price'].describe()['mean']

std=data_train['price'].describe()['std']

predicted_price_dollar=predicted_price_scaled*std+mu

print('\n predict my house rent price, here are my house attributes:')

for i in range(len(factors)):

    print('    ',factors[i],':',my[0][i])

print('则标准化的预测的租金为：',float(predicted_price_dollar) )
