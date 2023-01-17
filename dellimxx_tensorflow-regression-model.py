import IPython

#import graphlab as gl

import pandas as pd

import numpy as np

from mpl_toolkits.basemap import Basemap 

import matplotlib.pyplot as plt

# I usually use Graphlab but this kernel i can't use graphlab so i merely write code

data = pd.read_csv("../input/autos.csv",encoding='latin-1')

import os

os.getcwd()
data # 현재 총 데이터 수 37,1528
data_dropna = data.dropna()

len(data_dropna)

# 우선적으로 Null이 사이사이에 들어가 있기 때문에 Null을 일단을 제외시킴.

# 그 이유: 이 차가 어떤 브랜드인지, 어떤 타입인지 알 수 가 없다. 그래서 제외시키는 것이 맞다고 판단.
data_dropna # NUll을 제외시킨 데이터 총 26,0956
data_dropna['vehicleType']= np.where(data_dropna['vehicleType']=='kleinwagen','compactcar',data_dropna['vehicleType'])

data_dropna['gearbox']=np.where(data_dropna['gearbox']=='manuell','manual','automatic')

data_dropna['notRepairedDamage']= np.where(data_dropna['notRepairedDamage']=='nein','no','yes')

data_dropna['fuelType']=np.where(data_dropna['fuelType']=='benzin','gasoline',np.where(data_dropna['fuelType']=='elektro','electronic',data_dropna['fuelType']))

# german-> english
# 일단 중고차를 살때 중요한것은 그차가 얼마나 됬는지가 중요하다.

# 이 데이터에서는 언제 차가 등록이 되었는지만 나와있지 차가 얼마나오래 되었는지는 나와있지 않다.

# 뿐만 아니라 가격과 상관관계를 같는 컬럼이 존재 하지 않는다. 그래서 차의 연식으로 파악.

data_dropna = data_dropna.drop(data_dropna[data_dropna['monthOfRegistration']==0].index)

data_dropna=data_dropna.drop(data_dropna[data_dropna["yearOfRegistration"]<1950].index)

data_dropna=data_dropna.drop(data_dropna[data_dropna["yearOfRegistration"]>2016].index)

# 분포상 2016년도 데이터에 2017,2018년이 있는 것은 다 자름.. 그리고 month는 1-12 0을 제외

data_dropna["lastSeen"]=pd.to_datetime(data_dropna["lastSeen"])

data_dropna["dateCreated"]=pd.to_datetime(data_dropna["dateCreated"])

#Extracting the date from datetime

data_dropna["lastSeenDate"]=data_dropna["lastSeen"].dt.date

data_dropna["dateCreatedDate"]=data_dropna["dateCreated"].dt.date #시간을 자름.

data_dropna["registerMonthYear"] = data_dropna["yearOfRegistration"].astype(str).str.cat(data_dropna["monthOfRegistration"].astype(str), sep='-')

# the date for which the ad at ebay was created

data_dropna["registerDate"]=data_dropna["registerMonthYear"].astype(str)+'-01'

data_dropna["registerDate"]=pd.to_datetime(data_dropna["registerDate"])

data_dropna['registerDate']=data_dropna['registerDate'].dt.date

data_dropna["carAge"]=data_dropna["lastSeenDate"]-data_dropna["registerDate"]

#days가 나오는 것을 제외 시킨다.

data_dropna["carAge"]=data_dropna["carAge"].astype(str)

data_dropna["carAge"]=data_dropna.carAge.str.split(n=1).str.get(0)

data_dropna["carAge"]=data_dropna["carAge"].astype(str).astype(int)

# 데이터 전처리 일부 완료

data_dropna = data_dropna[data_dropna['price']>1000]

data_dropna= data_dropna[data_dropna['price']<=38000]
data_dropna #253309 
car_data = data_dropna.drop('nrOfPictures',axis=1)

# 우선적으로 axis=0이 기본(행_ 그래서 컬럼을 지울때는 axis=1로 줘서 삭제)
from sklearn import preprocessing, svm

# sklearn이라는것은 파이썬 머신러닝 모듈이다. 

# 라벨이 너무 짧거나 길면 작동이 잘안된다.

from sklearn.preprocessing import StandardScaler, Normalizer

car_colum  = list(car_data.columns)

labels = ['name', 'gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']

les= {}

for l in labels:

    les[l] = preprocessing.LabelEncoder()

    # 중복된것을 제거 및 정렬 시켜준다.

    les[l].fit(car_data[l])

    # 다시 fit 시켜준다.

    tr = les[l].transform(car_data[l]) 

    # >>> le.transform(["tokyo", "tokyo", "paris"]) array([2, 2, 1]...)

    car_data.loc[:, l + '_label'] = pd.Series(tr, index=car_data.index)



labeled = car_data[car_colum+[x+'_label' for x in labels]]
# correlation 그래프를 그려서 시각화 시키기

import numpy as np

# seaborn이 statistical data visualization이다.

def correlation_graph(a):

    corr = a.corr()

    names = car_colum

    plt.figure(figsize=(10, 10))

    plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')

    plt.colorbar()

    plt.xticks(range(len(corr)), corr.columns, rotation='vertical')

    plt.yticks(range(len(corr)), corr.columns);

    plt.suptitle('Stock Correlations Heat Map', fontsize=15, fontweight='bold')

    plt.show()

labeled.corr()
%matplotlib inline

correlation_graph(labeled)
labeled.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

# 어느 것이 제일 연관성이 있는지 따져봄.
data_dropna_final=data_dropna[['yearOfRegistration','carAge','kilometer','gearbox','powerPS','fuelType','brand','notRepairedDamage','price']]
data_dropna_final.to_csv('data_drop_final.csv')
"""# string _. float because placefolder is float32

data_dropna_final['gearbox'] = data_dropna_final['gearbox'].replace("'","")

data_dropna_final['gearbox']=[float(a) for a in data_dropna_final['gearbox']]

data_dropna_final['fuelType'] = data_dropna_final.replace("'","")

data_dropna_final['fuelType']=[float(a) for a in data_dropna_final['fuelType']]

data_dropna_final['brand'] = data_dropna_final.replace("'","")

data_dropna_final['brand']=[float(a) for a in data_dropna_final['brand']]

data_dropna_final['notRepairedDamage'] = data_dropna_final.replace("'","")

data_dropna_final['notRepairedDamage']=[float(a) for a in data_dropna_final['notRepairedDamage']]"""
import tensorflow as tf

import csv

import codecs



data = codecs.open('data_drop_final.csv','r',encoding='1252')

xy = csv.reader(data)

xy = list(xy)

header = xy[0]

value = xy[1:]

data = {a:b for a,b in zip(header,zip(*value))}

header1 = header[:-1]

print (header1)

data_Final = {}

for i in header:

    ii = np.array(data[i])

    data_Final.update({i:ii})

xy_data = []

for ii in header1:

    xy_data.append(data_Final[i])

xy_data = np.array(xy_data)

x_data = np.transpose(xy_data)

x_train_data = x_data[:168600,:]

x_test_data = x_data[168600:,:]

y_data = [data_Final['price']]

y_data = np.transpose(np.array(y_data))

y_train_data = y_data[:168600,:]

y_test_data = y_data[168600:,:]



print (x_test_data.shape,y_test_data.shape)

print (x_data.shape,y_data.shape)



X = tf.placeholder(tf.float32, shape=[None, 9])

Y = tf.placeholder(tf.float32, shape=[None, 1])



with tf.name_scope('layer1'):

    w = tf.Variable(tf.random_normal([9,1]))

    b = tf.Variable(tf.random_normal([1]))

    hypothesis = tf.matmul(X,w)+b



    

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    for step in range(2001):

        cost_val, hy_val,_= sess.run(

            [cost, hypothesis, optimizer], feed_dict={X: x_train_data, Y: y_train_data})

        if step % 400 == 0:

            print(step, "Cost: ", cost_val)



    print(sess.run(hypothesis, feed_dict={X: x_test_data[:3], Y: y_test_data[:3]}))













# Tensorflow predict price Vs reality pr|ice

data_dropna_final[168600:168603]