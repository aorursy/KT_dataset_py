import tensorflow as tf
 
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# Launch the graph in a session.
sess = tf.Session()
x=sess.run(a)
# Evaluate the tensor `c`.
print("c:",sess.run(c))
print("type of a :",type(a ))
print("type of c :",type(c )) #<class 'tensorflow.python.framework.ops.Tensor'>
print("a:",sess.run(a))
print("x:",x) 
 
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
  # We can also use 'c.eval()' here.
  print(c.eval())
sess.run(tf.Print(c,[c]))
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
c=tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(c,feed_dict={a:10,b:30}))
    print(sess.run(a,feed_dict={a:10,b:30}))
    
import numpy as np
x = tf.placeholder(tf.float32, shape=(2, 2))
y = tf.matmul(x, x)
 
with tf.Session() as sess:
  rand_array = np.random.rand(2, 2)  
  print(sess.run(y, feed_dict={x: rand_array}))
k=np.random.uniform(size=(5, 2))
print(k.shape)
print(k)
#dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2))) 
dataset = tf.data.Dataset.from_tensor_slices(k) 

print(dataset)
print(type(dataset))
#with tf.Session() as sess:
#    print(sess.run(dataset))#Can not convert a TensorSliceDataset into a Tensor or Operation.)

datas=dataset.make_one_shot_iterator().get_next()
sess=tf.Session()
print("sess.run(datas):",sess.run(datas))
import pandas as pd

s=pd.Series([1,2,3,4,5],index=['a','b','c','f','e'])
print(s)
print(s['b'])
print(s.shape)
import numpy as np
df = pd.DataFrame([1, 2, 3, 4, 5], columns=['cols'], index=['a','b','c','d','e'])
print(df)
print("df['cols']: \n",df['cols'])

df2 = pd.DataFrame([[1, 2, 3],[4, 5, 6]], columns=['col1','col2','col3'], index=['a','b'])
print("df2",df2)
print(df2["col3"])
print(df2.index)
print(df2.columns)
df2.index
df2.loc['a']   
import numpy as np
import pandas as pd
import tensorflow as tf
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
feature_values = np.array([[10, 7, 4], [3, 2, 1]])
num_buckets=4
boundaries = np.arange(1.0, num_buckets) / num_buckets
#quantiles = feature_values.quantile(boundaries) 
population={'city':['Beijing','Shanghai','Guangzhou','Shenzhen','Hangzhou','Chongqing'],
            'year':[2016,2017,2016,2017,2016,2016],
            'population':[2100,2300,1000,700,500,500]
            }
print(population)
population=pd.DataFrame(population)   ###
print(population)

cities={'Beijing':55000,'Shanghai':60000,'shenzhen':50000,'Hangzhou':20000,'Guangzhou':45000,'Suzhou':None}
apts=pd.Series(cities,name='income')
print("apts:\n",apts)
df1=pd.DataFrame(apts)
print("df1：\n",df1)
apts['shenzhen']=70000
print("apts:\n",apts)
less_than_50000=(apts<50000)
apts[less_than_50000]=40000
print("less_than_50000 apts:\n",apts)
apts2=pd.Series({'Beijing':10000,'Shanghai':8000,'shenzhen':6000,'Tianjin':40000,'Guangzhou':7000,'Chongqing':30000})
print("apts2 :\n",apts2)
apts=apts+apts2
apts[apts.isnull()]=apts.mean()
print("apts :\n",apts)

df=pd.DataFrame({'apts':apts,'apts2':apts2})   ###
print(df)
feature_values=df
num_buckets=4
boundaries = np.arange(0.0, num_buckets) / num_buckets
print("boundaries :\n",boundaries )
quantiles = feature_values.quantile(boundaries) 
returns=[quantiles[q] for q in quantiles.keys()]
print("returns:\n",returns)
print("quantiles:\n",quantiles)
df.describe()
df.head()
import pandas as pd
cities={'Beijing':55000,'Shanghai':60000,'shenzhen':50000,'Hangzhou':20000,'Guangzhou':45000,'Suzhou':None}
apts=pd.Series(cities,name='income')
apts['shenzhen']=70000
less_than_50000=(apts<50000)
apts[less_than_50000]=40000
apts2=pd.Series({'Beijing':10000,'Shanghai':8000,'shenzhen':6000,'Tianjin':40000,'Guangzhou':7000,'Chongqing':30000})
apts=apts+apts2
apts[apts.isnull()]=apts.mean()
df=pd.DataFrame({'apts':apts,'apts2':apts2})
df['bonus']=2000  
df['income']=df['apts']*2+df['apts2']*1.5+df['bonus']
print("df:\n",df)
df.to_csv('df.csv')
df.to_csv('df2.csv',index=False) #去掉第一列,行索引列

import os
df2_site = r"C:\Users\jinlong\df2.csv"
df_site = r"C:\Users\jinlong\df.csv"
pwd = os.getcwd()  #获取当前工作目录
os.chdir(os.path.dirname(df2_site))
tmp_df = pd.read_csv(os.path.basename(df2_site))   ###
print("tmp_df2:\n",tmp_df)
os.chdir(os.path.dirname(df_site))
tmp_df = pd.read_csv(os.path.basename(df_site))   ###
print("tmp_df1:\n",tmp_df)
tmp_df_index = pd.Index(['Beijing','Shanghai',"Suzhou",'Hangzhou','Tianjin','Chongqing','Nanjing','Shenzhen'])
tmp_df.index=tmp_df_index   #修改索引
print("tmp_df3:\n",tmp_df)
df.to_csv('df3.csv',sep='\t')
new_graph = tf.Graph()
with new_graph.as_default():
    new_g_const = tf.constant([1., 2.])
default_g = tf.get_default_graph()

