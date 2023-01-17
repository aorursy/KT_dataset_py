import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
import datetime
import time
data = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
data.head()
X = data[['weather','windspeed','temp']]
Y= data['count']
time= np.array(data['datetime'])
hour=np.array(data['datetime'])
month=np.array(data['datetime'])
weekday=np.array(data['datetime'])
for i in range(time.size):    
    hour[i]=int(time[i][-8:-6])
    month[i]=int(time[i][5:7])
    
    nn= time[i][:10].split("-")  
    weekday[i]=datetime.datetime(int(nn[0]),int(nn[1]),int(nn[2])).weekday()
    

weekday[:35]
hour[:24]
month[:24]
X.insert(0,"hour",hour)
X.insert(0,"weekday",weekday)
X.insert(0,"month",month)
X.head()
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
print(type(xtrain))

ytrain.head()
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
pred = model.predict(xtrain)
((pred-ytrain)*(pred-ytrain)).sum() / len(ytrain)
(abs(pred-ytrain)/ytrain).sum() / len(ytrain)

pred
test_data = test[['weather','windspeed','temp']]
test_data.head()
time1= np.array(test['datetime'])
hour1=np.array(test['datetime'])
month1=np.array(test['datetime'])
weekday1=np.array(test['datetime'])
for i in range(time1.size):    
    hour1[i]=int(time1[i][-8:-6])
    month1[i]=int(time1[i][5:7])
    
    nn1= time1[i][:10].split("-")  
    weekday1[i]=datetime.datetime(int(nn1[0]),int(nn1[1]),int(nn1[2])).weekday()
    
time1.shape
weekday1.size
test_data.insert(0,"hour",hour1)
test_data.insert(0,"weekday",weekday1)
test_data.insert(0,"month",month1)
test_data.head()
prediction = model.predict(test_data)
prediction
for i in range(len(prediction)):
    if prediction[i]<0:
        prediction[i]=0
prediction[1]
        
prediction
df = pd.DataFrame({"datetime": test['datetime'],"count": prediction})
df.to_csv('./submission3.csv',index = False)




