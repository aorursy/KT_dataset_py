import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data= pd.read_csv("../input/kc_house_data.csv")
data.head()
data.dtypes
features= data[['bedrooms','bathrooms','sqft_living','floors','condition','grade']]
labels = data['price']
features.head()
X_train,X_vali,y_train,y_vali = train_test_split(features,
                                                 labels,
                                                 test_size=1/3,
                                                 random_state=0)


plt.scatter(data['sqft_living'],labels)
data['sqft_living'].hist()
plt.show()
model = LinearRegression()
model.fit(X_train,y_train)
pd.DataFrame(list(zip(features.columns, model.coef_)))

model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，condition,grade预测房价
model.predict([[3,2,2500,2,5,7]])
pred= model.predict(X_train)
((pred-y_train)**2).sum()/len(y_train)
abs((pred-y_train)/y_train).sum()/len(y_train)
pred_vali= model.predict(X_vali)
((pred_vali-y_vali)**2).sum()/len(y_vali)
abs((pred_vali-y_vali)/y_vali).sum()/len(y_vali)

