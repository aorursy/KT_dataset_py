import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
data = pd.read_csv(r"D:\housing_price_prediction\train.csv")
data.head(5)
data.shape
data.info()
data1=data.fillna('0')
data1.info()
plt.hist(data1['median_house_value'],color='blue')
plt.show()
data1=data1[data1['median_house_value']<500000]
data1.info()
co=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']


X=data1[co]
Y=data1.median_house_value
x_train,x_test,y_train, y_test = train_test_split(X,Y,random_state=0,test_size=0.20)
print('Training Features Shape:', x_train.shape)
print('Training Target Shape:', y_train.shape)
print('Testing Features Shape:',x_test.shape )
print('Testing Target Shape:',y_test.shape)

ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss_y.transform(y_test.values.reshape(-1, 1))
# 两种k近邻回归行学习和预测
# 初始化k近邻回归模型 使用平均回归进行预测
uni_knr = KNeighborsRegressor(weights="uniform")
# 训练
uni_knr.fit(x_train, y_train)
# 预测保存预测结果
uni_knr_y_predict = uni_knr.predict(x_test)
uni_knr_y_predict
# 多初始化k近邻回归模型 使用距离加权回归
dis_knr = KNeighborsRegressor(weights="distance")
# 训练
dis_knr.fit(x_train, y_train)
# 预测 保存预测结果
dis_knr_y_predict = dis_knr.predict(x_test)
dis_knr_y_predict
# 5 模型评估

# 平均k近邻回归 模型评估

print("平均k近邻回归的默认评估值为：", uni_knr.score(x_test, y_test))

print("平均k近邻回归的R_squared值为：", r2_score(y_test, uni_knr_y_predict))

print("平均k近邻回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),

                                           ss_y.inverse_transform(uni_knr_y_predict)))

print("平均k近邻回归 的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),

                                               ss_y.inverse_transform(uni_knr_y_predict)))

# 距离加权k近邻回归 模型评估

print("距离加权k近邻回归的默认评估值为：", dis_knr.score(x_test, y_test))

print("距离加权k近邻回归的R_squared值为：", r2_score(y_test, dis_knr_y_predict))

print("距离加权k近邻回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),

                                             ss_y.inverse_transform(dis_knr_y_predict)))

print("距离加权k近邻回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),

                                                ss_y.inverse_transform(dis_knr_y_predict)))


