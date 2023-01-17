import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data = pd.read_csv("../input/kc_house_data.csv")
data.head()
data.dtypes
# 如果选择多列，必须要用放在[]
X = data[['bedrooms','bathrooms','sqft_living','floors']]
Y = data['price']
# random_state=0 这个是随机数种子，如果这个值是一样的，那么每次随机分出来的结果也是一样的
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
# 不做如下的转换 也可以
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
xtrain
# 表示sqft_living 与 Y  相关
plt.scatter(X['sqft_living'], Y)
plt.show()
# 表示sqft_living 与  floors 不  相关
plt.scatter(X['sqft_living'], X['floors'])
plt.show()
# 1m^2 ~ 11 sqft
X['sqft_living'].hist()
plt.show()
model=LinearRegression()
model.fit(xtrain, ytrain)
model.coef_
# np.transpose(model.coef_)  矩阵转置 i.e 行向量 转化为列向量
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
# -51118.881973 意思是 在bathrooms, sqft_living and floors 不变的情况下，如果batchroom越多，价格反而下降
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
# predict 默认接受多组数据，这里只有个组数据，但是也必须放在[] 里
model.predict([[3, 2, 2500, 2]])
pred = model.predict(xtrain)
((pred-ytrain) * (pred-ytrain)).sum() / len(ytrain)
# 大概意思就是大概0.3505482583370653 会预测错误
(abs(pred-ytrain)/ytrain).sum() / len(ytrain)
predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)
import math
math.sqrt(((pred-ytrain) * (pred-ytrain)).sum() / len(ytrain))