import numpy as np # 数值计算包

from sklearn import datasets # 数据集
from sklearn.linear_model import LogisticRegression # 导入LR算法
from sklearn.model_selection import train_test_split # 自动划分数据集训练集

import matplotlib.pyplot as plt # 绘图工具
from mpl_toolkits.mplot3d import Axes3D # 3D绘图
iris=datasets.load_iris()

X = iris.data
y = iris.target
# 选择三个属性，绘制三维图
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(1,1,1,projection='3d') 
ax.scatter(X[:50,0], X[:50,3], X[:50,1],color='red',s=80,label=u'Type 1')
ax.scatter(X[50:100,0], X[50:100,3], X[50:100,1],color='blue',s=80,label='Type 2')
ax.scatter(X[100:,0], X[100:,3], X[100:,1],color='orange',s=80,label='Type 3')
plt.legend(loc="upper right")
# 任选两个属性，查看属性之间的关系
fig = plt.figure(figsize=(16,10))
for j in range(4):
    for i in range(4):
        ax = plt.subplot(4,4,i+1+4*j)
        plt.xlabel(j);plt.ylabel(i)
        plt.scatter(X[:50,j],X[:50,i],c='r')
        plt.scatter(X[50:100,j],X[50:100,i],c='b')
        plt.scatter(X[100:,j],X[100:,i],c='orange')

# 划分数据集训练集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# 获取算法模型
model = LogisticRegression()

# 训练模型
model.fit(X_train,y_train)
# 预测结果
model.predict(X_test)
# 查看准确率
acc = model.score(X_test,y_test)
print("预测准确率为:%.2f%%,"%(acc*100))
