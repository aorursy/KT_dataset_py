import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data.shape
data.head()
data.dtypes
columns = data.columns[1:-1] #数据中从第一列到倒数第二列是feature
X = data[columns]
y = np.ravel(data['target'])
X
y #字符串的数组
distribution = data.groupby('target').size()
print(distribution)
distribution = data.groupby('target').size() / data.shape[0] * 100.0  #得到百分比
distribution.plot(kind='bar') #用柱状图画出来
plt.show()
#feature 20 
for id in range(1, 10):#Class 1-9对应的feat_20都画出来
    plt.subplot(3, 3, id)
    data[data.target=='Class_' + str(id)].feat_20.hist()
plt.show()
plt.scatter(data.feat_19, data.feat_20)
plt.xlabel('feat_19')
plt.ylabel('feat_20')
plt.show()#负相关
X.corr() #对角线上自己和自己相关性自然为1
fig = plt.figure()
ax  = fig.add_subplot(111) # 1 row, 1 column, 1st plot
cax = ax.matshow(X.corr(), interpolation='nearest')
fig.colorbar(cax)
plt.show()
num_fea = X.shape[1]
num_fea
#solver用来设置比如用什么梯度下降方法，包括学习率怎么调整这些
#lbfgs---优化方法，alpha is L-2 regularization coefficient 
#random_state = 1 随机种子，为了保持每次一致  #verbose = True 输出一些东西
model = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes = (30, 10), random_state = 1, verbose = True)
model.fit(X,y) #这里的激活函数是'relu'
#截距 
#ppt page16---第一层是93连到30,所以是30个截距（30个weight）；第二层是30连到10,所以是10个截距；第三层是10连到9,所以是9个截距
model.intercepts_ 
#系数
model.coefs_[0]
model.coefs_[0].shape
#总共有30+10+9+93*30+30*10+10*9个参数，对应ppt page16上的-30,10,20,......
print(model.coefs_[0].shape)
print(model.coefs_[1].shape) #第一隐藏层到第二隐藏层除去截距的个数
print(model.coefs_[2].shape)
pred = model.predict(X)
pred
#准确度
model.score(X, y)
#准确度也可以这么算
sum(pred == y) / len(y)
test_data = pd.read_csv('../input/test.csv')
test_data.shape
data.head()
Xtest = test_data[test_data.columns[1:]]
Xtest.head()
test_prob = model.predict_proba(Xtest)
test_prob.shape
test_prob
np.sum(test_prob, axis = 1) #对每一个商品预测出的属于每一种类别的概率加起来肯定是1
solution = pd.DataFrame(test_prob, columns = ['Class_1','Class_2','Class_3','Class_4', 'Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id'] #加入id列
solution.head()
solution.shape
cols = solution.columns.tolist() #拿出所有的column
cols = cols[-1:] + cols[:-1] #拿出最后一个column补在前面
solution = solution[cols]
solution
solution.to_csv("/Users/jiadileng/Desktop/machine learning/jiuzhang/week6/predict.csv", index = False)
