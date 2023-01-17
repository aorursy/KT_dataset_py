import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier # multip layer percepton i.e. neural network
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")
data.head()
# -1 表示倒数第一列，但是不包括倒数第一列，也就是从第一列到倒数第二列
columns = data.columns[1:-1]
X = data[columns]
y = np.ravel(data['target'])
# * 100 表示 ，Y轴，显示的百分比
# data.shape[0] 表示总的数据， 求出每一种的百分比
distribution = data.groupby('target').size() / data.shape[0] * 100 #  * 100 使得Y轴上显示的是百分比
distribution.plot(kind='bar')
plt.show()
# range (1, 10) -> 1, 2...8,9
# range(9) 0, 1, 2....7,8
for id in range(1, 10): # 这里表示一共九类
    # 3, 3 表示 3行 3列
    plt.subplot(3,3, id)
    data[data.target == 'Class_' + str(id)].feat_20.hist()
plt.show()
    
# crosstab 一般用来画 一个feature 对 label 的关系图， 
# scatter 一般用来画 两个feature关系 , e.g. plt.scatter(X['sqft_living'], Y) 房价 和 生活面积的关系
plt.scatter(data.feat_19, data.feat_20) # 成反比的关系
plt.show()
fig = plt.figure()
# 111 -> 1 row, 1 col, 1st plot
ax = fig.add_subplot(111)
#cax = ax.matshow(X.corr())
cax = ax.matshow(X.corr(), interpolation='nearest') # matshow specificlly for matrix
# display side bar in right side. 1.0 mean strong correlation
# 对角线颜色表示strong 相关性，因为 feature 1 和 feature 1,  同一个feature 肯定是相关性强的
fig.colorbar(cax)
plt.show()

X.corr().head()
# i.e. cloumn numbers
num_fea = X.shape[1]
num_fea
# alpha is L-2 regularization coefficient
# 93 * 30 * 10 * 9, 一共4层，93 为第一层输入层， 30 * 10 为中间两个hidden layer, 最后一层9 为第四层
model = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes = (30, 10), random_state=1, verbose = True) # alpha 是正则化系数
model.fit(X,y) # X, y 输入后，就可以detect 93 input 也就是第一层， 也可以detect 9 as 最后种类
# 第1层(i.e. input layer)， 第二层， 第三层 。。每一层has 一个intercept, 该intercept和下一层的每个点都练成线
# last layer (i.e output layer with 9 output) doesn't have intercept 
model.intercepts_  # 30， 10， 9
print(model.coefs_[0].shape) # weight variables (i.e arrows) connect input layer (1st layer) with 2rd layer
print(model.coefs_[1].shape) # weight variables (i.e arrows) connect 2rd layer with 3rd layer
print(model.coefs_[2].shape) # weight variables (i.e arrows) connect 3rd layer with 4th layer (i.e. last output layer)
pred = model.predict(X)
pred
model.score(X, y)
sum(pred == y) / len(y)
test_data = pd.read_csv('../input/test.csv')
Xtest = test_data[test_data.columns[1:]]
Xtest.head()
test_prob = model.predict_proba(Xtest)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id']

solution.head()
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1] # 最后一列id 放到第一列
solution = solution[cols]
solution.head() 
solution.to_csv('./otto_prediction.tsv', index = False)