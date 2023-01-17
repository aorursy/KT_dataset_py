import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data
data.dtypes
columns = data.columns[1:-1]
X = data[columns]
print(columns)
y = np.ravel(data['target'])
print(y)
np.unique(y)
# (subtotal counts in each class / total counts)  %
subsum = data.groupby('target').size()
total_counts = data.shape[0]
print(subsum, '\nTotal: ', total_counts)
distribution = subsum / total_counts * 100
distribution.plot(kind = 'bar')
plt.show()
data.shape
len(data)
for id in range(9):
    plt.subplot(3, 3, id+1)
    #plt.axis('off')  # not shown the axises
    data[data.target == 'Class_' + str(id+1)].feat_20.hist()  # Class_i
plt.show()
# 图像说明 feat_19 & feat_20 是负相关
plt.scatter(data.feat_19, data.feat_20)
plt.show()
X.corr()
fig = plt.figure()
ax = fig.add_subplot(111)  # 1 row, 1 col, 1st plot
cm = ax.matshow(X.corr(), interpolation='nearest')
fig.colorbar(cm)
plt.show()
num_fea = X.shape[1]
# log-loss function using LBFGS or stochastic gradient descent. LBFGS is more powerful than GD
# alpha: learning rate, chosen by using cross-validation
model = MLPClassifier(solver= 'lbfgs', hidden_layer_sizes= (30, 10), alpha= 1e-5, random_state=1)
#预测training需要约1分钟
model.fit(X, y)
model.coefs_
print(model.coefs_[0].shape)  # 输入层的shape
print(model.coefs_[1].shape)  # hidden一层的shape
print(model.coefs_[2].shape)  # hidden二层的shape
# 看一下它的图形，边权值比较高的（最亮的点）
fig = plt.figure()
ax = fig.add_subplot(111)  # 1 row, 1 col, 1st plot
cm = ax.matshow(model.coefs_[0], interpolation='nearest')
fig.colorbar(cm)
plt.show()
# bias 有三个，每层一组 intercept， intercept 的个数是下一层的层数： 30， 10， 9
model.intercepts_
pred = model.predict(X)
pred
model.score(X, y)
# 验证上面的score是否正确, 相同就是对的
sum(pred == y) / len(y)
test_data = pd.read_csv('../input/test.csv')
test_data.columns[1:]
# 因为训练时只对第一列到倒数第二列是feature进行了训练，所以这里也去掉Id列，取1-93列的feature做测试
Xtest = test_data[test_data.columns[1:]]
Xtest
#每一个feature分别属于class1，class2, ... , class9 的概率是多少
test_prob = model.predict_proba(Xtest)
test_prob
test_prob.shape
# 每一行的和是1
sum(test_prob[0,:])
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution
# 再加上Id 列
solution['ID'] = test_data['id']
solution
# 把 ID 列 挪到前面
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = solution[cols]
solution.to_csv('./otto_prediction.csv', index = False)
