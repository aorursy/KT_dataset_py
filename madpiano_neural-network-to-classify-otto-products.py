import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from __future__ import division
data = pd.read_csv('../input/train.csv')
data
data.dtypes
# all features are continuous number, no discrete
#data.info alternatively
columns = data.columns[1:-1] # feature column, last column is "label"

X = data[columns]
y = np.ravel(data['target'])
y.shape
distribution = data.groupby('target').size() / data.shape[0] * 100.0
# this will give you percentage of each class, total would be 100%
distribution.plot(kind='bar')
plt.ylabel('percentage')
plt.show()
# show how a specific feature distributes in 9 classes
# feature 20
for id in range(9):
    plt.subplot(3, 3, id + 1) # 3行3列
    #plt.axis('off') # 不显示坐标轴
    data[data.target == 'Class_' + str(id + 1)].feat_20.hist()
plt.show()    
# observe relationship between two features
plt.scatter(data.feat_19, data.feat_20)
plt.xlabel('feat_19')
plt.ylabel('feat_20')
plt.show()
# this means inverse proportional; for proportional, it should be a straightline
# On contrary, if we were to plot feat_19 again itself...
plt.scatter(data.feat_19, data.feat_19)
plt.xlabel('feat_19')
plt.ylabel('feat_19')
plt.show()
X.corr()
# Let's use visualization to help understand the correlation matrix
# show relationship between all pairs of features
# correlation

fig = plt.figure()
ax = fig.add_subplot(111) # 1 row, 1 col, 1st plot
cax = ax.matshow(X.corr(), interpolation='nearest') # correlation is -1 to 1
fig.colorbar(cax)
plt.xlabel('feature')
plt.ylabel('feature')
plt.show()
num_fea = X.shape[1]
#alpha is L-2 regularization coefficient
# normally need to iterate on # of nodes, and find the best
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (30, 10), random_state = 1, verbose = True)
# structure: 93 x 30 x 10 x 9
model.fit(X, y)
# could have standardize features
model.intercepts_
print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)
pred
model.score(X, y)
# alternatively, calculate in the following way
sum(pred == y) / len(y)
y
pred
len(y)
sum(pred == y)
test_data = pd.read_csv('../input/test.csv')
Xtest = test_data[test_data.columns[1:]]
Xtest
test_prob = model.predict_proba(Xtest)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution
solution['id'] = test_data['id']
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = solution[cols]
solution.to_csv('./otto_prediction.tsv', index = False)
