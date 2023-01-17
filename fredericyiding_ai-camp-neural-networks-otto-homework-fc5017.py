import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data_path = '../input/'
data = pd.read_csv(data_path + 'train.csv')
columns = data.columns[1:-1]
type(columns)
X = data[columns]
y = np.ravel(data['target'])
y
distribution = data.groupby('target').size() / data.shape[0] * 100
distribution.plot(kind='bar')
plt.show()
for idx in range(9):
    plt.subplot(3, 3, idx + 1)
    data[data.target == 'Class_' + str(idx + 1)].feat_20.hist()
plt.show()
data[data.target == 'Class_1'].feat_20.describe()
plt.scatter(data.feat_19, data.feat_20)
plt.show()
data.feat_20.describe()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(X.corr(), interpolation='nearest')
fig.colorbar(cax)
plt.show()
X.corr()
num_fea = X.shape[1]
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1, verbose=True)
model.fit(X, y)
model.intercepts_
print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)
pred
model.score(X, y)
sum(pred == y) / len(y)
test_data = pd.read_csv(data_path + 'test.csv')
Xtest = test_data[test_data.columns[1:]]
Xtest
test_prob = model.predict_proba(Xtest)
solution = pd.DataFrame(test_prob, columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
solution['id'] = test_data['id']
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = solution[cols]
#print(solution[1:][:1].sum(axis=1))
#solution.values[:, 1:].sum(axis=1)
solution.to_csv('./otto_prediction.csv', index=False)
