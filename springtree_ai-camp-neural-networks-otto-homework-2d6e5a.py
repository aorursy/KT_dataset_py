import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
print(data.head())
print(data.dtypes)
columns = data.columns[1:-1]
X = data[columns]
y = np.ravel(data['target'])
distribution = data.groupby('target').size() / data.shape[0] * 100.0
distribution.plot(kind='bar')
plt.show()
for id in range(1, 10):
    plt.subplot(3, 3, id)
    data[data.target=='Class_' + str(id)].feat_20.hist(color='purple')
plt.show()
plt.scatter(data.feat_19, data.feat_20, color='red')
plt.show()
X.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(X.corr(), interpolation='nearest')
fig.colorbar(cax)
plt.show()
num_fea = X.shape[1]
num_fea
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
test_data = pd.read_csv('../input/test.csv')
Xtest = test_data[test_data.columns[1:]]
Xtest.head()
test_prod = model.predict_proba(Xtest)
test_prod
np.sum(test_prod, axis=1)
solution = pd.DataFrame(test_prod, columns=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
solution['id'] = test_data['id']
solution.head()
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = solution[cols]
solution.head()
solution.to_csv('../input/otto_prediction.csv', index=False)
