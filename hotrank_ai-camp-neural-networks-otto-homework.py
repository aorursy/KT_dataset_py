import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data.head()
columns = data.columns[1:-1]
X = data[columns]
y = np.ravel(data['target'])
dist = data.groupby(data.target).size()/len(data)
dist.plot(kind = 'bar')

for id in range(9):
    plt.subplot(3,3,id+1)
    data[data.target == ('Class_'+str(id+1))]['feat_20'].hist()
plt.show()


plt.scatter(data.feat_19, data.feat_20)
corr = X.corr()
# plt.imshow(corr)

import seaborn as sns
plt.figure(figsize = (9,9) )
sns.heatmap(corr, square = True)
plt.show()
num_fea = X.shape[1]
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (30, 10), random_state = 1, verbose = True)
model.fit(X, y)
model.intercepts_
print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)
pred
model.score(X, y)
sum(pred == y) / len(y)
test = pd.read_csv('../input/test.csv')
X_test = test[test.columns[1:]]
y_pred = model.predict_proba(X_test)
result = pd.DataFrame(y_pred, columns = ['Class_'+str(i) for i in range(1,10)])
result.insert(0, 'id', test.id)
result.to_csv('./otto_prediction.csv', index = False)