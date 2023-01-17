import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')

data.head()
columns = data.columns.tolist()[1:-1]
X = data[columns]
y = np.ravel(data['target'])
df = data.groupby(['target']).size() / data.shape[0] * 100.

df.plot(kind='bar')

plt.show()
for i in range(9):

    plt.subplot(3, 3, i+1)

    data[data['target'] == 'Class_' + str(i+1)].feat_20.hist()

plt.show()
plt.scatter(data.feat_19, data.feat_20)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

cax = ax.matshow(X.corr())

fig.colorbar(cax)

plt.show()
num_fea = X.shape[1]

num_fea
model = MLPClassifier(hidden_layer_sizes=(30, 10), alpha=1e-5, activation='relu', \

                      solver='lbfgs', verbose=3, random_state=0)
model.fit(X, y)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)
pred = model.predict(X)

pred
model.score(X, y)
sum(pred == y) / len(y)
cross_val_score(model, X, y, cv=4)
test = pd.read_csv('../input/test.csv')
preds_test = model.predict_proba(test.iloc[:,1:])



result = pd.DataFrame(preds_test, columns=['Class_'+str(i+1) for i in range(9)])

result['id'] = test['id']



# reorder 

cols = result.columns.tolist()

cols = cols[-1:] + cols[:-1]

result = result[cols]
result.to_csv('./otto_predictions.csv', index=False)