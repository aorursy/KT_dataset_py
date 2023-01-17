import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
print(data.shape)

data.head(10)
data.dtypes
columns = data.columns[1:-1]

X = data[columns]
y = np.ravel(data['target'])
distribution = data.groupby('target').size() / data.shape[0]

distribution.plot(kind='bar')

plt.show()
for id in range(1, 10):

    plt.subplot(3, 3, id)

    data[data.target == 'Class_' + str(id)].feat_20.hist()

plt.show()
plt.scatter(data.feat_19, data.feat_20)

plt.show()
fig = plt.figure(figsize=[10,10])

ax = fig.add_subplot(111)

cax = ax.matshow(X.corr(), interpolation='nearest')

fig.colorbar(cax)

plt.show()
num_fea = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

#model_test = MLPClassifier(solver='sgd', learning_rate_init=0.001,alpha=1e-5, hidden_layer_sizes=(25, 9), random_state=31, verbose=True)

model_test = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(50, 30, 10), random_state=31, verbose=True)
model_test.fit(X_train, y_train)
y_pred = model_test.predict(X_test)

print('train score: ' + str(model_test.score(X_train, y_train)))

print('test score: ' + str(metrics.accuracy_score(y_test, y_pred)))



cross_results = cross_val_score(model_test, X, y, scoring='accuracy', cv=5)

print(cross_results)

print('cross_scores_mean: ' + str(cross_results.mean()))
model = model_test
model.fit(X, y)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)

print(model.coefs_[3].shape)
pred = model.predict(X)

pred
model.score(X, y)
sum(pred == y) / len(y)
test_data = pd.read_csv('../input/test.csv')

X_test_date = test_data[test_data.columns[1:]]

print(test_data.shape)

X_test_date.head()
test_prob = model.predict_proba(X_test_date)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])

solution['id'] = test_data['id']

cols = solution.columns.tolist()

cols = cols[-1:] + cols[:-1]

solution = solution[cols]



solution.to_csv('./otto_prediction.tsv', index = False)
