import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
df.head()
columns = df.columns[1:-1]
columns
X = df[columns]
y = np.ravel(df['target'])
y
df['target'].value_counts().plot(kind='bar')
for id, class_i in enumerate(set(y)):
    plt.subplot(3, 3, id + 1)
    df[df.target == class_i].feat_20.hist()
plt.show()
plt.scatter(np.ravel(df['feat_19']), np.ravel(df['feat_20']))
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(X.corr(), interpolation='nearest')
fig.colorbar(cax)
plt.show()
num_fea = X.shape[1]
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1, verbose=True)
model.fit(X, y)
model.intercepts_
print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)
pred
model.score(X,y)
sum(pred == y) / len(y)
pd_test = pd.read_csv('../input/test.csv')
features = pd_test.columns[1:]
features
X_test = pd_test[features]
X_test
y_test = model.predict_proba(X_test)
y_test
solution = pd.DataFrame(y_test, columns = ['Class_' + str(i) for i in range(1, 10)])
solution.head()
solution['id'] = pd_test.id
solution.head()
solution = solution[['id'] + ['Class_' + str(i) for i in range(1, 10)]]
solution.head()
solution.to_csv('./otto_prediction.csv', index=False)
