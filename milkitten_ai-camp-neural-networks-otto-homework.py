import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
print(data.dtypes)

data.head()
X = data.iloc[:, 1:-1]

X.head()
y = np.ravel(data['target'])
st = data['target'].value_counts()

distribution = st.sort_index()/data.shape[0]*100

distribution.plot.bar()

plt.show()
for i in range(len(st)):

    plt.subplot(3,3,i+1)

    data['feat_20'][data.target == st.index[i]].hist()

plt.show()    
plt.scatter(data['feat_19'], data['feat_20'])

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111) # 1 row, 1 col, 1st plot

cax = ax.matshow(X.corr(), interpolation='nearest')

fig.colorbar(cax)

plt.show()
num_fea = X.shape[1]
model = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes = (30, 10), random_state = 1, verbose = True)
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

test_X = test_data.iloc[:, 1:]

test_X.head()

pred = model.predict_proba(test_X)

pred.shape
columns = ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']

df = pd.DataFrame(pred, columns = columns[1:])

df['id'] = test_data['id']

df=df[columns]

# , 'Class_1': pred[:, 1], 'Class_2': pred[:, 2], 'Class_3': pred[:, 3], 'Class_4': pred[:, 4], 'Class_5': pred[:, 5], 'Class_6': pred[:, 6]})

df.to_csv('./otto_prediction.tsv', index = False)

df