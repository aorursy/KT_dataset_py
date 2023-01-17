import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/train.csv')
data.head()
columns = data.columns[1:-1]
columns
X = data[columns]
y = np.ravel(data['target'])
y_dist = data['target'].value_counts() / len(data)
y_dist.plot(kind = 'bar')
for id in range(1, 10):
    plt.subplot(3, 3, id)
    data[data['target'] == 'Class_' + str(id)].feat_20.hist()
plt.show()
plt.scatter(data['feat_19'], data['feat_20'])
plt.show()
sns.heatmap(X.corr())
num_fea = X.shape[1]
num_perceptron = int((num_fea * 9) ** (1/2))
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (num_perceptron, num_perceptron), random_state = 1, verbose = True)
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
Xtest = test[columns]
Xtest.head()
test_prob = model.predict_proba(Xtest)
test_prob.shape
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution.head()
solution['id'] = test['id']
solution.head()
solution = solution[['id', 'Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]
solution.head()
solution.to_csv('./otto_prediction.tsv', index = False)