import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
columns = data.columns[1:-1]
X = data[columns]
y = np.ravel(data['target'])
import seaborn as sns
sns.countplot(y)
for id in range(9):
    plt.subplot(3, 3, id + 1)
    data[data.target == 'Class_' + str(id + 1)].feat_20.hist()
plt.show()    
sns.jointplot("feat_20", "feat_19", data=data,size=5, ratio=3, color="r")
f,ax = plt.subplots(figsize=(30, 30))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
num_fea = X.shape[1]
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
test_data = pd.read_csv('../input/test.csv')
Xtest = test_data[test_data.columns[1:]]
Xtest
test_prob = model.predict_proba(Xtest)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id']
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = solution[cols]
solution.to_csv('./otto_prediction.tsv', index = False)
