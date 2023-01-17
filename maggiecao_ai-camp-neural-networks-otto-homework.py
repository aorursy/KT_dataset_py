import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data
X = data.iloc[:,1:-1]

X.shape
y = np.ravel(data['target'])
distribution = data.groupby('target').size()/data.shape[0]*100

distribution.plot(kind='bar')

plt.show()
for i in range(1,10):

    plt.subplot(3,3,i)

    data[data.target == 'Class_' + str(i)].feat_20.hist()



plt.show()
plt.scatter(data.feat_19,data.feat_20)

plt.show()
fig, ax = plt.subplots()

cax = ax.matshow(X.corr(), interpolation='nearest')

fig.colorbar(cax)

plt.show()
num_fea = X.shape[1]
# hidden layer的神经元数量：(MN)**0.5~(MN)**0.5+10 == 29~39

# (M+N)*2/3 = 68
model = MLPClassifier(hidden_layer_sizes =(30,10), solver='lbfgs', alpha=1e-5, random_state=1, verbose=True)
model.fit(X,y)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)
pred = model.predict(X)

pred
model.score(X,y)
sum(pred == y) / len(y)
model2 = MLPClassifier(hidden_layer_sizes=(30,35),solver='lbfgs', alpha=1e-5, random_state=1, verbose=True)

model2.fit(X,y)

pred2 = model2.predict(X)

print(model2.score(X,y))
model3 = MLPClassifier(hidden_layer_sizes=(30,10),solver='adam', alpha=1e-5, random_state=1, verbose=True)

model3.fit(X,y)

pred3 = model3.predict(X)

print(model3.score(X,y))
test = pd.read_csv('../input/test.csv')

Xtest = test.iloc[:,1:]

Xtest
test_prob = model.predict_proba(Xtest)

outcome = pd.DataFrame(test_prob,columns= ['class'+str(i) for i in range(1,10)])
outcome['id']=range(outcome.shape[0])

outcome.set_index('id',inplace=True)

outcome.reset_index(inplace=True)

outcome
outcome.to_csv('./otto_prediction.tsv', index = False)