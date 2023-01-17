import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv', index_col='id')

print(data.shape)

data.head()
columns = data.columns.tolist()[1:-1]

data.columns.tolist()
X = data[columns]
y = np.ravel(data['target'])
#data.groupby('target').size()

distribution = data['target'].value_counts()/data.shape[0]*100

distribution.plot(kind='bar')

plt.show()

target_list=distribution.index.tolist()
for ind,target in enumerate(target_list):

    plt.subplot(3,3,ind+1)

    data[data.target == target]['feat_20'].hist()

    plt.title('feature %s'%target)
plt.scatter(data['feat_19'],data['feat_20'])

plt.xlabel('feature_19')

plt.ylabel('feature_20')
fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

cax = ax.matshow(X.corr(), interpolation='nearest') #show matrix

fig.colorbar(cax)

plt.show()

num_fea = X.shape[1]
model = MLPClassifier(hidden_layer_sizes=(30,10),solver='lbfgs', alpha=1e-5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)

model.fit(X_train,y_train)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)
pred = model.predict(X_test)

pred
print('Training sample accuracy %s' %model.score(X_train, y_train))

print('Test sample accuracy %s' %metrics.accuracy_score(y_test, pred))

metrics.confusion_matrix(y_test, pred)

#Retraining model with all data

data_test = pd.read_csv('../input/test.csv', index_col='id')

X_test = data_test[columns]

test_prob = model.fit(X,y).predict_proba(X_test)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])

solution['id'] = data_test.index

cols = solution.columns.tolist()

cols = cols[-1:] + cols[:-1] #Move 'ID' to first column

solution = solution[cols]

solution
solution.to_csv('./otto_prediction.tsv', index = False)