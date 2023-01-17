# Load relevant packages

import numpy as np

np.random.seed(42)



import pandas as pd



import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import xgboost as xgb
# Prepare the data

data = pd.read_csv('../input/data.csv')

del data['id']

del data['Unnamed: 32']



X = data.ix[:, data.columns != 'diagnosis']

y = data['diagnosis']
data.info()
data.describe()
# Statistics
# Principal component analysis

pca = PCA(n_components = 2)

pca_2d = pca.fit_transform(X)
plt.figure(figsize = (10, 10))

plt.scatter(pca_2d[:,0], pca_2d[:,1], c = y)

plt.title('PCA scatter plot')

plt.show()
# t-SNE

perplexities = (2, 5, 10, 30, 50, 100)

plt.figure(figsize = (10, 10*len(perplexities)))

for i, perplex in enumerate(perplexities):

    print('perplexity: {}'.format(perplex))

    tsne = TSNE(n_components = 2, perplexity = perplex, n_iter = 1000, verbose = 1)

    tsne_2d = tsne.fit_transform(X)

    

    plt.subplot(int('{}1{}'.format(len(perplexities), i+1)))

    plt.title('t-SNE scatter plot, perplexity = {}'.format(perplex))

    plt.scatter(tsne_2d[:,0], tsne_2d[:,1], c = y)

plt.show()
# Split data

train, test = train_test_split(data, test_size = 0.3, random_state = 42)

y_train = train.pop('diagnosis').map({'M' : 1, 'B' : 0})

y_test = test.pop('diagnosis').map({'M' : 1, 'B' : 0})

X_train = train

X_test = test
# Train xgboost

dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test, y_test)

param = {'max_depth' : 3, 'eta' : 0.1, 'objective' : 'binary:logistic', 'seed' : 42}

num_round = 50

bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])
preds = bst.predict(dtest)

preds[preds > 0.5] = 1

preds[preds <= 0.5] = 0

print(accuracy_score(preds, y_test), 1 - accuracy_score(preds, y_test))
# Cross validation with xgboost

y_label = y.map({'M' : 1, 'B' : 0})

dcv = xgb.DMatrix(X, y_label)

param = {'max_depth' : 3, 'eta' : 0.1, 'objective' : 'binary:logistic', 'seed' : 42}

num_round = 50

nfolds = 3

bst_cv = xgb.cv(param, dcv, num_round, nfolds)
loss = bst_cv['test-error-mean'][num_round-1]

print(1 - loss, loss)
# Deep learning Keras stuff

from keras.models import Sequential

from keras.layers import Dense, Activation



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train.values)



model = Sequential()

model.add(Dense(32, input_dim = len(X_train.columns), init = 'uniform'))

model.add(Activation('relu'))

model.add(Dense(16, init = 'uniform'))

model.add(Activation('relu'))

model.add(Dense(1, init = 'uniform'))



model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(scaler.transform(X_train.values), y_train, nb_epoch = 20, batch_size = 10)
y_keras_pred = model.predict_classes(scaler.transform(X_test.values))

print('Accuracy: {}'.format(np.sum(y_keras_pred[:,0] == y_test.values) / float(len(y_test.values))))