%config IPCompleter.greedy=True

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



from sklearn.metrics import mean_squared_error

from math import sqrt

def showNumbers(dataset, labels = None, number = 0):

    nplots = 10

    size = 15

    if number > 0:

        nplots = 1

        size = 3

    fig1, ax1 = plt.subplots(nplots,nplots, figsize=(size,size))

    

    if nplots == 1:

        ax1 = [ax1]

    else:

        ax1 = ax1.flatten()

    for i in range(nplots*nplots):

        ax1[i].imshow(dataset.values[i+number].reshape((28,28)), cmap="gray_r")

        ax1[i].axis('off')

        if (labels is not None):

            ax1[i].set_title(labels[i+number])

        else:

            ax1[i].set_title('True: %s'%dataset.index[i])
train_data = pd.read_csv('../input/train.csv',index_col='label')

test_data = pd.read_csv('../input/test.csv')
train_data.describe()
showNumbers(train_data, train_data.index)
def transform_train(dataset):

    pca = PCA(n_components=50, whiten=True)

    pca.fit(dataset.values)

    return pca

def transform(dataset, pca):

    dataset = pd.DataFrame(data=pca.transform(dataset))

    return dataset

def itransform(dataset, pca):

    columns = ['pixel%i'%i for i in range(28*28)]

    return pd.DataFrame(data=pca.inverse_transform(dataset), columns=columns)
X = train_data.values

Y = train_data.index

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05, random_state=1)

pca = transform_train(train_data)

X_train = transform(X_train, pca)

X_test = transform(X_test, pca)



#X_train.head(10)

h = plt.hist(Y)
#model = GradientBoostingClassifier(n_estimators=100)

#model = MLPClassifier(max_iter = 1500, hidden_layer_sizes=(100))

#model = RandomForestClassifier(n_estimators = 100)

model = SVC()



model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)

test_score = model.score(X_test, y_test)

print('Train: %.3f test: %0.3f'%(train_score,test_score))

fig, ax = plt.subplots(1,1, figsize=(5,5))

р = ax.hist2d(x=model.predict(X_test),y=y_test, cmap=plt.cm.jet)

ax.set_xlabel('Predicted')
#X_test['Prediction'] = model.predict(X_test)

showNumbers(itransform(X_test,pca), y_test)#model.predict(X_test))

#X_test.head()
showNumbers(test_data, model.predict(transform(test_data,pca)))
result = pd.DataFrame(data=model.predict(transform(test_data,pca)), columns = ['Label'])

result.index = np.arange(1, len(result)+1)

result.to_csv('sub4.csv', index_label='ImageId')

result.head()