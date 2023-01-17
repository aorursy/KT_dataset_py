import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import itertools
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def Multi_KNN_Class(X, y, test_s=0.1, neighbors=[1,2,3], p_val=[1,2], leaf=[30], iterations=20, fig_s=(15,9), path=os.getcwd(), plot=False, verbose=True, jobs=-1):
    """test out all combinations of hyperparameters to find the best model configuration. Returns statistics for mean and standard
  deviation of accuracy over the amount of iterations for each hyperparameter settings."""
    mu_sigma_list = []
    for s in list(set(itertools.product(neighbors, p_val, leaf))):
        i, j, k = s
        acc_list = []
        knn = KNeighborsClassifier(n_neighbors=i, p=j, leaf_size=k, n_jobs=jobs)
        for r in range(iterations):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_s)
            knn.fit(X_train, y_train)
            acc = knn.score(X_val, y_val)
            acc_list.append(acc)
            
        mu = np.mean(acc_list)
        sigma = np.std(acc_list)
        mu_sigma_list.append(('{}_NN__p_{}__leafsize_{}'.format(i,j,k), mu, sigma))
        if verbose: print('{}_NN__p_{}__leafsize_{}'.format(i,j,k), mu, sigma)
        
        if plot:
            x_axis = list(range(len(acc_list)))
            plt.figure(figsize = fig_s)
            text = 'Accuracy: {}_NN__p_{}__leafsize_{}'.format(i,j,k)
            plt.title(text)
            plt.plot(x_axis, acc_list)
            plt.save_fig(path + '/{}.png'.format(text))
    return mu_sigma_list
df = pd.read_csv('../input/train.csv')
target = 'label'
X = df.drop(target,axis=1)/255
y = df[target]

#stats = Multi_KNN_Class(X, y, iterations=1, neighbors=[1], p_val=[1])
knn = KNeighborsClassifier(n_neighbors=1, p=1, n_jobs=-1)
knn.fit(X, y)
X_test = np.array(pd.read_csv('../input/test.csv'))/255
preds = knn.predict(X_test)
submit = np.array(list(enumerate(preds)))
print(submit[0:100])
submit = pd.DataFrame.from_records(submit, columns=['ImageId','Label'])
submit['ImageId'] = submit['ImageId'] + 1
submit.head()
submit.to_csv('Prediction.csv', index=False)