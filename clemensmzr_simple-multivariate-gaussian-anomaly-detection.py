import numpy as np 

import pandas as pd 



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/creditcard.csv')

data.head()
data.describe()
normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]
plt.figure();
matplotlib.style.use('ggplot')

pca_columns = list(data)[1:-2]

normal_data[pca_columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2));
normal_data["Amount"].loc[normal_data["Amount"] < 500].hist(bins=100);
print("Mean", normal_data["Amount"].mean(), fraud_data["Amount"].mean())

print("Median", normal_data["Amount"].median(), fraud_data["Amount"].median())
fraud_data["Amount"].hist(bins=100);
normal_data["Time"].hist(bins=100);
fraud_data["Time"].hist(bins=50);
normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]

plt.matshow(normal_pca_data.corr());
num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1)[:-num_test].values



X_train = shuffled_data[:-2*num_test]



X_valid = np.concatenate([shuffled_data[-2*num_test:-num_test], fraud_pca_data[:246]])

y_valid = np.concatenate([np.zeros(num_test), np.ones(246)])



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[246:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(246)])
def covariance_matrix(X):

    m, n = X.shape 

    tmp_mat = np.zeros((n, n))

    mu = X.mean(axis=0)

    for i in range(m):

        tmp_mat += np.outer(X[i] - mu, X[i] - mu)

    return tmp_mat / m
cov_mat = covariance_matrix(X_train)
cov_mat_inv = np.linalg.pinv(cov_mat)

cov_mat_det = np.linalg.det(cov_mat)

def multi_gauss(x):

    n = len(cov_mat)

    return (np.exp(-0.5 * np.dot(x, np.dot(cov_mat_inv, x.T))) 

            / (2. * np.pi)**(n/2.) 

            / np.sqrt(cov_mat_det))
from sklearn.metrics import confusion_matrix



def stats(X_test, y_test, eps):

    predictions = np.array([multi_gauss(x) <= eps for x in X_test], dtype=bool)

    y_test = np.array(y_test, dtype=bool)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    recall = tp / (tp + fn)

    prec = tp / (tp + fp)

    F1 = 2 * recall * prec / (recall + prec)

    return recall, prec, F1
eps = max([multi_gauss(x) for x in fraud_pca_data.values])

print(eps)
recall, prec, F1 = stats(X_valid, y_valid, eps)

print("For a boundary of:", eps)

print("Recall:", recall)

print("Precision:", prec)

print("F1-score:", F1)
validation = []

for thresh in np.array([1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]) * eps:

    recall, prec, F1 = stats(X_valid, y_valid, thresh)

    validation.append([thresh, recall, prec, F1])
x = np.array(validation)[:, 0]

y1 = np.array(validation)[:, 1]

y2 = np.array(validation)[:, 2]

y3 = np.array(validation)[:, 3]

plt.plot(x, y1)

plt.title("Recall")

plt.xscale('log')

plt.show()

plt.plot(x, y2)

plt.title("Precision")

plt.xscale('log')

plt.show()

plt.plot(x, y3)

plt.title("F1 score")

plt.xscale('log')

plt.show()
data.plot.scatter("V1","V2", c="Class")

data.plot.scatter("V2","V3", c="Class")

data.plot.scatter("V1","V3", c="Class")