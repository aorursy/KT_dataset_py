import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline
train_data = np.load('/kaggle/input/eval-lab-4-f464/train.npy', allow_pickle=True)
train_data
temp = train_data[:, 0:1]

#temp[1][0]

y_train = []

for i in range(2275):

    y_train.append(temp[i][0])

y_train = np.array(y_train)

y_train.shape
temp = train_data[:, 1:2]

#temp[1][0]

x_train = []

for i in range(2275):

    x_train.append(temp[i][0])

x_train = np.array(x_train)

x_train.shape
test_data = np.load('/kaggle/input/eval-lab-4-f464/test.npy', allow_pickle=True)

test_data[:,0].size
temp = test_data[:, 1:2]

#temp[1][0]

x_test = []

for i in range(976):

    x_test.append(temp[i][0])

x_test = np.array(x_test)

x_test.shape
temp = test_data[:, 0:1]

#temp[1][0]

IDs = []

for i in range(976):

    IDs.append(temp[i][0])

IDs = np.array(IDs)

IDs.shape
# convert each image to 1 dimensional array



X = x_train.reshape(len(x_train),-1)

Y = y_train



# normalize the data to 0 - 1



X = X / 255.



print(X.shape)

X[0]
# convert each image to 1 dimensional array



x_test = x_test.reshape(len(x_test),-1)



# normalize the data to 0 - 1



x_test = x_test / 255.



print(x_test.shape)

x_test
print(y_train.shape)
labels, levels = pd.factorize(Y)

labels
from sklearn.cluster import MiniBatchKMeans



# Initialize KMeans model



kmeans = MiniBatchKMeans(n_clusters = 19)



# Fit the model to the training data



kmeans.fit(X)



kmeans.labels_

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 30)

Y
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = MinMaxScaler()

X_test_encoded = scaler.fit_transform(X_test) 

X_train_encoded = scaler.fit_transform(X_train)
x_test_encoded = scaler.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train_encoded,y_train)

y_pred = logreg.predict(X_train_encoded)



print('Train accuracy score:',accuracy_score(y_train,y_pred))

print('Test accuracy score:', accuracy_score(y_test,logreg.predict(X_test_encoded)))
from sklearn.decomposition import PCA



n_components = 150

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(X_train)



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)

x_test_pca = pca.transform(x_test)
from sklearn.svm import SVC 



svm_model_linear = SVC(kernel = 'rbf', C = 1000, gamma=0.005).fit(X_train_pca, y_train) 

svm_predictions = svm_model_linear.predict(X_test_pca) 

print('Test accuracy score:', accuracy_score(y_test,svm_predictions))
from sklearn.ensemble import RandomForestClassifier as RFC



rfc_b = RFC(n_estimators=950, min_samples_split=6)

rfc_b.fit(X_train_pca,y_train)

y_pred = rfc_b.predict(X_train_pca)



print('Train accuracy score:',accuracy_score(y_train,y_pred))

print('Test accuracy score:', accuracy_score(y_test,rfc_b.predict(X_test_pca)))
predictions = svm_model_linear.predict(x_test_pca)

predictions
submission = pd.DataFrame({'ImageId':IDs,'Celebrity':predictions})

submission.shape
filename = 'predictions13.csv'



submission.to_csv(filename,index=False)