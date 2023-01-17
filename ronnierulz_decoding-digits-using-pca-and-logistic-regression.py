# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as dt



# For Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# To Scale our data

from sklearn.preprocessing import scale



# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



from sklearn.datasets import fetch_mldata

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mnist = pd.read_csv("../input/train.csv",  sep = ',',encoding = "ISO-8859-1", header= 0)

holdout = pd.read_csv("../input/test.csv",  sep = ',',encoding = "ISO-8859-1", header= 0)



print("Dimensions of train: {}".format(mnist.shape))

print("Dimensions of test: {}".format(holdout.shape))
mnist.head()
# Checking for total count and percentage of null values in all columns of the dataframe.



total = pd.DataFrame(mnist.isnull().sum().sort_values(ascending=False), columns=['Total'])

percentage = pd.DataFrame(round(100*(mnist.isnull().sum()/mnist.shape[0]),2).sort_values(ascending=False)\

                          ,columns=['Percentage'])

pd.concat([total, percentage], axis = 1).head()
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = mnist.drop(['label'], axis=1)



X.head()
# Putting response variable to y

y = mnist['label']



y.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X = scaler.fit_transform(X)



pd.DataFrame(X).head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
pd.DataFrame(X_test).head()
import numpy as np

import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(X_train[0:5], y_train[0:5])):

 plt.subplot(1, 5, index + 1)

 plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)

 plt.title('Training: %i\n' % label, fontsize = 20)
X_train.shape

# We have 30 variables after creating our dummy variables for our categories
#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data

pca.fit(X_train)
pca.n_components_
pca.components_
colnames = list(pd.DataFrame(X_train).columns)

pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})

pcs_df.head()
%matplotlib inline

fig = plt.figure(figsize = (10,10))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
pd.options.display.float_format = '{:.2f}'.format

pca.explained_variance_ratio_
#Making the screeplot - plotting the cumulative variance against the number of components

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train_pca, y_train)
logisticRegr.predict(X_train_pca[0:10])
logisticRegr.predict(X_train_pca)
score = logisticRegr.score(X_train_pca, y_train)

print(score)
logisticRegr.predict(X_test_pca)
score = logisticRegr.score(X_test_pca, y_test)

print(score)
X_train.shape

# We have 30 variables after creating our dummy variables for our categories
#Improting the PCA module

from sklearn.decomposition import PCA

pca_last = PCA(0.90)
#Doing the PCA on the train data

pca_last.fit(X_train)
pca_last.n_components_
pca_last.components_
colnames = list(pd.DataFrame(X_train).columns)

pcs_df = pd.DataFrame({'PC1':pca_last.components_[0],'PC2':pca_last.components_[1], 'Feature':colnames})

pcs_df.head()
%matplotlib inline

fig = plt.figure(figsize = (10,10))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
X_train_pca = pca_last.transform(X_train)

X_test_pca = pca_last.transform(X_test)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
model_pca = logisticRegr.fit(X_train_pca, y_train)

model_pca
logisticRegr.predict(X_train_pca[0:10])
predictions = logisticRegr.predict(X_train_pca)

predictions
score = logisticRegr.score(X_train_pca, y_train)

print(score)
%matplotlib inline

fig = plt.figure(figsize = (8,8))

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c = y_train)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.tight_layout()

plt.show()
%matplotlib notebook

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,8))

ax = Axes3D(fig)

ax = plt.axes(projection='3d')

ax.scatter(X_train_pca[:,2], X_train_pca[:,0], X_train_pca[:,1],zdir='z', s=20, marker = 'o', c=y_train)

ax.set_xlabel('Principal Component 1')

ax.set_ylabel('Principal Component 2')

ax.set_zlabel('Principal Component 3')

plt.tight_layout()

plt.show()
import statsmodels.api as sm

# Logistic regression model

logpca = sm.GLM(y_train,(sm.add_constant(X_train_pca)), family = sm.families.Binomial())

logpca.fit().summary()
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics
cm = metrics.confusion_matrix(y_train, predictions)

print(cm)
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix



fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
print(metrics.accuracy_score(y_train, predictions))
recall = np.diag(cm) / np.sum(cm, axis = 1)

recall
from sklearn.metrics import precision_score, recall_score
recall_score(y_train, predictions,average='macro')
precision = np.diag(cm) / np.sum(cm, axis = 0)

precision
precision_score(y_train, predictions,average='macro')
from sklearn.metrics import precision_recall_fscore_support as score



precision, recall, fscore, support = score(y_train, predictions)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
from sklearn.metrics import classification_report

print(classification_report(y_train, predictions))
predict_test = logisticRegr.predict(X_test_pca)

predict_test
score = logisticRegr.score(X_test_pca, y_test)

print(score)
print(metrics.accuracy_score(y_test, predict_test))
recall_score(y_test, predict_test,average='macro')
precision_score(y_test, predict_test,average='macro')
from sklearn.metrics import classification_report

print(classification_report(y_test, predict_test))
holdout.head()
holdout.shape
# Checking for total count and percentage of null values in all columns of the dataframe.



total = pd.DataFrame(holdout.isnull().sum().sort_values(ascending=False), columns=['Total'])

percentage = pd.DataFrame(round(100*(holdout.isnull().sum()/holdout.shape[0]),2).sort_values(ascending=False)\

                          ,columns=['Percentage'])

pd.concat([total, percentage], axis = 1).head()
from sklearn.preprocessing import StandardScaler
holdout = scaler.transform(holdout)



pd.DataFrame(holdout).head()
holdout_pca = pca_last.transform(holdout)

pd.DataFrame(holdout_pca).head()
predict_holdout = logisticRegr.predict(holdout_pca)

predict_holdout
holdout_ids = np.arange(1,holdout.shape[0]+1)

submission_df = {"ImageId": holdout_ids,"Label": predict_holdout}

submission = pd.DataFrame(submission_df)

submission.head()
submission.to_csv("submission.csv",index=False)