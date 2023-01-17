# Importing Libraries

#Dataframe
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
# Plot the Figures Inline
%matplotlib inline

# Machine learning 
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model

import random
import scipy.stats as st

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# Importimg Datasets
train_set = pd.read_csv('../input/train.csv')
test_set= pd.read_csv('../input/test.csv')
# Separating features from labels 
X_train = (train_set.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train_set.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test_set.values.astype('float32')
#Printing the data
X_train_fig = X_train.reshape(X_train.shape[0], 28, 28)

fig = plt.figure()
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.tight_layout()
  plt.imshow(X_train_fig[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
X_train.shape
X_test.shape
# Now let's use a KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)

y_pred = knn_clf.predict(X_test)
# preparing dataframe with image index and predictions for testing dataset (X_test)
df = pd.DataFrame(y_pred)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.head()
# CSV file with image predictions
df.to_csv('results.csv', header=True)
