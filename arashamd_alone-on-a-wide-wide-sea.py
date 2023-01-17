# Importing critical libraries
import pandas as pd
import numpy as np
# Importing train & test data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
# Data Visualization and Analysis
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot') # optional: for ggplot-like style

train['label'].value_counts(sort = False).plot(kind='bar')

plt.xlabel('Digit') # add to x-label to the plot
plt.ylabel('Number of instances') # add y-label to the plot
plt.title('Digits in the training data') # add title to the plot

plt.show()
# Lets define feature sets, X_train:
X_train = train.drop(['label'], axis = 1).values # to use scikit-learn library, we have to convert the Pandas data frame to a Numpy array.
X_test = test.values
print(X_train.dtype) # checking data type.
# Lets define our labels, y:
y_train = train['label'].values
print(y_train.dtype)
# Normalize Data
# X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
# I don't think this is necessary in this case since all our data is based on pixel brightness levels.
from sklearn.decomposition import PCA

n_components = 50
# Finding the top n principal components in the data
pca_rand = PCA(n_components = n_components, svd_solver='randomized', whiten = True).fit(X_train)

# Projecting the data onto the eigenspace
X_train_pca = pca_rand.transform(X_train)
X_test_pca = pca_rand.transform(X_test)
# Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 1
neigh = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X_train,y_train)
output = neigh.predict(X_test)
output
submission = pd.DataFrame(data = output, columns=["Label"])
ImageId = np.arange(1, len(submission.index) + 1)
submission.insert(0, "ImageId", ImageId, True)
submission.to_csv("submission.csv", index = False)
