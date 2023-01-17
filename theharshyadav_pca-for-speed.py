from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
# You can add the parameter data_home to wherever to where you want to download your data
mnist = fetch_mldata('MNIST original')
mnist
# These are the images
mnist.data.shape
# These are the labels
mnist.target.shape
# Splitting the data into training and testing
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)
(train_img.shape)
train_lbl.shape
test_img.shape
test_lbl.shape
# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
# choose the minimum number of principal components such that 95% of the variance is retained.
pca = PCA(.95)
pca.fit(train_img)
pca.n_components_
# Apply transformation to test Logistic Regression
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it
# solver = 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)
# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))
# Actual Label
test_lbl[0]
# Predict for Multiple Observations (images) at Once
logisticRegr.predict(test_img[0:10])
# accuracy (fraction of correct predictions): correct predictions / total number of data points
score = logisticRegr.score(test_img, test_lbl)
print(score)
pd.DataFrame(data = [[1.00, 784, 48.94, .9158],
                     [.99, 541, 34.69, .9169],
                     [.95, 330, 13.89, .92],
                     [.90, 236, 10.56, .9168],
                     [.85, 184, 8.85, .9156]], 
             columns = ['Variance Retained',
                      'Number of Components', 
                      'Time (seconds)',
                      'Accuracy'])

