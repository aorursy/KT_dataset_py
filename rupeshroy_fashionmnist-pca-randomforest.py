#1.0 Call libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 2.0 Import train and test dataset
train = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
test = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
# 2.1 find the shape of train and test dataset
train.shape
test.shape
# 2.2 Seperate Target column
y_train = train.pop('label')
y_test =  test.pop('label')
X_train = train
X_test = test
# 2.3 Get shape of train and test dataset
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# 2.4 change datasets into numpy arrays
X_train_1 = X_train.to_numpy()
X_test_1 = X_test.to_numpy()
y_train_1 = y_train.to_numpy()
y_test_1 = y_test.to_numpy()
# 3.0 Use PCA on train dataset
pca = PCA()
pca.fit(X_train_1)
# 3.1 Find the statistics from PCA
#     How much variance is explained by each principal component
pca.explained_variance_ratio_[0:20]
#     Cumulative sum of variance of each principal component
cumsum = np.cumsum(pca.explained_variance_ratio_)
cumsum[0:20]
# 3.2 Get the column (principal component) number 
#      when cum explained variance threshold just exceeds 0.95
d = np.argmax(cumsum >= 0.95) + 1
d
# 3.3 Let us also plot cumsum
#     Saturation occurs are Elbow

abc = plt.figure(figsize=(6,4))
abc = plt.plot(cumsum, linewidth=3)
# 4.3.1 Define axes limits
abc = plt.axis([0, 400, 0, 1])
# 4.3.2 Axes labels
abc = plt.xlabel("Dimensions")
abc = plt.ylabel("Explained Variance")
# 4.3.3 Draw a (vertical) line from (d,0) to (d,0.95)
#       Should be black and dotted
abc = plt.plot([d, d], [0, 0.95], "k:")
# 4.3.4 Draw another dotted (horizontal) line 
#       from (0,0.95) to (d,0.95)
abc = plt.plot([0, d], [0.95, 0.95], "k:")
# 4.3.5 Draw a point at (d,0.95)
abc = plt.plot(d, 0.95, "ko")
# 4.3.6 Annotate graph
abc = plt.annotate(
                   "Elbow",             # Text to publish
                   xy=(65, 0.85),       # This parameter is the point (x, y) to annotate.
                   xytext=(70, 0.7),    # The position (x, y) to place the text at.
                   arrowprops=dict(arrowstyle="->"), # A dictionary with properties used
                                                     #  to draw an arrow between the
                                                     #    positions xy and xytext.
                   fontsize=16
                  )
# 4.3.7 Draw a grid
plt.grid(True)
plt.show()
# 4.0 out of 784 ,187 columns have cumulative variance of 95%  
# Get transformed dataset upto 95% explained variance 
pca = PCA(n_components=187)
X_reduced =pca.fit_transform(X_train_1)
X_reduced.shape
# 4.1 Recheck sum of explained variance
np.sum(pca.explained_variance_ratio_)
# 4.2 Use PCA's function inverse_transform() to get origianl
#      dimensions back from reduced dimesionality
X_recovered = pca.inverse_transform(X_reduced)
# 4.3 Check shape of recovered dataset
X_recovered.shape  
# 5.0 Plot few digits from original dataset
#     Digit shapes
fig,axe = plt.subplots(2,5)
axe = axe.flatten()
for i in range(10):
    abc = axe[i].imshow(X_train_1[i,:].reshape(28,28))
    
# 5.1 And few digits from compressed dataset
#     And compare both
fig,axe = plt.subplots(2,5)
axe = axe.flatten()
for i in range(10):
    abc = axe[i].imshow(X_recovered[i,:].reshape(28,28))
# 6.0 Use RandomForestClassifier to train the reduced train  dataset
rf1 = rf()
rf1.fit(X_reduced,y_train_1)
# 6.1 Apply PCA on test dataset and predict the target.
X_test_reduced =pca.fit_transform(X_test_1)
y_predict = rf1.predict(X_test_reduced)
#6.2 Check accuracy score 
accuracy_score(y_predict,y_test)