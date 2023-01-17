# 1.0 Call libraries
import numpy as np
import pandas as pd
import os
# 1.2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 2.0 Check version of sklearn.
#     There should not be any assertion error
import sklearn
assert sklearn.__version__ >= "0.20"
# 2.1 Display mulitple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
X = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
X.head()
print("No of articles:",X.shape[0])
print("No of columns associated to pixels(total columns-label column):",X.shape[1]-1)
y=X.pop('label')
#Split dataset. Default split test-size is 0.25
X_train,X_test,y_train,y_test=train_test_split(X,y)
#Train PCA on dataset
pca = PCA()
pca.fit(X_train)
#Get statistics from pca
#     How much variance is explained by each principal component
pca.explained_variance_ratio_[:10]
#     Cumulative sum of variance of each principal component
cumsum = np.cumsum(pca.explained_variance_ratio_)
cumsum[:10]
#Get the column (principal component) number 
#      when cum explained variance threshold just exceeds 0.95
d = np.argmax(cumsum >= 0.95) + 1
d    
#Let us also plot cumsum
#     Saturation occurs are Elbow

abc = plt.figure(figsize=(6,4))
abc = plt.plot(cumsum, linewidth=3)
#  Define axes limits
abc = plt.axis([0, 400, 0, 1])
#  Axes labels
abc = plt.xlabel("Dimensions")
abc = plt.ylabel("Explained Variance")
# Draw a (vertical) line from (d,0) to (d,0.95)
#       Should be black and dotted
abc = plt.plot([d, d], [0, 0.95], "k:")
#  Draw another dotted (horizontal) line 
#       from (0,0.95) to (d,0.95)
abc = plt.plot([0, d], [0.95, 0.95], "k:")
#  Draw a point at (d,0.95)
abc = plt.plot(d, 0.95, "ko")
# Annotate graph
abc = plt.annotate(
                   "Elbow",             # Text to publish
                   xy=(65, 0.85),       # This parameter is the point (x, y) to annotate.
                   xytext=(70, 0.7),    # The position (x, y) to place the text at.
                   arrowprops=dict(arrowstyle="->"), # A dictionary with properties used
                                                     #  to draw an arrow between the
                                                     #    positions xy and xytext.
                   fontsize=16
                  )
#  Draw a grid
plt.grid(True)
plt.show()
#Get transformed dataset upto 95%
#       explained variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
#shape of reduced data
pca.n_components_
X_reduced.shape
#Recheck sum of explained variance
np.sum(pca.explained_variance_ratio_)
#      dimensions back from reduced dimesionality
X_recovered = pca.inverse_transform(X_reduced)
X_recovered.shape     
#Plot few digits from original dataset
#     Digit shapes
fig,axe = plt.subplots(2,5)
axe = axe.flatten()
plt.title("A")
for i in range(10):
    abc = axe[i].imshow(X_train.iloc[i,:].to_numpy().reshape(28,28))
plt.suptitle('Few Images from Original Dataset')
# few images from compressed dataset
#     And compare both
fig,axe = plt.subplots(2,5)
axe = axe.flatten()
for i in range(10):
    abc = axe[i].imshow(X_recovered[i,:].reshape(28,28))
    
plt.suptitle('Few Images from compressed Dataset')

