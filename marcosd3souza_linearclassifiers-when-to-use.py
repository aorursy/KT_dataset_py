# This notebook describe the step-by-step
# when it's necessary the use of linear classifiers
# to solving the classification problems
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.datasets import make_blobs, make_classification
from sklearn import decomposition
def show_pca_explanation(X):
    # check data by 4 pca components
    pca = decomposition.PCA(n_components=4)
    
    pc = pca.fit_transform(X)
    
    # show barplot for components
    df = pd.DataFrame({'var':pca.explained_variance_ratio_, 'PC':['PC1','PC2','PC3','PC4']})
    sns.barplot(x='PC',y="var", data=df, color="c")
# generation of fake data with linearly separation
# Large class Sep, Easy decision boundary
X, y = make_blobs(n_features=10, n_samples=100, centers=4, random_state=0)

_, (ax) = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax)
ax.set_title("Large class Sep, Easy decision boundary")
plt.show()
show_pca_explanation(X)
# generation of fake data with linearly separation
# Large class Sep, Easy decision boundary
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_classes=3,
    n_clusters_per_class=2,
    class_sep=0.8,
    flip_y=0,
    weights=[0.5,0.5,0.5])

_, (ax) = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax)
ax.set_title("Low class Sep, Hard decision boundary")
plt.show()
show_pca_explanation(X)