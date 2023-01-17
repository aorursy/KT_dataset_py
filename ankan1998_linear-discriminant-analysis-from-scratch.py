import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
class LDA:
    
    # Compare LDA with my PCA graph and see what is the difference between two
    # number_of_important_feature is the component axes in mathematical terms
    def __init__(self, number_of_important_features=2):
        self.number_of_important_features=number_of_important_features
        self.LDs=None
    
    def fit(self, X,y):
        feature_count=X.shape[1]
        # Getting unique classes in y
        type_of_class_in_y=np.unique(y)
        # Calculating mean of all samples
        mean_all=np.mean(X,axis=0)
        # Initialising with zeros these below matrix
        separation_within_class=np.zeros((feature_count,feature_count))
        separation_between_class=np.zeros((feature_count,feature_count))
        # Iterating over each type of unique classes of y
        for c in type_of_class_in_y:
            X_of_each_class=X[y==c]
            # Calculating the mean of each unique class
            mean_of_each_class=np.mean(X_of_each_class,axis=0)
            # Calculating separation within class(squared) and summing over it
            separation_within_class=separation_within_class+np.dot((X_of_each_class-mean_of_each_class).T,(X_of_each_class-mean_of_each_class))
            # Calculating difference between mean of each class with mean of overall samples
            mean_difference_with_overall_mean=(mean_of_each_class-mean_all).reshape(feature_count,1)
            # Calculating and summing over separation between classes
            separation_between_class=separation_between_class+(X.shape[0]*np.dot(mean_difference_with_overall_mean,mean_difference_with_overall_mean.T))
            # calculating these formula (d1(squared)+d2(squared)+d3(squared)..)/s1(squared)+s2(squared)+s3(squared)
            # separation_within_class(inverse)xseparation_between_class==>mat_trans
            mat_trans=np.dot(np.linalg.inv(separation_within_class),separation_between_class)
            # Same as PCA 
            # Refer to PCA for explanation and dimensions
            # Link https://www.kaggle.com/ankan1998/pca-from-scratch
            # Details on Eigenvectors
            # For more resources visit https://www.kaggle.com/getting-started/176613
            eigenvalues,eigenvector=np.linalg.eig(mat_trans)
            eigenvector=eigenvector.T
            indexs=np.argsort(eigenvalues)[::-1]
            eigenvector=eigenvector[indexs]
            eigenvalues=eigenvalues[indexs]
            self.LDs=eigenvector[:self.number_of_important_features]
            print(indexs)
            
    def apply(self,X):
        # Projecting on New Axis
        return np.dot(X,self.LDs.T)
    
        
        
dataset=pd.read_csv("../input/wine-pca/Wine.csv")
dataset.head()
dataset.isnull().sum()
dataset=dataset.sample(frac=1)
len(dataset)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
# Splitting
X_train=dataset.iloc[:150,:-1]
X_test=dataset.iloc[150:,:-1]
y_train=dataset.iloc[:150,-1]
y_test=dataset.iloc[150:,-1]
# Standardizing
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
print(X[0:5])
lda=LDA(2)
lda.fit(X,y)
# Projecting
projected=lda.apply(X)
x0=projected[:,0]
x1=projected[:,1]
plt.scatter(x0,x1,c=y)
import seaborn as sns
sns.kdeplot(x0,x1,shade=True,cmap="Purples_d",cbar=True)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,pred)
cn
