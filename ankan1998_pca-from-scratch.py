import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
class PCA:

    

    # I will denote components as features here

    # Though its not mathematically accurate but machine learning it works 

    def __init__(self,number_of_important_features=2):

        # number of specified features

        # Default being passed as 2

        self.number_of_important_features=number_of_important_features

        # Best possible features

        self.features=None

        self._mean=None

        

        

    def fit(self,X):

        # placing mean to as origin of axis

        # axis =0 is mean of rows along the column direction 

        self._mean=np.mean(X,axis=0)

        X=X-self._mean

        

        # Co-variance of N,D -->DxD

        # Also called Autocorrelation as both are X's

        covariance=np.dot(X.T,X)/(X.shape[0]-1)

        print(covariance.shape)

        

        # Eigenvalues,eigenvectors detail discussion below

        # Eigenvector is the vector which doesnot chnage it span(simply, direction) after matrix transformation

        # So, why eigen importance. Best intuitive way to say

        # for 3D object, the eigenvector represents its axis of rotation(For earth eigenvector is the axis of rotation)

        # Formula A(matrix).v(eigenvector)=lambda(eigenvalue).v(eigenvector)

        # So, Intuitively above formula means, matrix transformation of eigenvector is the eigenvector scaled by eigenvalue

        # Here we are finding the eigenvector and eigenvalue of the covariance matrix

        # how to solve is (A-lambda.I(identity matrix))-v=0,  As v is non-zero --> det(A-lambda.I)=0(area under transformation=0)

        # Here lambda is the knob by tweaking it, we change the det = 0

        # We can do all this by only one line of code, isnt it awesome!!!

        # There is very powerful application of eigen's i.e eigenbasis-->diagonalisation()

        # A gift for the patience

        # you can say this to your gf or bf --> "My love for you is like eigenvector"

        eigenvalues,eigenvector=np.linalg.eig(covariance)

        print("eigenvalues-->",eigenvalues.shape)

        print("eigenvalues \n",eigenvalues)

        print("eigenvector-->",eigenvector.shape)

        print("eigenvector \n",eigenvector)

        #sort the eigenvalues from highest to lowest

        # If we didnt transpose, then applying indexs will require more steps and computation

        eigenvector=eigenvector.T

        print("eigenvector.T-->",eigenvector.shape)

        print("eigenvector after Transpose\n",eigenvector)

        indexs=np.argsort(eigenvalues)[::-1]

        #taking those indices and storing in eigenvalues and eigenvectors accordingly

        eigenvector=eigenvector[indexs]

        print("eigenvector-indexs-->",eigenvector.shape)

        print("eigenvector after indexes \n",eigenvector)

        eigenvalues=eigenvalues[indexs]

        print("eigenvalues-indexs-->",eigenvalues.shape)

        print("eigenvalues \n",eigenvalues)

        

        ## This below code snippet is for seeing how to determine which feature to be calculated

        total = sum(eigenvalues)

        variance_of_each_feature = [(i / total)*100 for i in eigenvalues]

        print("variance of each feature-->",variance_of_each_feature)

        

        # Now taking only number of specified componenets

        self.features=eigenvector[:self.number_of_important_features]

        print("self.features",self.features.shape)

        # So, now the we have chosen most significant features componenet

        

    def apply(self,X):

        # Here we project the data onto Principal component line

        X=X-self._mean

        # Check the dimensionality with (.shape) to confirm for yourselves

        # Here X-->(N,4);self.features-->2,4

        # (X,self.features.T)-->(N,4)x(4,2)==(N,2) i.e N samples with 2 feature vector 

        return np.dot(X,self.features.T)

        

        

        
from sklearn.datasets import load_iris
iris = load_iris()

iris_df = pd.DataFrame(iris.data,columns=[iris.feature_names])

iris_df.head()
iris_df.columns=iris_df.columns.sort_values()
iris_df.head()
X = iris_df.iloc[:,:]
X.shape
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

print(X[0:5])
pca=PCA(2)
pca.fit(X)
projected=pca.apply(X)
x0=projected[:,0]

x1=projected[:,1]
y=iris.target

# For coloring the graph, unsupervised method no need to think much
plt.scatter(x0,x1,c=y)