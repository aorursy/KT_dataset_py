#Problem---We have a set of features and want to reduce the number of features while retaining the variance in the data.

#Solution---We will use scikit-learns PCA-principal component analysis



#Importing libraries

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn import datasets



#loading data

digits=datasets.load_digits()



#Standardizing the feature matrix

feature =StandardScaler()

features=feature.fit_transform(digits.data)



#Creating a PCA that will maintaing 99% of variance

pca = PCA(n_components=0.99,whiten=True)



#Applying PCA

features_pca= pca.fit_transform(features)



#Displying results

print("Original no. of features",features.shape[1])

print("After reduction",features_pca.shape[1])
#Problem---The data givenis linearly inseparable and want to reduce the dimensions

#Solution---We will use extended PCA that uses kernels to allow non-linear dimensionality reduction



#importing libraries

from sklearn.decomposition import PCA,KernelPCA

from sklearn.datasets import make_circles



#Creating linearly inseparable data

features,_ = make_circles(n_samples =1000,random_state =1,noise =0.1,factor=0.1)



#applying kernal PCA with radius basis function (RBF) kernel

kpca = KernelPCA(kernel ="rbf", gamma=15,n_components =1)

features_kpca = kpca.fit_transform(features)



#Showing numberof features

print("Original number of features",features.shape[1])

print("Reduced number of features",features_kpca.shape[1])

#Problem---Reduce the features to be used by a classifier

#Solution---We will use linear discriminant analysis(LDA) to project the features onto component axes that maximize the separation of classes



#importing libraries

from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#loading Iris flower dataets

iris = datasets.load_iris()

features = iris.data

target = iris.target



#creating and running an LDA then use it to transform features

lda = LinearDiscriminantAnalysis(n_components=1)

features_lda = lda.fit(features,target).transform(features)



#printing the number of features

print("Original number of features",features.shape[1])

print("Reduced number of features",features_lda.shape[1])
#To see the amount of variance we use explained_variance_ratio_

lda.explained_variance_ratio_
lda.explained_variance_ratio_
#creating and running an LDA then use it to transform features

lda = LinearDiscriminantAnalysis(n_components=None)

features_lda = lda.fit(features,target)



#creating an array of explained variance ratios

lda_var_ratios=lda.explained_variance_ratio_



#creating function

def select_n_components(var_ratio,goal_var:float)->int:

    #Setting initial variance explained so far

    total_variance=0.0

    

    #Setting initial number of features

    n_components=0

    

    #For the explained_variance of each feature

    for explained_variance in var_ratio:

        #adding the explained variance in total

        total_variance+=explained_variance

        #Add one to the number of components

        n_components +=1

        #if we reach our goal level of explained variance

        if total_variance>=goal_var:

            #end the loop

            break

    return n_components





#Run function

select_n_components(lda_var_ratios,0.95)

    
#Problem---Feature matrix consist of non-negative values and reduce the dimensionality.

#Solution---We will use non-negative matrix factorization(NMF) to reduce the dimensionality of the feature matrix



#importing libraries

from sklearn.decomposition import NMF

from sklearn import datasets



#loading data

digits = datasets.load_digits()



#loading feature matrix

features =digits.data



#creating object and fitting and transforming features

nmf =NMF(n_components=10,random_state=1)

features_nmf = nmf.fit_transform(features)



#displaying results

print("Original number of features",features.shape[1])

print("Reduced number of features",features_nmf.shape[1])
#Problem--Reduce the dimensionality of the sparse feature matrix

#Solution---We will use Truncated Singular Value Decomposition(TSVD)



#importing libraries

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csr_matrix

from sklearn import datasets

import numpy as np



#loading data

digits=datasets.load_digits()



#standardizing the feature matrix

features=StandardScaler().fit_transform(digits.data)



#making sparse matrix

features_sparse =csr_matrix(features)



#creating a TSVD

tsvd =TruncatedSVD(n_components=10)



#Applying TSVD to sparse matrix

features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)



#Displaying results

print("Original number of features:", features_sparse.shape[1])

print("Reduced number of features:", features_sparse_tsvd.shape[1])
#Sum of first 3 components' explained variance ratios

tsvd.explained_variance_ratio_[0:3].sum()
#Creating and running an TSVD with one less than number of features

tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)

features_tsvd =tsvd.fit(features)



#List of explained variances

tsvd_var_ratios =tsvd.explained_variance_ratio_



#creating function

def select_n_components(var_ratio,goal_var):

    #setting initial variance 

    total_variance =0.0

    #setting initial number of features

    n_components=0

    

    #for the explained variance of each features

    for explained_variance in var_ratio:

        

        #add the explained variance to the total

        total_variance +=explained_variance

        #add one to the number of components

        n_components+=1

        

        if total_variance>=goal_var:

            #end the loop

            break

    return n_components





select_n_components(tsvd_var_ratios,0.95)