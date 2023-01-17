import numpy as np # linear algebra

import pandas as pd



df = pd.read_csv(

    filepath_or_buffer='../input/transfusion.csv',

    header=None,

    sep=',')



df.columns=['Recency(months)', 'Frequency (times)', 'Monetary (c.c. bloods)', 'Time (months)', 'whether he/she donated blood in March 2007']

df.dropna(how="all", inplace=True) # drops the empty line at file-end

#df['whether he/she donated blood in March 2007'] = np.where(df['whether he/she donated blood in March 2007']==1, 'Donated in March 2007', 'Not donated in March 2007')#put 'Donated in March 2007' if it is 1 labeled and 'forged' otherwise

df.tail()

# split data table into data X and class labels y



X = df.ix[1:,0:4].values

y = df.ix[1:,4].values



print(X.shape)

print(y.shape)
#Standardizing

from sklearn.preprocessing import StandardScaler

X_std=StandardScaler().fit_transform(X)
#The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the “core” of a PCA: The eigenvectors (principal components) determine the directions of the new feature space, and the eigenvalues determine their magnitude. In other words, the eigenvalues explain the variance of the data along the new feature axes.
#PCA Starts here



#Step 1: Eigendecomposition- Computing Eigenvectors and Eigenvalues



#finding covariance (manually)

mean_vec=np.mean(X_std,axis=0)#find mean in row-wise (axis=0) for each column..will be a (1,X_std_col) dimensional vector

cov_mat=((X_std-mean_vec).T.dot(X_std-mean_vec))/(X_std.shape[0]-1)#cov_mat will be a (X_std[1],X_std[1]) dimensional matrix...that is a (#features,#features) dimensional matrix    

print('Covariance matrix \n%s' %cov_mat)
#Or we could have used the numpy's covariance finding function called 'cov()' to do the same task...both will yield the same result   

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
#Next we perform an eiendecomposition on the covariance matrix

cov_mat=np.cov(X_std.T)#Covariance finding using NumPy

eig_vals,eig_vecs=np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)#a (#features,#features) matrix

print('\nEigenvalues \n%s' %eig_vals)
#In the field of Finance the covarrelation matrix is more used then covariance matrix, like, if we have found out the correlation matrix   

cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)

#Eigendecomposition of the raw data based on the correlation matrix

cor_mat2=np.corrcoef(X.T)

eig_val,eig_vecs=np.linalg.eig(cor_mat2)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
#Observation: Eigendecomposition of the covariance matrix on standard data is same as eigendecomposition of correlation matrix on standard or non standard data...i.e, correlation matrix doesn't care if the data is standardized or not    

#correlation_between_two_variables=(covariance_between_two_variables/multiplication_of_these_variables'_standard_deviation)   
#Singular vector decomposition

#Although eigendecomposition of the covariance or correlation matrix may be more intuitiuve, most PCA implementations perform a Singular Vector Decomposition (SVD) to improve the computational efficiency. So, let us perform an SVD to confirm that the result are indeed the same:   



u,s,v = np.linalg.svd(X_std.T)

u
#Step 2: Selecting Principle Components



#Sorting eigenparis:



#First making sure the eigenvectors have all the same unit (1) length...as their task only to show the direction

#Eigenvectors will form axes in new subspace...taking a few of them will approximate the original dimensions by occupying less memory   



for ev in eig_vecs.T:

    for ev in eig_vecs.T:

        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

print('Everything ok!')

#To choose which eigenvector(s) can be dropped (to reduce dimension) without lossing too much information we have to sort them w.r.t corresponding eigenvalues  

#If we take all the eigenvectors it will exactly mimic the real dimensions but as we want to reduce dimension so we have to discard less important eigenvectors (having corresponding lower eigenvalues)    

#lower eigenvalue means data points on graph doesn't vary too much towards that corresponding eigenvector..i.e, that eigenvector doesn't posess so much information and it is safe to discard it

#although discarding eigenvectors will make it unable to exactly mimic the real dimensions (i.e, real information)...but as we want to reduce dimension (i.e, reduce space complixity) so we have to compensate through the loss   



#Making a list of (eigenvalue, eigenvector) tuples

eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]#..pair like (eval,array[corresponding_evec])  

#Sorting the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
#Finding Explained Variance (a calculation to decide how many principle components we are going to choose from new feature subspace)   

tot=sum(eig_vals)

var_exp=[(i/tot)*100 for i in sorted(eig_vals,reverse=True)]#descending-ordered sorted

cum_var_exp=np.cumsum(var_exp)
from matplotlib import pyplot as plt

with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(4), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(4), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()

#As the first two PCs are having larger variances so taking only them will ensure a good approximation to the data

#N.B: eigenvectors are direction cosines for principal components, while eigenvalues are the magnitude (the variance) in the principal components.  
#Projection Matrix (basically just a matrix of our top k eigenvectors...associated with corresponding eigenvalues in "eig_pairs" variable)    

#Here reducing 4 dimensional feature space to 2 dimensional feature subspace, by choosing "top2" eigenvectors with the highest eigenvalues to construct our dxk-dimensional eigenvector matrix W   



matrix_w=np.hstack((eig_pairs[0][1].reshape(4,1),

                    eig_pairs[1][1].reshape(4,1)))#eig_pair[index][eig_val=0 or eig_vec=1]...reshape is converting them to a (4x1) dimensional row matrix  

                   #hstak will concatenate these row matrices together side by side

print('Matrix W:\n',matrix_w)
#Step 3: Projection onto the feature space

#In the last step we got 4x2-dimensional projection  matrix W. Now we will use it to transform our samples onto the new dimensional space via the equation:  

#Y=X x W, where Y will be a (X_row x W_col) diensional matrix of our transformed samples



Y=X_std.dot(matrix_w)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6,4))

    for lab,col in zip(('0','1'),('blue','yellow')):

        plt.scatter(Y[y==lab,0],

                    Y[y==lab,1],

                    label=lab,

                    c=col)#taking for two PCs from corresponding two cols 0 and 1 when correnponding class label from Y is matched with the given class label y   

    plt.xlabel('Principle Component 1')

    plt.ylabel('Principle Component 2')

    plt.legend(loc='lower-center')

    plt.tight_layout()

    plt.show()
#Using scikit-learn we can do the same thing in a very short length of manual coding

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca=sklearnPCA(n_components=2)#number of components=2

Y_sklearn=sklearn_pca.fit_transform(X_std)#It will take the X_std and do everything to reduce the dimensions
#Now displaying the result got by using scikit-learn library

with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6,4))

    for lab,col in zip(('0','1'),('blue','yellow')):

        plt.scatter(Y_sklearn[y==lab,0],

                    Y_sklearn[y==lab,1],

                    label=lab,

                    c=col)#taking for two PCs from corresponding two cols 0 and 1 when correnponding class label from Y is matched with the given class label y   

    plt.xlabel('Principle Component 1')

    plt.ylabel('Principle Component 2')

    plt.legend(loc='lower-center')

    plt.tight_layout()

    plt.show()
#Observation: Both yielding the same result. So the implementation is correct