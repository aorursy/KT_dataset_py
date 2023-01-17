# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#read data
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
target=train["label"]
train=train.drop("label",1)
train.head()
#obtaining covariance matrix for data.
cov_mat= np.cov(train, rowvar=False)
cov_mat.shape
#obtaining eigen values and the corresponding eigen vectors.
eigen_values, eigen_vectors=np.linalg.eig(cov_mat)
print("Eigenvalues of covariance matrix:")
print(eigen_values)
print("Eigen vectors of covariance matrix:")
print(eigen_vectors)
#obtain eigen pairs and sort them in reverse order.
eigen_pairs=[(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

eigen_pairs.sort(key=lambda x: x[0])

eigen_pairs.reverse()

eigen_values_sorted, eigen_vectors_sorted = zip(*eigen_pairs)

eigen_values_sorted=np.array(eigen_values_sorted)
eigen_vectors_sorted=np.array(eigen_vectors_sorted)
from matplotlib import pyplot as plt
#determine variance explained by each eigen value and making a plot.

var_comp_sum=np.cumsum(eigen_values_sorted)/sum(eigen_values_sorted)

print("Cumulative proportion of variance explained vector:")
print(var_comp_sum)

#Number of principal components kept in x-axis.
num_comp=range(1, len(eigen_values_sorted)+1)

plt.title('Cum. Prop. Variance Explain and Componets  Kept')

#x-label
plt.xlabel('Principal Components')

#y-label
plt.ylabel('Cum. Pro. Variance Explained')

plt.scatter(num_comp, var_comp_sum)

plt.show()
pca_one=eigen_vectors_sorted[0]
pca_two=eigen_vectors_sorted[1]
#making data 2d by projecting the data into the principal componetes.
P_reduce=np.array([pca_one, pca_two]).transpose()

project_data_2d=np.dot(train, P_reduce)
##train['pca-one']=project_data_2d[:,0]
train['pca-two']=project_data_2d[:,1]
project_data_2d.shape 