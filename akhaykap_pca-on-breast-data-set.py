# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#loading the datasets

from sklearn.datasets import load_breast_cancer

import pandas as pd

import numpy as np
#storing the data

breast_data = load_breast_cancer()

#lets see the type of this data, usually data loaded from sklearn are not in pandas dataframe

print("type of breast cancer data is ", type(breast_data))

# lets print how the data looks like in <class 'sklearn.utils.Bunch'>

print(breast_data)
# lets convert data from sklearn to pandas dataframe

df_b_cancer = pd.DataFrame(breast_data.data, columns = breast_data.feature_names)

df_b_cancer['target'] = pd.Series(breast_data.target)

#Source : https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset



#lets see the data now, its shape and basic summary

print("The shape of the dataset is  : ",df_b_cancer.shape)

print(df_b_cancer.head(5))
df_b_cancer['target'].replace(0,'Benign', inplace = True)

df_b_cancer['target'].replace(1,'Malignant', inplace = True)
df_b_cancer.groupby('target').describe().T
from sklearn.preprocessing import StandardScaler

x = df_b_cancer.iloc[:,0:30]

print("shape of x is" ,x.shape)

# x would be only containing features now

y = df_b_cancer.target

print("shape of y is :", y.shape)
# I would be using standard scalar for sacling the features for n samples

from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)

#lets see the mean of x now

print("mean of all the features ",np.mean(x))

print("std deviation of all the features ",np.std(x))
#lets see how x looks like

print("x is a nparray",x)

print("number of columns in x  are" ,x.shape[1])

print("number of rows in x are ",x.shape[0])
# its a numpy array, for better visualaization lets convert it to a pandas dataframe

feature_cols = ['feature' + str(i) for i in range (x.shape[1])]

normalised_breast = pd.DataFrame(x,columns=feature_cols)
#lets see 3 samples from normalized dataset

print(normalised_breast.head(3).T)
# now comes the fun part I would be projecting 30 dimensional data into 2 Principal components using PCA

from sklearn.decomposition import PCA

pca_breast = PCA(n_components=2)

principalComponents_breast = pca_breast.fit_transform(x)

# converting principalComponents_breast to dataframe

principal_breast_Df = pd.DataFrame(data = principalComponents_breast, columns = ['principal component 1', 'principal component 2'])

# lets see how this newly created datframe from two principal components looks like

principal_breast_Df.head(4)
# now I want to know how much variance is explained by both of the principal components

print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))
# let us write a function that can tells us how much variance is explained by n number of principal components

def pc_n(n):

    pca_b = PCA(n_components=n)

    principalComponents_breast = pca_b.fit_transform(x)

    temp = pca_b.explained_variance_ratio_

    return np.sum(temp) 
for n in range(30):

    temp = pc_n(n)

    if temp >=.95:

        break;

print("the optimal value of n is ", n, "with total explainable variance of ", temp)
# using only 2 dimensions for visulization purposes

pca_breast = PCA(n_components=2)

principalComponents_breast = pca_breast.fit_transform(x)

# converting principalComponents_breast to dataframe

principal_breast_Df = pd.DataFrame(data = principalComponents_breast, columns = ['principal component 1', 'principal component 2'])

# lets see how this newly created datframe from two principal components looks like

principal_breast_Df.head(4)
import matplotlib.pyplot as plt

%matplotlib inline

#plt.figure()

plt.figure(figsize=(10,10))

plt.xticks(fontsize=12)

plt.yticks(fontsize=14)

plt.xlabel('Principal Component - 1',fontsize=20)

plt.ylabel('Principal Component - 2',fontsize=20)

plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)

targets = ['Benign', 'Malignant']

colors = ['r', 'g']



for target, color in zip(targets,colors):

    indicesToKeep = df_b_cancer['target'] == target

    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']

               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)



plt.legend(targets,prop={'size': 15})