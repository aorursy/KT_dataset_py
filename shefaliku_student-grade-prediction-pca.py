# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

%matplotlib inline

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import time
student_mat= pd.read_csv("../input/student-mat.csv")

student_por= pd.read_csv("../input/student-por.csv")
#It show some basic statistical details like percentile, mean, std etc.

student_mat.describe()
student_por.describe()
student_data = pd.merge(student_mat,student_por,how="outer")

student_data.head()

#student_data.shape
col_str = student_data.columns[student_data.dtypes == object]

col_str
student_data = pd.get_dummies(student_data, columns = col_str, drop_first = True)

student_data.info()
print(student_data[["G1","G2","G3"]].corr())
student_data.drop(axis = 1,labels= ["G1","G2"])

student_data
label = student_data["G3"].values

predictors = student_data.drop(axis = 1,labels= ["G3"]).values

student_data.shape

#Using Linear Regression to predict grades

lr = linear_model.LinearRegression()

lr_score= cross_val_score(lr, predictors, label, cv=5)

print("LR Model Cross Validation score : " + str(lr_score))

print("LR Model Cross Validation Mean score : " + str(lr_score.mean()))
#Using PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=len(student_data.columns)-1)

pca.fit(predictors)

variance_ratio = pca.explained_variance_ratio_

pca.explained_variance_.shape
%matplotlib inline

import matplotlib.pyplot as plt

variance_ratio_cum_sum=np.cumsum(variance_ratio)

print(variance_ratio_cum_sum)

plt.plot(variance_ratio_cum_sum)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.annotate('10',xy=(10,.90))
# individual explained variance

plt.figure(figsize=(10, 5))



plt.bar(range(41),pca.explained_variance_, alpha=0.5,label='individual explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')
#correlation between the variables after transforming the data with PCA is 0

import seaborn as sns

correlation = pd.DataFrame(PCA().fit_transform(predictors)).corr()

sns.heatmap(correlation, vmax=1, square=True,cmap='viridis')

plt.title('Correlation between different features')
#Looking at above plot I'm taking 10 variables

pca = PCA(n_components=10)

pca.fit(predictors)

Transformed_vector =pca.fit_transform(predictors)

print(Transformed_vector)

#correlation between the variables after transforming the data with PCA is 0

import seaborn as sns

correlation = pd.DataFrame(Transformed_vector).corr()

sns.heatmap(correlation, vmax=1, square=True,cmap='viridis')

plt.title('Correlation between different features')
lr_pca = linear_model.LinearRegression()

lr_pca_score = cross_val_score(lr_pca, Transformed_vector, label, cv=5)

print("PCA Model Cross Validation score : " + str(lr_pca_score))

print("PCA Model Cross Validation Mean score : " + str(lr_pca_score.mean()))