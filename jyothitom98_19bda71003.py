# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns               

import matplotlib.pyplot as plt     

from scipy.stats import *                

from sklearn.decomposition import PCA   

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import classification_report

#reading the dataset

data = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")   
#previewing the first 5 rows of the dataframe

data.head()
#obtaining the dimensions of the dataframe

data.shape
#column names of the dataframe

data.columns
#data type of each column

data.dtypes
#checking for missing values 

data.isna().sum()
#checking for class imbalance

new_data['flag'].value_counts()/new_data.shape[0]
#descriptive statistics for each column

data.describe()
#segregating the data into independent and dependent variables

X = data.drop(['flag','timeindex'],1)

y = data['flag']
#plotting KDE plots to observe the distribution of each column for each category of 'flag'

for i, col in enumerate(X.columns):

    plt.figure(i)

    dat1=data[col][data['flag']==0]

    dat2 = data.currentBack[data['flag']==1]

    plt.ylabel('Probability')

    plt.xlabel(col)

    plt.title('KDE Plot for column '+ col)

    sns.kdeplot(dat1, label='Anomaly')

    sns.kdeplot(dat2, label='Normal')

    plt.show()
#to compute the Kruskal-Wallis test statistic to detect the relationship between y and each continuous variable

for i, col in enumerate(X.columns):

    dat1=data[col][data['flag']==0]

    dat2 = data.currentBack[data['flag']==1]

    print(kruskal(dat1,dat2))
#correlation matrix

cor = round(X.corr(),2)

cor
#heatmap depicting the correlation between independent variables

sns.heatmap(cor, cmap="Blues", linewidths=0.3)
#computing principal components to reduce multicollinearity

pca = PCA(n_components=5)     #5 principal components used

X=pca.fit_transform(X)
#calculating the variance explained by each principal component

var = pca.explained_variance_ratio_   

var
#calculating the cumulative variance explained by the principal components

cum_var = np.cumsum(pca.explained_variance_ratio_)*100

cum_var
#dataframe with the principal components

pca_X=pd.DataFrame(X)
#correlation matrix for the principal components, to confirm the removal of multicollinearity

cor_pca = round(pca_X.corr(),2)

cor_pca
#splitting the dataset into training and test data; test size = 30%

X_train, X_test, y_train, y_test = train_test_split(pca_X,y, test_size=0.3)
#building a SVM model on the training data

svc_model = svm.SVC()

svc_model.fit(X_train,y_train)
#predicting the class for the test data 

y_pred = svc_model.predict(X_test)
#to get the F1 score for each class

print(classification_report(y_test, y_pred))
#reading the sample data

sample = pd.read_csv('../input/bda-2019-ml-test/Test_Mask_Dataset.csv')

sample = sample.drop('timeindex',1)

sample.shape
#computing the principal components for the sample data

sample=pca.transform(sample)
#predicting the class for the records in the sample data

y = svc_model.predict(sample)

y.shape
#reading the Sample Submission csv file

output = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
#updating the values in the 'flag column'

output['flag']=y
#writing the new csv file

output.to_csv('Sample Submission.csv', index=False)