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
# Load the training and test data sets

traindf = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')

Xdf = traindf[[col for col in traindf.columns if col != "SalePrice"]]

ydf = traindf['SalePrice']

from sklearn.cross_validation import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf, ydf, train_size=0.8)

print (Xtrain.shape)

print (ytrain.shape)

print (Xtest.shape)

print (ytest.shape)
dfcorr = Xdf.corr()

print (type(dfcorr))

#print (dfcorr.info())

print (dfcorr.shape)

print (dfcorr[dfcorr > 0.2])
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(dfcorr, vmin=dfcorr.values.min(), vmax=1, square=True, 

        linewidths=0.3, annot=True, annot_kws={"size":5})

#_ = sns.pairplot(Xdf)

#plt.figure(figsize=(12, 10))

#_ = sns.corrplot(Xdf, annot=False)
#Xtrain.dtypes

#print (Xtrain.describe)

#print (Xtrain['Id'].dtypes)

#Derice Categorical features

categorical_features  = [col for col in Xtrain.columns if (Xtrain[col].dtypes == 'object')]

print (categorical_features)
# Select the benchmark models features

X_train = traindf[['YrSold','MoSold','LotArea','BedroomAbvGr']]

y_train = traindf['SalePrice']

X_test = testdf[['YrSold','MoSold','LotArea','BedroomAbvGr']]
from sklearn import linear_model
clf = linear_model.LinearRegression()

clf.fit(X_train,y_train)
yhat = clf.predict(X_test)
# Create a dataframe with the row ID and price predictions

yhatdf = pd.DataFrame(data={'Id':testdf.Id, 'SalePrice': yhat})
# Write to CSV file

filename = 'benchmark.csv'

yhatdf.to_csv(filename,index=False)