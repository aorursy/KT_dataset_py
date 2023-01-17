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
test = pd.read_csv("../input/test_AV3.csv")
test.head()
train = pd.read_csv("../input/train_AV3.csv")
train.head()
#the analysis is done for important features only
#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import sklearn
from sklearn.cluster import DBSCAN 
from collections import Counter
#looking into the structure,summary & dimensions of the dataset
print(test.head())
print(train.head())

print(test.describe())
print(test.shape)
print(test.columns)
print(test.dtypes)
print(test[test.Dependents.notnull()])
print(test[test.LoanAmount.notnull()])
print(test.isnull().sum())

print(train.describe())
print(train.shape)
print(train.columns)
print(train.dtypes)
print(train[train.Dependents.notnull()])
print(train[train.LoanAmount.notnull()])
print(train.isnull().sum())

#mean, median mode of the given features

#test dataset
print(test.ApplicantIncome.mean())
print(test.CoapplicantIncome.mean())
print(test.LoanAmount.mean())
print(test.Loan_Amount_Term.mean())
print(test.Credit_History.mean())

print(test.ApplicantIncome.median())
print(test.CoapplicantIncome.median())
print(test.LoanAmount.median())
print(test.Loan_Amount_Term.median())
print(test.Credit_History.median())

print(test.ApplicantIncome.mode())
print(test.CoapplicantIncome.mode())
print(test.LoanAmount.mode())
print(test.Loan_Amount_Term.mode())
print(test.Credit_History.mode())

#train dataset

print(train.ApplicantIncome.median())
print(train.CoapplicantIncome.median())
print(train.LoanAmount.median())
print(train.Loan_Amount_Term.median())
print(train.Credit_History.median())


print(train.ApplicantIncome.mode())
print(train.CoapplicantIncome.mode())
print(train.LoanAmount.mode())
print(train.Loan_Amount_Term.mode())
print(train.Credit_History.mode())

print(train.ApplicantIncome.mean())
print(train.CoapplicantIncome.mean())
print(train.LoanAmount.mean())
print(train.Loan_Amount_Term.mean())
print(train.Credit_History.mean())
#boxplots , histograms
test.dropna(inplace = True)
testset = test[['LoanAmount','ApplicantIncome']]
print(testset.head())
testset.boxplot(column = 'ApplicantIncome')
testset['ApplicantIncome'].hist(bins = 30)
testset.boxplot(column = 'LoanAmount')
testset['LoanAmount'].hist(bins = 30)
train.dropna(inplace = True)
trainset = train[['LoanAmount','ApplicantIncome']]
print(trainset.head())
trainset.boxplot(column = 'ApplicantIncome')
trainset['ApplicantIncome'].hist(bins = 30)
trainset.boxplot(column = 'LoanAmount')
trainset['LoanAmount'].hist(bins = 30)
#imputing missing values by various techniques
#test dataset
#droping missing values
test.dropna(inplace = True)
print(test.isnull().sum())
print(test.LoanAmount.mean())
print(test.LoanAmount.head(100))
#replacing by mean
print(test.LoanAmount.replace(np.NaN , test.LoanAmount.mean()).head(100))
print(test.CoapplicantIncome.replace(np.NaN , test.CoapplicantIncome.mean()).head(100))
#filling missing values
print(test.LoanAmount.head(100))
print(test.LoanAmount.fillna('others').head(100))

#train dataset
#droping missing values
train.dropna(inplace = True)
print(train.isnull().sum())
print(train.LoanAmount.mean())
print(train.LoanAmount.head(100))
#replacing by mean
print(train.LoanAmount.replace(np.NaN , test.LoanAmount.mean()).head(100))
print(train.CoapplicantIncome.replace(np.NaN , test.CoapplicantIncome.mean()).head(100))
#filling missing values
print(train.LoanAmount.head(100))
print(train.LoanAmount.fillna('others').head(100))
#barplots and density plots

#test dataset
test.dropna(inplace = True)
positions  = range(0,5)
income1 = test['LoanAmount'].head()
income2 = test['ApplicantIncome'].head()
plt.bar(positions , income1)
plt.bar(positions , income2)
sns.distplot(testset['ApplicantIncome'],hist = True , kde = True)
sns.distplot(testset['LoanAmount'],hist = True , kde = True)
#train dataset
train.dropna(inplace = True)
positions  = range(0,5)
income1 = train['LoanAmount'].head()
income2 = train['ApplicantIncome'].head()
plt.bar(positions , income1)
plt.bar(positions , income2)
sns.distplot(trainset['ApplicantIncome'],hist = True , kde = True)
sns.distplot(trainset['LoanAmount'],hist = True , kde = True)
#new features

testset['lowerclass'] = (testset['ApplicantIncome']<2865)
testset['lowermiddleclass'] = ((testset['ApplicantIncome']>2865) & (testset['ApplicantIncome']<3787) )
testset['uppermiddleclass'] = ((testset['ApplicantIncome']>3787) & (testset['ApplicantIncome']<5061) )
testset['upperclass'] = ((testset['ApplicantIncome']>5061) & (testset['ApplicantIncome']<72530) )
print(testset.head())

print(train.ApplicantIncome.describe())
trainset['lowerclass'] = (trainset['ApplicantIncome']<2878)
trainset['lowermiddleclass'] = ((trainset['ApplicantIncome']>2878) & (trainset['ApplicantIncome']<3812) )
trainset['uppermiddleclass'] = ((trainset['ApplicantIncome']>3812) & (trainset['ApplicantIncome']<5795) )
trainset['upperclass'] = ((trainset['ApplicantIncome']>5795) & (trainset['ApplicantIncome']<81000) )
print(trainset.head())
#detecting outliers

print(testset.ApplicantIncome.describe())
testset.boxplot(column = 'ApplicantIncome')
sepal = testset['ApplicantIncome']
# formula for outliers = 3rd quartile + 1.5(IQR) & 1st quartile - 1.5(IQR)
out = (sepal > 8354)
print(testset[out])

print(test.head())
data = test.iloc[:,:10]
target = test.iloc[:,10]
print(test[:10])

#DBSCAN and ploting outliers

test.dropna(inplace = True)
test1 = test.select_dtypes(include=[np.number])
print(test1.head())
model = DBSCAN(eps = 600 , min_samples = 10).fit(test1)
print(model)
fig = plt.figure()
ax = fig.add_axes([10,10,10,10])
colors = model.labels_
ax.scatter(test1[:,2].head(), test1[test1:,1], c = colors, s=120)