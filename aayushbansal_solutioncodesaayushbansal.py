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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#To upload the csv files I changed the name of the csv files. Test.csv->Dataset1.csv and Train.csv->Dataset2.csv
pd.read_csv('../input/dataset/Dataset1.csv')
pd.read_csv('../input/dataset/Dataset2.csv')
test = pd.read_csv('../input/dataset/Dataset1.csv')
train = pd.read_csv('../input/dataset/Dataset2.csv')
test.dtypes
test.describe()
test.isnull()
test.isnull().sum()
test.shape
test.describe()
test.median(axis=0)
test.mean(axis=0)
test.mode(axis=0, numeric_only=True)
test.isnull().sum()

train.isnull().sum()
train_nan=train[['LoanAmount','Loan_Amount_Term','Credit_History']]
trainnn=(train_nan - train_nan.mean())
trainnn.fillna(0, inplace=True)
list2=trainnn[trainnn.Loan_Amount_Term==0.000000]
list2
from scipy.spatial import distance
for k in range(0,len(list2)): list32=list2.iloc[k] 
euclidean_distances = trainnn.apply(lambda row: distance.euclidean(row, list32), axis=1)
dispd = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
dispd.sort_values(by = 'dist',inplace=True)
list2.iloc[0]['Loan_Amount_Term']=dispd.iloc[1]['dist']
trainnn.iloc[list2.axes[0].tolist()[0],1]=dispd.iloc[1]['dist']
list2=trainnn[trainnn.Loan_Amount_Term==0.000000]
list2
list32=list2.iloc[0] 
euclidean_distances = trainnn.apply(lambda row: distance.euclidean(row, list32), axis=1)
dispd = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
dispd.sort_values(by = 'dist',inplace=True)
list2.iloc[0]['Loan_Amount_Term']=dispd.iloc[1]['dist']
trainnn.iloc[list2.axes[0].tolist()[0],1]=dispd.iloc[1]['dist']
list2=trainnn[trainnn.Loan_Amount_Term==0.000000]
list2
list32=list2.iloc[0] 
euclidean_distances = trainnn.apply(lambda row: distance.euclidean(row, list32), axis=1)
dispd = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
dispd.sort_values(by = 'dist',inplace=True)
list2.iloc[0]['Loan_Amount_Term']=dispd.iloc[1]['dist']
trainnn.iloc[list2.axes[0].tolist()[0],1]=dispd.iloc[1]['dist']
list2=trainnn[trainnn.Loan_Amount_Term==0.000000]
list2
trainnn

list2=trainnn[trainnn.LoanAmount==0.000000]
list2
list32=list2.iloc[0] 
euclidean_distances = trainnn.apply(lambda row: distance.euclidean(row, list32), axis=1)
dispd = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
dispd.sort_values(by = 'dist',inplace=True)
list2.iloc[0]['Loan_Amount_Term']=dispd.iloc[1]['dist']
trainnn.iloc[list2.axes[0].tolist()[0],1]=dispd.iloc[1]['dist']
list2=trainnn[trainnn.Loan_Amount_Term==0.000000]
list2
list2
list2=trainnn[trainnn.LoanAmount==0.000000]
list32=list2.iloc[0] 
euclidean_distances = trainnn.apply(lambda row: distance.euclidean(row, list32), axis=1)
dispd = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
dispd.sort_values(by = 'dist',inplace=True)
list2.iloc[0]['LoanAmount']=dispd.iloc[14]['dist']
trainnn.iloc[list2.axes[0].tolist()[0],0]=dispd.iloc[14]['dist']
list2=trainnn[trainnn.LoanAmount==0.000000]
list2

list2=trainnn[trainnn.Credit_History==0.000000]
list2
list32=list2.iloc[0] 
euclidean_distances = trainnn.apply(lambda row: distance.euclidean(row, list32), axis=1)
dispd = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
dispd.sort_values(by = 'dist',inplace=True)
list2.iloc[0]['Credit_History']=dispd.iloc[1]['dist']
trainnn.iloc[list2.axes[0].tolist()[0],2]=dispd.iloc[0]['dist']
list2=trainnn[trainnn.Credit_History==0.000000]
list2
