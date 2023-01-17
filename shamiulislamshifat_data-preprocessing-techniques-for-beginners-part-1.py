# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading dataset
dfA=pd.read_csv("/kaggle/input/motion-data-for-wearable-activity-recognition/DataA.csv")
dfA
#Q-1 problem 1: lots of NaN values needed to be replaced with zeros
#NaN values are replaced with zeros
dfA_NaN_removed=dfA.fillna(0)
dfA_NaN_removed
#Q-1 problem 2:
#lets check if outliers are present in dataset or not!
#as there are multiple variables, we are using z-score to find outliers.
from scipy import stats
z = np.abs(stats.zscore(dfA_NaN_removed.iloc[:, 1:82]))
threshold = 3  #please look how z-score works
count_outliers=len(np.where(z>3))
print("no. of outliers:",np.count_nonzero(z > 3))
print("outliers position:",np.where(z > 3)) #first array row position, second array column position z[60][55]......
#another technique-IQR-interquartile range (IQR) to find outliers
#calculating IQR
Q1 = dfA_NaN_removed.iloc[:, 1:82].quantile(0.25)
Q3 = dfA_NaN_removed.iloc[:, 1:82].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#lets find outliers. 'true' means there is an outlier, 'false' means valid.
print(dfA_NaN_removed.iloc[:, 1:82] < (Q1 - 1.5 * IQR)) |(dfA_NaN_removed.iloc[:, 1:82] > (Q3 + 1.5 * IQR))
#lets remove outliers using Z score and IQR
#We are using IQR here only as this is more robust than z score
dfA_out_removed= dfA_NaN_removed.iloc[:, 1:82][~((dfA_NaN_removed.iloc[:, 1:82] < (Q1 - 1.5 * IQR)) |(dfA_NaN_removed.iloc[:, 1:82] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("original dataset shape: ",dfA_NaN_removed.iloc[:, 1:82].shape)
print("After removing  outliers: ", dfA_out_removed.shape)
print()
#Q-1 problem 3:
#Lets normalize data using min-max and z-score
# Min-max normalization: Guarantees all features will have the exact same scale but does not handle outliers well.
#Z-score normalization: Handles outliers, but does not produce normalized data with the exact same scale.
#after removing outliers from dataset, then normalizing.
#1. min-max norm using scikit learn library
from sklearn import preprocessing

x = dfA_out_removed.iloc[:, 1:82].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfA_normMinMax = pd.DataFrame(x_scaled)
print(" After min max score norm:", dfA_normMinMax)



##2. z score norm using scikit learn library
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#using after removing outliears values
y = dfA_out_removed.values #returns a numpy array
z_scaler = preprocessing.StandardScaler()
z_scaled = z_scaler.fit_transform(y)
dfA_normZ = pd.DataFrame(z_scaled)
print(" After Z score norm:", dfA_normZ)


# plotting histograms of feature 9 and 24 before and after normalization

print("before normalization")
dfA_out_removed[["fea.8", "fea.24"]].plot.hist()

print("after normalization")
dfA_normMinMax[[8, 24]].plot.hist()
