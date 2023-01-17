import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# columns = train.select_dtypes(include=['object']).columns

# columns = columns.drop(['TotalCharges','gender'])

# encoded = pd.get_dummies(train[columns])

# encoded.head()

train = pd.read_csv("../input/eval-lab-3-f464/train.csv")

train = train.drop(["TotalCharges", "custId", 'gender'], axis = 1)

train = pd.get_dummies(train)

train.head()

#train = train.drop(['TotalCharges','Satisfied','custId','PaymentMethod'], axis = 1)
#train.head()

# for column in encoded.columns:

#     train[column] = encoded[column]

# for column in columns:

#     train = train.drop(column,axis=1)

# #columns

x_train = train.drop(['Satisfied'],axis=1)

x_train.head()
x_train.columns
x_train.head()
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# x_train = scaler.fit_transform(np.array(x_train))

# from sklearn import preprocessing# Get column names first

# names = x_train.columns# Create the Scaler object

# scaler = preprocessing.StandardScaler()# Fit your data on the scaler object

# scaled_train = scaler.fit_transform(x_train)

# scaled_train = pd.DataFrame(scaled_train, columns=names)

# scaled_test.head()

from sklearn.preprocessing import Normalizer

transformer = Normalizer().fit(x_train)

trained = pd.DataFrame(transformer.transform(x_train))

trained.head()
#scaled_train.head()

test = pd.read_csv("../input/eval-lab-3-f464/test.csv")

test = test.drop(['gender', 'TotalCharges', 'custId'], axis=1)

test = pd.get_dummies(test)
# names = X.columns# Create the Scaler object

# scaler = preprocessing.StandardScaler()# Fit your data on the scaler object

# scaled_test = scaler.fit_transform(X)

# scaled_test = pd.DataFrame(scaled_test, columns=names)

from sklearn.preprocessing import Normalizer

transformer = Normalizer().fit(test)

normalized = pd.DataFrame(transformer.transform(test))

normalized.head()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2).fit(x_train)

test_pred = kmeans.fit_predict(normalized)
# from sklearn.cluster import KMeans

# from sklearn.cluster import MiniBatchKMeans

# from sklearn.cluster import AffinityPropagation



# affninityPr = AffinityPropagation (n_clusters = 2).fit(X)

# test_pred = minikmeans.fit_predict(X)

# y_train = pd.read_csv('test.csv')['Satisfied']

# if(accuracy(y_train,pre))

# test_pred
final = pd.DataFrame(pd.read_csv("../input/eval-lab-3-f464/test.csv")["custId"], columns = ['custId'])

final["Satisfied"] = test_pred

final.to_csv("../output/semifinal.csv", index=False)
final.head()