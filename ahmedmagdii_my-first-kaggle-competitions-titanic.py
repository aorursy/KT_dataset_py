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




import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Data processing



dataset_train = pd.read_csv('../input/titanic/train.csv')

dataset_test = pd.read_csv('../input/titanic/test.csv')

X =dataset_train.iloc[:,[2,4,5,6,7,10]].values

X_test2 = dataset_test.iloc[:,[1,3,4,5,6,9]].values



#processing Missing Data



from sklearn.impute import SimpleImputer

imputer_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer_most = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

X[:,[2]]=imputer_mean.fit_transform(X[:,[2]])

X_test2[:,[2]]=imputer_mean.fit_transform(X_test2[:,[2]])

X[:,[5]] = imputer_most.fit_transform(X[:,[5]])

X_test2[:,[5]]=imputer_most.fit_transform(X_test2[:,[5]])



# New Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()

X[:,1] = labelencoder_X.fit_transform(X[:,1])

X_test2[:,1]=labelencoder_X.fit_transform(X_test2[:,1])

X[:,5] = labelencoder_X.fit_transform(X[:,5])

X_test2[:,5]=labelencoder_X.fit_transform(X_test2[:,5])

Y = dataset_train.iloc[:,1].values



# Test & Training

from sklearn.model_selection import train_test_split

X_train,X_test1,Y_train,Y_test = train_test_split(X,Y,test_size =0.05)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test1 = sc.transform(X_test1)

X_test2 = sc.transform(X_test2)

#Classification

from sklearn.svm import SVC

Classifier = SVC(kernel='rbf',random_state=(0))

Classifier.fit(X_train,Y_train)



Y_pred = Classifier.predict(X_test2)

print(Y_pred)



#output

Y_pred = pd.Series(Y_pred)

Y_pred.to_csv('/kaggle/working/Y_pred.csv')