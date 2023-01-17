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
# Read the data

Data = pd.read_csv("../input/datasetknn/DataSetKNN.csv")

Data.head()
X = Data.iloc[:,0:-1]

target = Data.iloc[:,-1]    # separete attributes and target in other place
Data.isna().sum()
X.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.boxplot(Data["TARGET CLASS"])   # check if data is  bias or not
col_names = X.columns
plot = sns.pairplot(X)

plot.map_lower(sns.kdeplot)

sns.heatmap(X.corr())     # check the correlation of attributes with each other
# Scaling the data into standard form



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

scaled_feature = scaler.transform(X)

Scaled_Data = pd.DataFrame(scaled_feature,columns=col_names)

Scaled_Data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Scaled_Data,target,test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
# Check the error in prediction on the basis of neighboring value



error_rate = []



# Will take some time

for i in range(1,50):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

    

min_error= min(error_rate)

n = error_rate.index(min_error)

print("optimium n_neighbors value is = ",  n) 
# prediction using optimium n_neighbors value

knn = KNeighborsClassifier(n_neighbors=n)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report



print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))