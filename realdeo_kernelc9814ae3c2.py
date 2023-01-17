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
df = pd.read_csv("../input/BlackFriday.csv")
df.head(10)
customer_data = df.groupby(['User_ID'])
customer_data.groups.keys()
customer_data.groups[1000001]
hasilnya = df.iloc[customer_data.groups[1000001],:]

hasilnya.tail(10)
tabulasi = customer_data.agg({'Purchase':'sum','Product_ID':'count'})

tabulasi.head(10)
biodata = customer_data.first()[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]

biodata.head(10)
combined =  tabulasi.join(biodata)

combined.head()
dataKeacak = combined.sample(frac = 1)

dataKeacak.head(10)
sudahDiOneEncoding = pd.get_dummies(dataKeacak, columns=['Age', 'Occupation', 'City_Category', 

                                  'Stay_In_Current_City_Years', 'Marital_Status'])

sudahDiOneEncoding.head(10)
sudahDiOneEncoding['Gender'] = sudahDiOneEncoding['Gender'] == "F"
sudahDiOneEncoding.head(10)
X = sudahDiOneEncoding.drop(["Purchase"], axis=1)

y = sudahDiOneEncoding[['Purchase']]
X.head(10)
from sklearn.linear_model import LinearRegression as LR

from sklearn.model_selection import KFold as KF

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.preprocessing import StandardScaler
kf = KF(n_splits  = 5, shuffle = True)

for train_index, test_index in kf.split(X):

    scaler = StandardScaler()

    

    train_X = scaler.fit_transform(X.values[train_index])

    train_y = y.values[train_index]

    

    test_X  = scaler.transform(X.values[test_index])

    test_y  = y.values[test_index]

    

    linReg = LR()

    linReg.fit(train_X, train_y)

    prediction = linReg.predict(test_X)

    

    # Evaluasi

    score = MAE(test_y, prediction)

    print(score)
X = sudahDiOneEncoding.drop(["Gender"], axis=1)

y = sudahDiOneEncoding[['Gender']]
X.head(10)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import log_loss
kf = KF(n_splits  = 5, shuffle = True)

for train_index, test_index in kf.split(X):

    

    train_X = X.values[train_index]

    train_y = y.values[train_index]

    

    test_X  = X.values[test_index]

    test_y  = y.values[test_index]

    

    linReg = DecisionTreeClassifier( min_samples_leaf=100)

    linReg.fit(train_X, train_y)

    prediction = linReg.predict_proba(test_X)

    

    # Evaluasi

    score = log_loss(test_y, prediction)

    print(score)
prediction[:50]