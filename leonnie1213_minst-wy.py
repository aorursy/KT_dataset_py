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
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:, 0]
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = MLPClassifier().fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(df_test)
y_predict = model.predict(X_test_scaled)
df_predict = pd.DataFrame(y_predict, index=df_test.index)
df_predict.index.names = ['ImageID']
df_predict.rename(columns={0:'Label'}, inplace=True)
df_predict.to_csv('WY_MINST')
