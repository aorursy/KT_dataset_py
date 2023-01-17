# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%pylab inline

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler 

from sklearn.neural_network import MLPRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_excel("../input/Planon.xlsx")
data.head(5)
#transform from date to float

j = 0

for i in data["REGISTRATIONDATE"]:

    data.loc[j,"REGISTRATIONDATE"] = i.timestamp()

    j += 1
data.head()
data.describe()
Target = data["CONSUMPTION"]

Data = data.drop(["CONSUMPTION"], axis=1).values
std_scale_data = StandardScaler()
std_scale_data.fit(Data)
Data = std_scale_data.transform(Data)
from sklearn import decomposition



pca = decomposition.PCA(n_components=3)

pca.fit(Data)

Data = pca.transform(Data)
print (pca.explained_variance_ratio_)

print (pca.explained_variance_ratio_.sum())
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Data,Target,test_size=0.2)
X_train.shape