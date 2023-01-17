# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dataset = pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#dataset.drop(axis=1,columns = 'Unnamed: 32', inplace= True)

dataset.head(n = 20)# this will give a preview of the first 20 patients
sns.countplot(x = 'diagnosis', data = dataset).set_title('Bar plot for the two diagnosis ')



#Selective data set



dataset_select = dataset.iloc[:, 1:7]

dataset_select.head(n = 10)
sns.pairplot(dataset_select, hue = 'diagnosis')



sns.lmplot('radius_mean', 'texture_mean', hue = 'diagnosis', data = dataset, fit_reg = False)



# Importing the Logistic Regression class from sklearn

from sklearn.linear_model import LogisticRegression

#Creating the X_train and y_train

X_train =  dataset.loc[:,'radius_mean':'texture_mean'].values

y_train = dataset.loc[:, 'diagnosis'].values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
#Encoding the diagnosis 

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

y_train = label.fit_transform(y_train)
#Logistic classification



classifier = LogisticRegression(max_iter=1000, C=1E5)

classifier.fit(X_train, y_train)
#Creating a mesh for visulization

h = .01 #mesh step size

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5

y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5

x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = classifier.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
Z = Z.reshape(x_mesh.shape)

plt.figure(figsize=(12, 9))

plt.pcolormesh(x_mesh, y_mesh, Z, cmap=plt.cm.Paired)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('radius_mean')

plt.ylabel('texture_mean')

plt.show()