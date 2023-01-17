# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#%% Importing Libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import svm

import os
# %% Importing dataset



os.chdir('/kaggle/input/iris-flower-dataset/')

data = pd.read_csv('IRIS.csv')

data.head(10)
# %% Visualising Feature



sns.lmplot('sepal_length','petal_length',data=data,palette='Set2',hue='species',fit_reg=False,scatter_kws={"s": 70})

# %% Feature 



emoFeature = data[['sepal_length','petal_length']].as_matrix()

type_label = np.where(data['species']=='Iris-setosa', 0, 1)

emoFeature1 = data.columns.values[1:].tolist()

emoFeature1
# %% Model Prediction



model = svm.SVC(kernel='linear')

model.fit(emoFeature,type_label)

model
# %% Hyperplane

# Get the separating hyperplane

w = model.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(1, 12)

yy = a * xx - (model.intercept_[0]) / w[1]



# Plot the parallels to the separating hyperplane that pass through the support vectors

b = model.support_vectors_[0]

yy_down = a * xx + (b[1] - a * b[0])

b = model.support_vectors_[-1]

yy_up = a * xx + (b[1] - a * b[0])
# %% Ploting the Hyperplane



sns.lmplot('sepal_length','petal_length', data=data, hue='species', palette='Set1', fit_reg=False)

plt.plot(xx, yy, linewidth=2, color='black');

plt.plot(xx, yy_up,'k--');

plt.plot(xx, yy_down,'k--');
# %% Function 



def flower(sepal,petal):

    if(model.predict([[sepal,petal]]))==0:

        print("You're looking for Iris-setosa")

    else:

        print("May be Iris-virginica or Iris-versicolor")
# %% User Input of values



# a = float(input("Enter Sepeal Length: "))

# b = float(input("Enter Petal Length: "))

# flower(a,b)
#%% User Input



flower(1.4,2)