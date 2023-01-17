# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mlt # for visualising dataset

import matplotlib.pyplot as plt # library for generating plots

import seaborn as sns #for plots and other graphs



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
o=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
print(o)
print(o.describe())
print(o.info())
print(o.isnull())
print(o.isnull().any())
a=o['sepal_length']

b=o['sepal_width']

c=o['petal_length']

d=o['petal_width']



plt.style.use('Solarize_Light2')



fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

fig.suptitle('Categorical Plotting')

axs[0].hist(a)

axs[1].hist(b)

axs[2].hist(c)

axs[3].hist(d)

plt.style.use('grayscale')



fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

fig.suptitle('Categorical Plotting')

axs[0].scatter(a,b)

axs[1].scatter(c,d)
plt.style.use('ggplot')



fig, axs = plt.subplots(1, 2, figsize=(20, 20), sharey=True)

fig.suptitle('Categorical Plotting')

axs[0].plot(a,b)

axs[1].plot(c,d)
#covariance among different parameters

print(o.cov())
#correlation to show different parameters depend on each other

print(o.corr())
plt.style.use('grayscale')



fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

fig.suptitle('Categorical Plotting')

axs[0].scatter(a,c)

axs[1].scatter(b,c)
#heatmap for correlation matrices

corr_matrix=o.corr()

sns.heatmap(corr_matrix,annot=True)
plt.style.use('default')

f=sns.pairplot(o,hue='species')
o
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
o['species']=le.fit_transform(o['species'])
o['species'].unique()
o
X_train = o.drop("species", axis=1)             # drop labels for training set

y_train = o["species"].copy()
X_train
y_train
from sklearn.linear_model import LogisticRegression



softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",random_state=42)

softmax_reg.fit(X_train,y_train)
y_pred_soft = softmax_reg.predict(X_train)
y_pred_soft
from sklearn.metrics import accuracy_score



accuracy_score(y_train,y_pred_soft)

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_train,y_pred_soft)

cnf_matrix
from sklearn.metrics import classification_report

print(classification_report(y_train,y_pred_soft))
from sklearn.metrics import f1_score



f1_score(y_train, y_pred_soft,average = 'weighted')
submission = pd.DataFrame({'species':o['species'],'species':y_pred_soft})

submission.to_csv("iris_kaggle",index=False)
