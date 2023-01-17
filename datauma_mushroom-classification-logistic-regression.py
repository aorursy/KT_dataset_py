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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

data.head()
print("The Shape of the dataset: {} ".format(data.shape))
print("Summary of the data: {}\n".format(data.info()))
print("Summary of the data: {}\n".format(data.describe()))
print("The Missing values in the dataset are : {}".format(data.isnull().sum()))
data.describe(include='all').transpose()
#dropping variable veil_type

data = data.drop(["veil-type"], axis = 1)
#seperating the dependant and independant variables.

features = data.columns

target = 'class'

features = list(features.drop(target))

features
# There are 21 variables, so a plot is divided into 11 rows and 2 columns.

fig, axs = plt.subplots(nrows=11, ncols=2, figsize=(11, 66))



for f, ax in zip(features, axs.ravel()):

    sns.countplot(x=f, hue='class', data=data, ax = ax)
#Converting categories columns to number column(Label encoding)



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for col in data.columns:

    data[col] = le.fit_transform(data[col])

data.describe().transpose()
plt.figure(figsize=(14,12))

sns.heatmap(data.corr(),linewidths=.1,cmap="GnBu", annot=True)
X = data.drop('class', axis=1)

y = data['class']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=0)
data.head()
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print("Accuracy score is {}".format(lr.score(X_test_scaled, y_test)))
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score



#Let's see how our model performed

print(classification_report(y_test, y_pred))
#Confusion matrix for the LR classification

print(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import cross_val_score



#Now lets try to do some evaluation using cross validation.

LR_eval = cross_val_score(estimator = lr, X = X_train_scaled, y = y_train, cv = 10)

LR_eval.mean()
