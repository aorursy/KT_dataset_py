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
import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
print(df.shape)
df.head()
df.info()
df.describe()
class_names ={0:'Not Fraud',1:'Fraud'}

print(df.Class.value_counts().rename(index=class_names))
#Plotting the variables using subplots
fig=plt.figure(figsize=(15,12))
fig = plt.figure(figsize = (15, 12))



plt.subplot(5, 6, 1) ; plt.plot(df.V1) ; plt.subplot(5, 6, 15) ; plt.plot(df.V15)

plt.subplot(5, 6, 2) ; plt.plot(df.V2) ; plt.subplot(5, 6, 16) ; plt.plot(df.V16)

plt.subplot(5, 6, 3) ; plt.plot(df.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df.V17)

plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)

plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)

plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)

plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)

plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)

plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)

plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24)

plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25)

plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26)

plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27)

plt.subplot(5, 6, 14) ; plt.plot(df.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df.V28)

plt.subplot(5, 6, 29) ; plt.plot(df.Amount)

plt.show()
from sklearn.model_selection import train_test_split
feature_names = df.iloc[:, 1:30].columns

target = df.iloc[:1, 30: ].columns

print(feature_names)

print(target)
data_features =df[feature_names]

data_target =df[target]



print(df[feature_names])
print(df[target])
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)

print("Length of X_train is: {X_train}".format(X_train = len(X_train)))

print("Length of X_test is: {X_test}".format(X_test = len(X_test)))

print("Length of y_train is: {y_train}".format(y_train = len(y_train)))

print("Length of y_test is: {y_test}".format(y_test = len(y_test)))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
model = LogisticRegression()

model.fit(X_train, y_train.values.ravel())
pred =model.predict(X_test)
class_names = ['not_fraud', 'fraud']

matrix = confusion_matrix(y_test, pred)

# Create pandas dataframe

dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

# Create heatmap

sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')

plt.title("Confusion Matrix"), plt.tight_layout()

plt.ylabel("True Class"), plt.xlabel("Predicted Class")

plt.show()