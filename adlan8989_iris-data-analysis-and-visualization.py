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
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
df = pd.read_csv("../input/Iris.csv")



df.head()
print('Rows :',df.shape[0])

print('Columns :',df.shape[1])
df['Species'].value_counts()
sns.pairplot(df,hue='Species',size=2)
plt.subplot(1, 2, 1)

plt.scatter(x = df['SepalLengthCm'], y = df['SepalWidthCm'], color = 'red', marker = 'o',)

plt.title('Length vs Width')

plt.xlabel('Sepal Length in cm')

plt.ylabel('Sepal Width in cm')



plt.subplot(1, 2, 2)

plt.scatter(x = df['SepalLengthCm'], y = df['PetalLengthCm'], color = 'blue', marker = 'x',)

plt.title('Sepal vs Petal')

plt.xlabel('Sepal Length in cm')

plt.ylabel('petal Length in cm')

plt.show()
features = list(df.columns)



print(features)
features.remove('Id')

features.remove('Species')



print(features)


Y = df.Species

X = df[features].values.astype(np.float32)



print(X.shape)

print(Y.shape)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



# feeding the into the scaler



x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



model = SVC()



model.fit(x_train,y_train)



y_pred = model.predict(x_test)



print("training accuracy :", model.score(x_train, y_train))

print("testing accuracy :", model.score(x_test, y_test))



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

print(cm)


