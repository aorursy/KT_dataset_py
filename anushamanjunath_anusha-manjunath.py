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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
import pandas as pd

data=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

data
x=data.drop("quality",axis=1)

y=data["quality"]

data.info()
data.shape
data.isnull().sum()
data.info()
x.shape
y.shape
d = data['density']

bins = 10





plt.hist(d,bins)

plt.title("histogram (1)")

plt.xlabel("density(water)")

plt.ylabel("Frequency")

plt.show()
r = data['residual sugar']

bins = 10





plt.hist(r,bins)

plt.title("histogram (2)")

plt.xlabel("residual sugar(wine)")

plt.ylabel("Frequency")

plt.show()
fsd = data['free sulfur dioxide']

bins = 10





plt.hist(fsd,bins)

plt.title("histogram (3)")

plt.xlabel(" free form of SO2")

plt.ylabel("Frequency")

plt.show()
ph = data['pH']

bins = 10





plt.hist(ph,bins)

plt.title("histogram (4)")

plt.xlabel(" free form of SO2")

plt.ylabel("Frequency")

plt.show()
cl = data['chlorides']

bins = 10





plt.hist(cl,bins)

plt.title("histogram (5)")

plt.xlabel("amount of salt in wine")

plt.ylabel("Frequency")

plt.show()
for cl in data:

    plt.figure()

    data.boxplot([cl])
d = data["density"]

ph = data["pH"]



plt.scatter(d, ph)

plt.title("Scatter Plot (1)")

plt.xlabel("density (water)")

plt.ylabel("pH (wine)")

plt.show()
fa = data["fixed acidity"]

va = data["volatile acidity"]



plt.scatter(fa, va)

plt.title("Scatter Plot (2)")

plt.xlabel("nonvolatile acid (wine)")

plt.ylabel("acetic acid (wine)")

plt.show()
ca = data["citric acid"]

va = data["volatile acidity"]



plt.scatter(ca, va)

plt.title("Scatter Plot (3)")

plt.xlabel("citric acid (wine)")

plt.ylabel("acetic acid (wine)")

plt.show()
sl = data["sulphates"]

ph = data["pH"]



plt.scatter(sl, ph,)

plt.title("Scatter Plot (4)")

plt.xlabel("sulphur di-oxide gas (water)")

plt.ylabel("pH (wine)")

plt.show()
fsd = data["free sulfur dioxide"]

tsd = data["total sulfur dioxide"]



plt.scatter(fsd, tsd,)

plt.title("Scatter Plot (5)")

plt.xlabel("free sulfur dioxide (water)")

plt.ylabel("total sulfur dioxide(wine)")

plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=48)



from sklearn.neighbors import KNeighborsClassifier

nkn=KNeighborsClassifier()

nkn.fit(x_train,y_train)
from sklearn.neighbors import KNeighborsClassifier

nkn.score(x_test,y_test)


nkn.predict(x_test)
print('percentage Accuracy using KNN is :',100*nkn.score(x_test,y_test))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)
from sklearn.linear_model import LogisticRegression

model.score(x_test,y_test)
model.predict(x_test)
print('percentage Accuracy  is :',100*model.score(x_test,y_test))