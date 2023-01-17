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
dataset = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
dataset.head()
dataset.isnull().sum()
dataset.info()
d = dataset['density']

bins = 10





plt.hist(d,bins)

plt.title("histogram (1)")

plt.xlabel("density(water)")

plt.ylabel("Frequency")

plt.show()
r = dataset['residual sugar']

bins = 10





plt.hist(r,bins,color = "green")

plt.title("histogram (2)")

plt.xlabel("residual sugar(wine)")

plt.ylabel("Frequency")

plt.show()
fsd = dataset['free sulfur dioxide']

bins = 10





plt.hist(fsd,bins,color = "red")

plt.title("histogram (3)")

plt.xlabel(" free form of SO2")

plt.ylabel("Frequency")

plt.show()
ph = dataset['pH']

bins = 10





plt.hist(ph,bins,color = "yellow")

plt.title("histogram (4)")

plt.xlabel(" free form of SO2")

plt.ylabel("Frequency")

plt.show()
cl = dataset['chlorides']

bins = 10





plt.hist(cl,bins,color = "skyblue")

plt.title("histogram (5)")

plt.xlabel("amount of salt in wine")

plt.ylabel("Frequency")

plt.show()
for cl in dataset:

    plt.figure()

    dataset.boxplot([cl])
d = dataset["density"]

ph = dataset["pH"]



plt.scatter(d, ph)

plt.title("Scatter Plot (1)")

plt.xlabel("density (water)")

plt.ylabel("pH (wine)")

plt.show()
fa = dataset["fixed acidity"]

va = dataset["volatile acidity"]



plt.scatter(fa, va,color = 'green')

plt.title("Scatter Plot (2)")

plt.xlabel("nonvolatile acid (wine)")

plt.ylabel("acetic acid (wine)")

plt.show()
ca = dataset["citric acid"]

va = dataset["volatile acidity"]



plt.scatter(ca, va,color = 'red')

plt.title("Scatter Plot (3)")

plt.xlabel("citric acid (wine)")

plt.ylabel("acetic acid (wine)")

plt.show()
sl = dataset["sulphates"]

ph = dataset["pH"]



plt.scatter(sl, ph,color = 'yellow')

plt.title("Scatter Plot (4)")

plt.xlabel("sulphur di-oxide gas (water)")

plt.ylabel("pH (wine)")

plt.show()
fsd = dataset["free sulfur dioxide"]

tsd = dataset["total sulfur dioxide"]



plt.scatter(fsd, tsd,color = 'skyblue')

plt.title("Scatter Plot (5)")

plt.xlabel("free sulfur dioxide (water)")

plt.ylabel("total sulfur dioxide(wine)")

plt.show()
features = ['citric acid','alcohol','residual sugar','pH','total sulfur dioxide']  #input values



x = dataset[features]

y = dataset['quality']                                                             #output values
from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()

y= le.fit_transform(y)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.10 , random_state=3,shuffle=True)
s_c = StandardScaler()

train_x=s_c.fit_transform(train_x)

test_x=s_c.fit_transform(test_x)

Knn = KNeighborsClassifier (n_neighbors=7,p=1)

Knn.fit(train_x,train_y)
y_pred=Knn.predict(test_x)

y_pred
test_y
c_m = confusion_matrix(test_y,y_pred)

c_m
print("Wrong values predicted out of total values : ")

print((test_y!=y_pred).sum(),'/',((test_y==y_pred).sum()+(test_y!=y_pred).sum()))
print('percentage Accuracy using KNN is : ',100*accuracy_score(test_y,y_pred))