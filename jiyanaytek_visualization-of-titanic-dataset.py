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
import seaborn as sns
from matplotlib import pyplot as plt
#%matplotlib qt
# pip install PyQt5
#from IPython.display import Image
%matplotlib inline 
#there are 12 variable of titanic dataset
data=pd.read_csv("/kaggle/input/titanic/train.csv", sep=",")
data.head(5)
data.Survived.value_counts()
data.Fare.value_counts().head(10)
data.info()
data.dtypes
data.size
data.shape
data.describe().T
type(data)
data.count()
data.isnull().values.any()
data.isnull().sum()
a=sns.barplot(x="Sex", y="Survived", hue="Sex", data=data);
a.set_title("Cinsiyete Göre Survived Dağılımı");
sns.catplot(x="Pclass", y="Fare", kind="violin", hue="Pclass", col="Sex", orient="v", data=data);
sns.catplot(x="Pclass", y="Fare", kind="bar", hue="Pclass", col="Sex", orient="v", data=data);
sns.distplot(data.Pclass);
#kdeplot
sns.kdeplot(data["Fare"]).set_title("Fare Distribution");
sns.pairplot(data)
import matplotlib.pyplot as plt
corr=data.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, vmax=.8, linewidths=0.05,square=True,annot=True,linecolor="pink");
plt.style.use("fivethirtyeight")
plt.bar(data.Age, data.Fare, color="#444444")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Fare Distribution By Age")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
ages=data["Age"]
fare=data["Fare"]
plt.style.use("classic")
plt.hist(ages, color="#B0E0E6")
plt.title("Fare Plot By Age")
plt.xlabel("ages")
plt.ylabel("fare")
plt.tight_layout()
plt.show();
