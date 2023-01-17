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
%ls "../input/titanic"
pwd
import pandas as pd
df = pd.read_csv("../input/titanic/train.csv")
titanic_dataset = df.copy()
titanic_dataset.shape
df.describe()
df.head()
del df["Name"]
df.head()
df2 = pd.read_csv("../input/titanic/gender_submission.csv")
def f(str):
    if str=="male":
        return 0
    else:
        return 1
df["Gender"]  = df.Sex.apply(f)
del df["Sex"]
df.head()
df.tail()
del df["Fare"], df["Ticket"], df["Embarked"]
df.head()
df.isnull().sum()
a = df[df.Survived==0]
a.Age.fillna(a.Age.mean())
#print(a.Age.mean())
b = df[df.Survived==1]
b.Age.fillna(b.Age.mean())
#print(b.Age.mean())
df.isnull().sum()
df.tail()
df.Cabin.fillna("X00", inplace=True)
df.head()
df.head(200)
import matplotlib.pyplot as plt
plt.scatter(df.Survived, df.Age, alpha=0.5)
plt.show()
df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5) 
df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha = 0.5)
for x in [1,2,3]:    ## for 3 classes
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Age wrt Pclass")
plt.legend(("1st","2nd","3rd"))
