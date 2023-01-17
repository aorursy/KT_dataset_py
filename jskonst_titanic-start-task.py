import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head(10)
data.describe()
data.loc[0:4]
data[["Name", "Age"]].loc[:3]
data["Sex"].value_counts()
total = data.shape[0]
# n_surv = data["Survived"].value_counts()[0]

# surv = data["Survived"].value_counts()[1]

n_surv,surv= data["Survived"].value_counts()

print(np.round(surv*100/total,2))
f_class = data["Pclass"].value_counts()[1]

print(np.round(f_class*100/total,2))
data["Pclass"].hist();
print(data["Age"].mean())

print(data["Age"].median())
data["SibSp"].corr(data["Parch"])
names = data[data["Sex"]=="female"]["Name"]
def filter_names(name):

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index("Miss.")

        return lst[idx+1]

    if "(" in name:

        idx = name.find("(")

        return name[idx+1:-1].split()[0]
f_names= names.apply(filter_names)
f_names.value_counts()
data.head()
pdata = data[["Survived","Pclass","Sex","Age"]]

pdata.head()