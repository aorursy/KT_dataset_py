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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head(10)
data.describe()
data ["Sex"].value_counts()
total = data.shape

print(total)
total = data.shape[0]

print(total)
data["Survived"].value_counts()
n_alive,alive = data["Survived"].value_counts()

p_alive = alive*100/total

print(n_alive,alive)

print(np.round (p_alive,2))
data["Pclass"].value_counts()
data["Pclass"].value_counts()[1]
fc_pass = data["Pclass"].value_counts()[1]

p_fclass = fc_pass*100/total

print(p_fclass)
fc_pass = data["Pclass"].value_counts()[1]

p_fclass = fc_pass*100/total

print(np.round(p_fclass,2))
data["Pclass"].hist()
print(data["Age"].mean())

print(data["Age"].median())
data["SibSp"].corr(data["Parch"])
data[["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare"]]
numeric = data[["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare"]].corr()
import seaborn as sns 

%matplotlib inline
sns.heatmap(numeric,annot=True);
numeric[(numeric>0.2)]

#df[[len(x) <2)for x in df[ column nfme]]]

[x for x in numeric]

for x in numeric:

 print(numeric[(numeric>0.2)].loc[x])

        #df[df['column name'].map(len)<2]

        #f = df.drop(df[df.score<50].index)
numeric[(numeric>0.2)]
import seaborn as sns 

%matplotlib inline
sns.heatmap(numeric[(numeric>0.2)],annot=True);
n = numeric[(numeric>0.2)]

#df[[len(x) <2)for x in df[ column nfme]]]

[n[x] > 0 for x in numeric]

#df[df['column name'].map(len)<2]

#f = df.drop(df[df.score<50].index)
!pip install numpy

!pip install pandas
names = data["Name"]
for name in names:

    print(name)
names = data[data["Sex"]=="female"]["Name"]
for name in names:

    print(name)
for name in names:

    if "Miss." in name:

        print(name)
for name in names:

    if "Miss." in name:

        ist = name.split(" ")

        print(ist)
for name in names:

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index('Miss.')

        print(lst[idx+1])
for name in names:

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index('Miss.')

        continue

        #print(lst[idx+1])

    if "(" in name:

        idx = name.find("(")

        print(name[idx:])
for name in names:

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index('Miss.')

        continue

        #print(lst[idx+1])

    if "(" in name:

        idx = name.find("(")

        print(name[idx+1:-1].split(" ")[0])
for name in names:

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index('Miss.')

        continue

        #print(lst[idx+1])

    if "(" in name:

        idx = name.find("(")

        #print(name[idx+1:-1].split(" ")[0])

        continue

    print(name)
def filter_names(name):

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index('Miss.')

        return lst[idx+1]

    if "(" in name:

        idx = name.find("(")

        return name[idx+1:-1].split(" ")[0]

   # print(name)
names.apply(filter_names)
names_chart = names.apply(filter_names).value_counts()
names_chart