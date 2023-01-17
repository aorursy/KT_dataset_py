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
data=pd.read_csv("/kaggle/input/hard-drive-failure-data/hard_drive_failure_data.csv")

data.head()
capacities=np.array(data["capacity_bytes"]).astype(int)

max(capacities)
suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

def humansize(nbytes):

    i = 0

    while nbytes >= 1024 and i < len(suffixes)-1:

        nbytes /= 1024.

        i += 1

    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')

    return '%s %s' % (f, suffixes[i])
humansize(max(capacities))
humansize(min(capacities))
lifure=data[["lifetime","failure"]]

lifure.corr()
print(data["failure"].isna().sum())

print(data["failure"].isnull().sum())
set(data["failure"])
expected=data.failure.shape[0]

actual=np.array(data["failure"]).astype(int).sum()

if expected==actual:

    print("All Values is 1")

else:

    print("Something else")
del data["failure"]
data.head()
set(data["@version"])
del data["@version"]
data.head()
models=data["model"]

for model in models:

    print(model)
models2letter=[]

for model in models:

    models2letter.append(model[0:2])

    
set(models2letter)
data["newmodels"]=pd.Series(models2letter)

data.head()
pd.Series(models2letter)
%matplotlib inline

import seaborn as sns

sns.countplot(data=data,x="newmodels")
data.lifetime.describe()
lifetime0=data[data["lifetime"]==0]

lifetime0
sns.countplot(data=lifetime0,x="capacity_bytes")
humancapacities=pd.Series(capacities).apply(humansize)
data["humancapaci"]=humancapacities

data.head()
lifetime0=data[data["lifetime"]==0]

sns.countplot(data=lifetime0,x="humancapaci")
set(lifetime0["model"])
bestdrives=data[data["lifetime"]>600]

bestdrives
sns.countplot(data=bestdrives,x="newmodels")
#bakalim[5:7]

month=[]

dates=data["date"]

for date in dates:

    month.append(date[5:7])
sns.countplot(month)