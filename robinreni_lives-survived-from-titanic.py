import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display



#Display For Notebooks

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv('../input/train.csv',sep=',')

display(train.head())

test=pd.read_csv('../input/test.csv',sep=',')

display(test.head())
train.isnull().sum()
def describe_dataset(dataset,threshold=0.90):

    #Threshold value limits the Dataset

    da=dataset.isnull().sum(axis=0).reset_index()

    da.columns=['feature_name','missing_value']

    da['missing_ratio']=da['missing_value']/dataset.shape[0]

    return da
miss_data=describe_dataset(train)

ds=miss_data.sort_values("missing_value",ascending=False)

display(ds)
miss_value=describe_dataset(test)

da=miss_value.sort_values("missing_value",ascending=False)

display(da)
sns.factorplot(x="Sex", y="Age", hue="Survived", data=train, size=5,kind="bar", palette="muted")

plt.show()
fig = plt.figure(figsize=(8,8))

sns.violinplot(x="Sex", y="Pclass", hue="Survived", data=train, split=True, scale="count")

plt.show()
fig = plt.figure(figsize=(8,8))

sns.violinplot(x="Embarked", y="Pclass", hue="Survived", data=train, split=True, scale="count")

plt.show()
features = list(train.columns.values)

features.remove('PassengerId')

features.remove('Name')

corr=train.loc