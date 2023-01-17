import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline
data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv');

data.head(3)
data.info()
data.isnull().sum()
data["salary"].describe()
sns.distplot(data["salary"])

plt.xlim([100000,940000])
data.loc[data["salary"].isnull()]["status"].describe()
data["salary"].fillna(0.0,inplace=True)
categorical=[]

numerical=[]

for colname, coltype in data.dtypes.iteritems():

    if(coltype == "object"):

        categorical.append(colname)

        continue

    if(coltype == "float64"):

        numerical.append(colname)

        continue

sns.pairplot(data)
sns.scatterplot(x=data["salary"],y=data["sl_no"]);
#numerical



for i in numerical:

  print(i)

  print(data[i].describe())

  print()
temp=data["mba_p"]

for i in numerical:

  if (i == "salary"  or i == "mba_p"):

    continue

  temp=temp + data[i]

temp=temp * 0.2

data["avg_score"]=temp
fig = plt.figure(figsize=(10,6))



sns.distplot(data.loc[data["status"] == "Placed"]["avg_score"],label="Placed")

sns.distplot(data.loc[data["status"] == "Not Placed"]["avg_score"],label="Not Placed")

fig.legend(labels=['Placed','Not Placed'])

plt.show()

fig,ax = plt.subplots(2,3,figsize=(19,9))

index=0

for i in range(2):

  for j in range(3):

    sns.distplot(data.loc[data["status"] == "Placed"][numerical[index]],label="Placed",ax=ax[i][j])

    sns.distplot(data.loc[data["status"] == "Not Placed"][numerical[index]],label="Not Placed",ax=ax[i][j])

    ax[i][j].legend(labels=['Placed','Not Placed'])

    # ax[i][j].set_title(numerical[index])

    index+=1

fig.delaxes(ax[1][2])

plt.tight_layout()

# plt.show()

sns.countplot(data["gender"],hue=data["status"])

#so the data shows male got placed
plt.figure(figsize =(10,6))

sns.boxplot("salary", "gender", data=data)

plt.show()


sns.countplot(data["specialisation"],hue=data["status"])

plt.figure(figsize =(10,6))

sns.boxplot("salary", "specialisation", data=data.loc[data["status"]=="Placed"])

plt.show()


fig,ax=plt.subplots(1,2,figsize=(15,7))



sns.countplot(data.loc[data["status"]=="Placed"]["workex"],ax=ax[0])

ax[0].set_title("PLACED");

total=len(data.loc[data["status"]=="Placed"]["workex"])



for p in ax[0].patches:

    percentage = '{:.1f}%'.format(100 * p.get_height()/total)

    x = p.get_x() + p.get_width() / 2 - 0.05

    y = p.get_y() + p.get_height()

    ax[0].annotate(percentage, (x, y), size = 19)

sns.countplot(data.loc[data["status"]=="Not Placed"]["workex"],ax=ax[1])

ax[0].set_title("PLACED"+" ( Total "+str(total)+" )");



total=len(data.loc[data["status"]=="Not Placed"]["workex"])

ax[1].set_title("NOT PLACED"+" ( Total "+str(total)+" )");



for p in ax[1].patches:

    percentage = '{:.1f}%'.format(100 * p.get_height()/total)

    x = p.get_x() + p.get_width() / 2 - 0.05

    y = p.get_y() + p.get_height()

    ax[1].annotate(percentage, (x, y), size = 19)

plt.tight_layout()



ax[0].get_yaxis().set_visible(False)

ax[1].get_yaxis().set_visible(False)



plt.figure(figsize =(10,6))

sns.boxplot("salary", "workex", data=data.loc[data["status"]=="Placed"])

plt.show()
#degree_t



sns.countplot(data["degree_t"],hue=data["status"]);

plt.figure(figsize =(10,6))

sns.boxplot("salary", "degree_t", data=data.loc[data["status"]=="Placed"])

plt.show()