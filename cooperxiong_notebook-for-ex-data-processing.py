# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score,train_test_split

df = pd.read_csv("../input/train.csv")
data = df.drop(["Name","DateTime","OutcomeSubtype"],axis=1)
data.dropna(inplace=True)
data.info()
#AnimalID          26710 non-null object
#OutcomeType       26710 non-null object
#AnimalType        26710 non-null object
#SexuponOutcome    26710 non-null object
#AgeuponOutcome    26710 non-null object
#Breed             26710 non-null object
#Color             26710 non-null object
#print data.Color.unique()
#print len(data.Color.unique())
for i,cate in zip(range(len(data.OutcomeType.unique())),data.OutcomeType.unique()):
    data.OutcomeType.replace(cate,i,inplace=True)

data.AnimalType.replace("Dog",0,inplace=True)
data.AnimalType.replace("Cat",1,inplace=True)

for i,cate in zip(range(len(data.SexuponOutcome.unique())),data.SexuponOutcome.unique()):
    data.SexuponOutcome.replace(cate,i,inplace=True)

num = data.AgeuponOutcome.str.extract("(\d+)").astype(int)
unit = data.AgeuponOutcome.str.extract(" (.*)")
for u,n in zip(["years","year","months","month","weeks","week","days","day"],[365,365,30,30,7,7,1,1]):
    unit.replace(u,n,inplace=True) 
age = pd.DataFrame({"age":num.mul(unit)})

data.loc[data.Breed.str.contains("Mix"),"mix"]=1
data.loc[~data.Breed.str.contains("Mix"),"mix"]=0

data.loc[data.Color.str.contains(r"/"),"mottled"]=1
data.loc[~data.Color.str.contains(r"/"),"mottled"]=0

data = pd.concat([data,age],axis=1)
data = data.drop(["AnimalID","AgeuponOutcome","Breed","Color"],axis=1)

data[:20]

#sns.corrplot(data)
#sns.countplot(x="OutcomeType",data=data)
#sns.factorplot(x="OutcomeType", y="age", hue="AnimalType", kind='bar',data=data)
#sns.factorplot(x="SexuponOutcome", y="mix", hue="OutcomeType",kind='bar', data=data)
#sns.barplot(x="SexuponOutcome", y="mix", hue="OutcomeType", data=data);
#sns.boxplot(x="OutcomeType", y="age", data=data)
#sns.kdeplot(data.age,hue="OutcomeType")

#g = sns.FacetGrid(data, hue="OutcomeType",size=4,aspect=3)
#g.map(sns.kdeplot,'age',shade= True)
#plt.show()

sns.corrplot(data)
sns.countplot(x="OutcomeType",data=data)
sns.factorplot(x="OutcomeType", y="age", hue="AnimalType", kind='bar',data=data)
sns.barplot(x="SexuponOutcome", y="mix", hue="OutcomeType", data=data)
sns.boxplot(x="OutcomeType", y="age", data=data)
g = sns.FacetGrid(data, hue="OutcomeType",size=4,aspect=3)
g.map(sns.kdeplot,'age',shade= True)
#X = data[:,1:]
#y = data[:,1]