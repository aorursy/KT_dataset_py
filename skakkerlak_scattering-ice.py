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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
import seaborn as sns


sns.jointplot(x='Age',y='Fare',data=train)
sns.jointplot(x='Age',y='Fare',data=train[train['Fare']<500])
import sklearn.cluster as cluster

firstclass=train[train['Pclass']==1]

nonan=train[train['Age']>=0]

kmeans=cluster.KMeans(n_clusters=10).fit(nonan[['Age','Fare']])


import matplotlib.pyplot as plt

label=pd.DataFrame(kmeans.labels_)

nonan=nonan.assign(Label=label)

g=sns.FacetGrid(nonan,col='Label')

g.map(plt.scatter,'Age','Fare')
g=sns.pairplot(nonan[['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
g=sns.pairplot(nonan[nonan['Fare']<200][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
g=sns.pairplot(nonan[nonan['Fare']<100][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
g=sns.pairplot(nonan[nonan['Fare']<50][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
g=sns.pairplot(nonan[nonan['Fare']<25][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
south=nonan[nonan['Embarked']=='S']

g=sns.pairplot(south[south['Fare']<50][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
cher=nonan[nonan['Embarked']=='C']

g=sns.pairplot(cher[cher['Fare']<50][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")
queen=nonan[nonan['Embarked']=='Q']

g=sns.pairplot(queen[queen['Fare']<50][['Age','Fare','Pclass']],hue='Pclass',diag_kind="hist")