import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.ticker as mtick

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv') #let's read the training data set
df.info()
df.head()
df.mean()['Survived'] #the average survival rate in total
df.drop(columns = ['PassengerId','Name','Ticket','Cabin'], inplace = True) #remove some columns will not be used in the analysis
df.groupby('Pclass').mean() #let's see the properties of each class
Pclass = df['Pclass'].value_counts().sort_index(ascending = False)

import matplotlib.pyplot as plt

plot = Pclass.plot.barh()

#plot = (Pclass * 100 /Pclass.sum()).plot.barh()

#plot.xaxis.set_major_formatter(mtick.PercentFormatter())
def cat_chart(i):

    x = df.groupby(['Pclass',i]).count().max(axis= 1).reset_index().pivot(index='Pclass', columns=i, values=0).sort_index(ascending = False).sort_values(i, axis=1, ascending=False)

    x.plot.barh(stacked = True);

    print((x.T * 100 /x.T.sum()).T)

    return  None
for i in ['Survived', 'Sex']:

    cat_chart(i)
for i in ['Fare','Age']:

    df.boxplot(column=i,by = 'Pclass',figsize=[10,5])
col = df.Survived.map({0:'b', 1:'r'})



df.plot.scatter(x ='Age' , y= 'Fare',c=col)