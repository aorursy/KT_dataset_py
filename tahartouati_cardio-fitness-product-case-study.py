import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
df.head()
df.info()
y = pd.get_dummies(df.Product, prefix='Product')
print(y.head())
df['TM195'] = y['Product_TM195']
df['TM498'] = y['Product_TM498']
df['TM798'] = y['Product_TM798']
df.head()
df['Gender'].replace('Male',1,inplace=True)
df['Gender'].replace('Female',0,inplace=True)
df.head()


TM195 = df[df['TM195'] == 1]
TM498 = df[df['TM498'] == 1]
TM798 = df[df['TM798'] == 1]
TM195.shape[0],TM498.shape[0],TM798.shape[0]
# Dropping redundant variables:
TM195.drop(['TM498','TM798','TM195'],axis=1,inplace = True)
TM498.drop(['TM195','TM798','TM498'],axis=1,inplace = True)
TM798.drop(['TM195','TM498','TM798'],axis=1,inplace = True)
df.corr()[['Age','Gender','Education','Usage','Fitness','Income','Miles']].iloc[[-3,-2,-1]]
numTM195 = TM195[['Age','Education','Usage','Fitness','Income']]
numTM498 = TM498[['Age','Education','Usage','Fitness','Income']]
numTM798 = TM798[['Age','Education','Usage','Fitness','Income']]
def plotdist (col):
    print('Distribution and range for column :',col.name)
    print(col.hist(bins=12, alpha=0.5))
    plt.show()
    print('range of mean +- 2 standard deviation: [',col.mean() - 2* col.std(),',',col.mean() + 2* col.std(),']');
    print('Median ',col.name,': ',col.median());
    print('Mean ',col.name,': ',col.mean());
    print('__________________________________________________________________________')
    print()
TM195[['Age','Income','Miles']].apply(plotdist)
TM498[['Age','Income','Miles']].apply(plotdist)
TM798[['Age','Income','Miles']].apply(plotdist)
TM195.head()
TM195[['MaritalStatus']].mode()
def modvalue(col):
    print('Mode value for column :',col.name);
    print('--------------------------------')
    print('First Mode value/ Most occurring value :',col.mode());
    print('__________________________________________________________________________')
    print()
TM195[['Gender','Education','MaritalStatus','Usage','Fitness']].apply(modvalue)
TM498[['Gender','Education','MaritalStatus','Usage','Fitness']].apply(modvalue)
TM798[['Gender','Education','MaritalStatus','Usage','Fitness']].apply(modvalue)
