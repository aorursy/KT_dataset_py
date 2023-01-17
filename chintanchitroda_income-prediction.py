import pandas as pd

import numpy as np

from sklearn import tree

import graphviz

from sklearn.model_selection import cross_val_score
df = pd.read_csv('/kaggle/input/predict-the-income-bi-hack/train.csv')

dft = pd.read_csv('/kaggle/input/predict-the-income-bi-hack/test.csv')

print('train dataset length',len(df),'\ntest dataset length',len(dft))

print(len(df.columns))

print(len(dft.columns))
df.head(3)
ids=dft['ID']

df.drop('ID', axis=1,inplace =True)

dft.drop('ID', axis=1,inplace =True)
df.columns
df = pd.get_dummies(df, columns=['Work', 'Education', 'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Gender','Nationality'])

dft = pd.get_dummies(dft, columns=['Work', 'Education', 'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Gender','Nationality'])
df.drop('Nationality_Holand-Netherlands',axis=1, inplace=True)
print(len(df.columns))

print(len(dft.columns))
df['Income'].value_counts()
df["Income"] = df["Income"].replace({'<=50K':1,'>50K':0})# creating binary for income
d_train = df.copy()

d_test = dft.copy()

df1 = df
X = d_train.drop('Income', axis=1)

y = d_train['Income']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
t = t.fit(X_train,y_train)
t.score(X_test,y_test)
print(len(df.columns))

print(len(dft.columns))
sol=t.predict(dft)

print(sol)
with open('income-predicted.csv','w') as fw:

    fw.write('ID,Income\n')

    ct=24001

    for i in sol:

        s=""

        if i==1:

            s="<=50K"

        else:

            s=">50K"

        fw.write(str(ct)+','+str(s)+'\n')

        ct+=1
sdf = pd.read_csv('income-predicted.csv', sep=',')

sdf.head()
sdf.Income.value_counts()