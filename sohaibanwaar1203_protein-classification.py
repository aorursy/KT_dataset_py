# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/table-of-instability"))

# Any results you write to the current directory are saved as output.


import pandas as pd



def sumDIWV(s):
    s=0
    for i in range(len(str(abs(s)))-1):
        s+=instability_index_table[s[i+1]][s[i]]
    return s 




soluble_set= pd.read_csv('../input/proteins-sequence/Soluble_Set_2.csv', names=['id','name','subunit','residue'],header=None)
insoluble_set=pd.read_excel('../input/proteins-sequence/Insoluble_Set_2.xlsx', names=['id','name','subunit','residue'],header=None)
test_set=pd.read_excel("../input/testset/Test_Set.xlsx",names=['label','id','name','subunit','residue'],header=None)
instability_index=pd.read_excel("../input/table-of-instability/Table for Instability Index.xlsx",header=None)
#instability_index=instability_index_table[1:]
cols=instability_index[:1].values
cols=cols[0][1:]
instability_index=instability_index[1:]
instability_index.set_index(0,inplace=True)
instability_index.columns=cols







asp_soluble= soluble_set.residue.apply((lambda str: str.count('N')))
gly_soluble= soluble_set.residue.apply((lambda str: str.count('G')))
pro_soluble= soluble_set.residue.apply((lambda str: str.count('P')))
ser_soluble= soluble_set.residue.apply((lambda str: str.count('S')))
arg_soluble= soluble_set.residue.apply((lambda str: str.count('R')))
lys_soluble= soluble_set.residue.apply((lambda str: str.count('K')))
aspAcd_soluble= soluble_set.residue.apply((lambda str: str.count('D')))
gluAcd_soluble= soluble_set.residue.apply((lambda str: str.count('E')))
alan_soluble=soluble_set.residue.apply((lambda str: str.count('A')))
val_soluble= soluble_set.residue.apply((lambda str: str.count('V')))
iso_soluble=soluble_set.residue.apply((lambda str: str.count('I')))
leu_soluble= soluble_set.residue.apply((lambda str: str.count('L')))

asp_insoluble= insoluble_set.residue.apply((lambda str: str.count('N')))
gly_insoluble= insoluble_set.residue.apply((lambda str: str.count('G')))
pro_insoluble= insoluble_set.residue.apply((lambda str: str.count('P')))
ser_insoluble= insoluble_set.residue.apply((lambda str: str.count('S')))
arg_insoluble= insoluble_set.residue.apply((lambda str: str.count('R')))
lys_insoluble= insoluble_set.residue.apply((lambda str: str.count('K')))
aspAcd_insoluble= insoluble_set.residue.apply((lambda str: str.count('D')))
gluAcd_insoluble= insoluble_set.residue.apply((lambda str: str.count('E')))
alan_insoluble=insoluble_set.residue.apply((lambda str: str.count('A')))
val_insoluble= insoluble_set.residue.apply((lambda str: str.count('V')))
iso_insoluble=insoluble_set.residue.apply((lambda str: str.count('I')))
leu_insoluble= insoluble_set.residue.apply((lambda str: str.count('L')))




AI=  alan_soluble + 2.9*val_soluble + 3.9*(iso_soluble+leu_soluble)
N= soluble_set.residue.apply(lambda str: len(str))
CV=  (15.43 * ((asp_soluble+gly_soluble+pro_soluble+ser_soluble)/N)) - (29.56* abs( ( (arg_soluble+lys_soluble)-(aspAcd_soluble+gluAcd_soluble))/N - 0.03 ) )
Stp= 1/(N-1) * (N-1) * 0.2
sum= soluble_set.residue.apply(sumDIWV)
II=  10/N * sum
Fn=asp_soluble
Ft= soluble_set.residue.apply((lambda str: str.count('T')))
Fy= soluble_set.residue.apply((lambda str: str.count('Y')))
SI= ( 0.648*AI + 0.274*II - 0.539*Fn - 0.508*Ft - 0.604*Fy - Stp* 10000 )/100
soluble=pd.concat([AI,CV,Stp,II,Fn,Ft,Fy,SI],axis=1)
soluble.columns=(['AI','CV','Stp','II','Fn','Ft','Fy','SI'])
soluble['label']='SOLUBLE'



AI=  alan_insoluble + 2.9*val_insoluble + 3.9*(iso_insoluble+leu_insoluble)
N= insoluble_set.residue.apply(lambda str: len(str))
CV=  (15.43 * ((asp_insoluble+gly_insoluble+pro_insoluble+ser_insoluble)/N)) - (29.56* abs( ( (arg_insoluble+lys_insoluble)-(aspAcd_insoluble+gluAcd_insoluble))/N - 0.03 ) )
Stp= 1/(N-1) * (N-1) * 0.2
sum= insoluble_set.residue.apply(sumDIWV)
II=  10/N * sum
Fn=asp_insoluble
Ft= insoluble_set.residue.apply((lambda str: str.count('T')))
Fy= insoluble_set.residue.apply((lambda str: str.count('Y')))
SI= ( 0.648*AI + 0.274*II - 0.539*Fn - 0.508*Ft - 0.604*Fy - Stp* 10000 )/100
insoluble=pd.concat([AI,CV,Stp,II,Fn,Ft,Fy,SI],axis=1)
insoluble.columns=(['AI','CV','Stp','II','Fn','Ft','Fy','SI'])
insoluble['label']="INSOLUBLE"

frames = [insoluble, soluble]

df = pd.concat(frames)
df.head()
df_x=df.drop('label',axis=1)
df_y=df['label']
df_y.head()

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier()
i=0

while(i<10):
    print('Iteration: ',i)
    clf = clf.fit(X_train, y_train)
    print('Accuracy of Decession Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    i=i+1





from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X_train, y_train) 
print('Accuracy of SGDC classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
predicted = clf.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print('Accuracy of KNN classifier with 3 neighbour on test set : {:.2f}'.format(clf.score(X_test, y_test)))



clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
print('Accuracy of KNN classifier with 5 neighbour on test set : {:.2f}'.format(clf.score(X_test, y_test)))
 
