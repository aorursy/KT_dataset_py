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
df=pd.read_csv('../input/student-job/student_job.csv')
df.head()
df.info()
def change(x):

    if x=='Good' or x=='Yes':

        return 1

    elif x=='Bad' or x=='No':

        return 0

    
from sklearn.preprocessing import LabelEncoder
df=df.apply(LabelEncoder().fit_transform)

df.head()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
x_train,x_test,y_train,y_test=train_test_split(df.drop('class',axis=1),df['class'],test_size=0.3,random_state=101)
gauss=GaussianNB()
gauss.fit(x_train,y_train)
pred=gauss.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,pred,normalize=True))
y_test
pred
x.loc[4]=[1,1,1,0]
x
gauss.predict(x)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth=3,random_state=101)
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred,normalize=True))
from IPython.display import Image  

from sklearn import tree

from sklearn.externals.six import StringIO  

import pydot          



dot_data = StringIO() 

tree.export_graphviz(dtree, out_file=dot_data)  

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("Dtree.pdf")