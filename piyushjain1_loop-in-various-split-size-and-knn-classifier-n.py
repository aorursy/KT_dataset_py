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

from sklearn.datasets import load_iris

irdata=load_iris()

type(irdata)








irdata.target_names

irdata.data

irdf=pd.DataFrame(irdata.data,columns=irdata.feature_names)

irdict=irdf.to_dict()

irdf=pd.DataFrame(irdf)

irdata.keys()

irdata['target']

pd.DataFrame(irdata['target']).head(2)

irtargetdf=pd.DataFrame(irdata['target'])

type(irtargetdf)

irtargetdf.rename(columns={0:'target'},inplace=True)

y=irtargetdf

X=irdf

X.shape

y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.40,random_state=42)

print("X y train is X_train: {0}   \n  y_train :{1} \n  X_test: {2}  \n  y_test: {3}".format(X_train.head(2),y_train.head(2),X_test.head(2),y_test.head(2)))
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics 

listscores=[]

listvalue=[]



del j

j=.1

while j<1.0 :

    #    print('type of y is{}',type(y))

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=j,random_state=42)

  del i 

  i=1

# for i in range(1,10,1):

  while i<10:

                    knn=KNeighborsClassifier(n_neighbors=i)

                    knn.fit(X_train,y_train)

               # y_train=y_train.values.ravel()

          #      print('type of y is{}',type(y))

               # if knn.score(X_test,y_test)<.98 and knn.score(X_test,y_test)>.94:

                    listscores.append(round(knn.score(X_test,y_test),2))

                    

      #              print('accurancy score is {3} when  test size is {0} and n_neighbors is {1}  \n'.format(round(j,2),i, round(knn.score(X_test,y_test),2),round(metrics.accuracy_score(y_test,knn.predict(X_test)),2)))

                    listvalue.append(i)

            

                    y_pred=knn.predict(X_test)

                    y_pred

                    i=i+1

                    del knn

  j=j+.1

#     print('j value is {}',j)

  j=round(j,1)

    

    

#pd.DataFrame({'k_value'})

    

listscores

listvalue

y_pred

len(listscores)

i=0

while i <= len(listscores)-1:

#    listscores[ I = lambda x: i if listscores[i]>.96  ]

     f = lambda x: x if listscores[x]>.96 and listscores[x]<.98 else 0 

     print(listscores[f(i)])

     i=i+1

    


for item in listscores:

      if item<.98 and item>.96 : 

            print(item)
print(len(listscores),len(listvalue))

dfplot=pd.DataFrame({'k_value':listvalue,'scores':listscores })

dfplot.plot.line()

line=pd.DataFrame({'pred':y_pred,'y_test':y_test}).plot.line()