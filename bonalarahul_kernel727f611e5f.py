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
import numpy as np

import pandas as pd
dataset=pd.read_csv('/kaggle/input/stack-overflow-2018-developer-survey/survey_results_public.csv')

dataset2=pd.read_csv('/kaggle/input/stack-overflow-2018-developer-survey/survey_results_schema.csv')
mydata=dataset.iloc[:,:17]
mydata.info()
mydata = mydata.dropna(axis=0, subset=['JobSatisfaction'])
mydata = mydata.dropna(axis=0, subset=['Country'])
mydata = mydata.dropna(axis=0, subset=['Student'])
mydata = mydata.dropna(axis=0, subset=['Employment'])
mydata = mydata.dropna(axis=0, subset=['FormalEducation'])
mydata = mydata.dropna(axis=0, subset=['UndergradMajor'])
mydata = mydata.dropna(axis=0, subset=['CompanySize'])
mydata = mydata.dropna(axis=0, subset=['DevType'])
mydata = mydata.dropna(axis=0, subset=['YearsCoding'])
mydata = mydata.dropna(axis=0, subset=['YearsCodingProf'])
mydata = mydata.dropna(axis=0, subset=['CareerSatisfaction'])
mydata = mydata.dropna(axis=0, subset=['HopeFiveYears'])
mydata = mydata.dropna(axis=0, subset=['JobSearchStatus'])
mydata = mydata.dropna(axis=0, subset=['LastNewJob'])
mydata = mydata.dropna(axis=0, subset=['OpenSource'])
mydata.info()
mydata.isnull().sum()
y=pd.DataFrame(mydata.iloc[:,2])



df=pd.DataFrame(mydata)

X=df.drop(['OpenSource'],axis=1)

a=pd.get_dummies(X,drop_first=True)
b=pd.get_dummies(y, drop_first=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(a,b,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier

regression=DecisionTreeClassifier()

regression.fit(X_train,y_train)

y_pred=regression.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
total=sum(sum(cm))
accuracy=(cm[0,0]+cm[1,1])/total
accuracy
from sklearn.ensemble import RandomForestClassifier

randomclassifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

randomclassifier.fit(X_train,y_train)
y_pred2=randomclassifier.predict(X_test)
cm2=confusion_matrix(y_test,y_pred2)
cm2
total2=sum(sum(cm2))
accuracy2=(cm2[0,0]+cm2[1,1])/total2
accuracy2
m=[]

z=0

for i in range(187,5977):

    k=X_train.columns[i]

    for j in range(0,42440):

     if(X_train[k].values[j]==1):

        

       if(y_train.iloc[j][0]==1):

            m.append(True)

       else:

            m.append(False)

            

            

    if(all(m)==any(m)):

        print(i)

        z=z+1

            

    

        

    

          



m=[]

z=0

for i in range(2,161):

    k=X_train.columns[i]

    for j in range(0,42440):

     if(X_train[k].values[j]==1):

        

       if(y_train.iloc[j][0]==1):

            m.append(True)

       else:

            m.append(False)

            

            

    if(all(m)==any(m)):

        print(i)

        z=z+1
X_train.drop(X_train.iloc[:,2:160],inplace=True,axis=1)
X_train.drop(X_train.iloc[:,187:5976],inplace=True,axis=1)
X_test.drop(X_test.iloc[:,2:160],inplace=True,axis=1)
X_test.drop(X_test.iloc[:,187:5976],inplace=True,axis=1)
from sklearn.tree import DecisionTreeClassifier

classification=DecisionTreeClassifier()

classification.fit(X_train,y_train)
y3_pred=classification.predict(X_test)
from sklearn.metrics import confusion_matrix
cm3=confusion_matrix(y_test,y3_pred)
accuracy3=(cm3[0,0]+cm3[1,1])/sum(sum(cm3))
accuracy3
from sklearn.ensemble import RandomForestClassifier

randomclassifier2=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

randomclassifier2.fit(X_train,y_train)
y_pred4=randomclassifier2.predict(X_test)
cm4=confusion_matrix(y_test,y_pred4)
accuracy4=(cm4[0,0]+cm4[1,1])/sum(sum(cm4))
accuracy4