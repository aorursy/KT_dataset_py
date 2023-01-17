# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')



df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df.head()
df = df.drop(['sl_no'], axis = 1)

df = pd.DataFrame(df)

sns.countplot(x="status",data=df)
df["salary"].plot.hist(bins=200,figsize=(20,10))
sns.countplot(x="status",hue="hsc_s",data=df)
hsc_s=pd.get_dummies(df["hsc_s"])

degree_t=pd.get_dummies(df["degree_t"])

gender=pd.get_dummies(df["gender"])

workex=pd.get_dummies(df["workex"])

specialisation=pd.get_dummies(df["specialisation"])



df = df.drop(["gender","hsc_s","degree_t","workex","specialisation","ssc_b","hsc_b","salary"], axis = 1)

df

df=pd.concat([df,gender	,hsc_s,degree_t,workex,specialisation],axis=1)

df
X = df.drop(['status',], axis = 1)

y = df['status']







from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

s=0

a=[]

for i in range(2,9):

    classifier = KNeighborsClassifier(n_neighbors = i)

    classifier.fit(X_train, y_train)





    y_pred = classifier.predict(X_test)









    result2 = accuracy_score(y_test,y_pred)

    if result2>s:

        s=result2

        a=[i,result2]

        

print("Highest Accuracy is for n=",a[0],"and accuracy is ==",a[1])

X = df.drop(['status',], axis = 1)



y=pd.get_dummies(df["status"],drop_first=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.linear_model import *

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC # "Support Vector Classifier" 





classifier = SVC(kernel='linear')

classifier.fit(X_train, y_train)





y_pred = classifier.predict(X_test)









result2 = accuracy_score(y_test,y_pred)



print("Accuracy==",result2)

X = df.drop(['status',], axis = 1)



y=pd.get_dummies(df["status"],drop_first=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.linear_model import *

from sklearn.metrics import accuracy_score



classifier = LogisticRegression()

classifier.fit(X_train, y_train)





y_pred = classifier.predict(X_test)









result2 = accuracy_score(y_test,y_pred)



print("Accuracy==",result2)

X = df.drop(['status',], axis = 1)



y=pd.get_dummies(df["status"],drop_first=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)





y_pred = classifier.predict(X_test)









result2 = accuracy_score(y_test,y_pred)



print("Accuracy==",result2)







X = df.drop(['status',], axis = 1)



y=pd.get_dummies(df["status"],drop_first=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)





y_pred = classifier.predict(X_test)









result2 = accuracy_score(y_test,y_pred)



print("Accuracy==",result2)







X = df.drop(['status',], axis = 1)



y=pd.get_dummies(df["status"],drop_first=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



classifier = GaussianNB()

classifier.fit(X_train, y_train)





y_pred = classifier.predict(X_test)









result2 = accuracy_score(y_test,y_pred)



print("Accuracy==",result2)






