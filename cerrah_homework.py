# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data.head()
X = data.drop(["gameId","blueWins"],axis=1)

y = data["blueWins"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)



from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

pred1=lr.predict(X_test)

print(classification_report(y_test,pred1))
from  sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)

pred2 = model.predict(X_test)

print(classification_report(y_test,pred2))
from sklearn.neighbors import KNeighborsClassifier

knnscore=[]

for i,k in enumerate(range(1,40)):

    knn = KNeighborsClassifier(n_neighbors=k)

    

    knn.fit(X_train,y_train)

    

    knnscore.append(knn.score(X_test,y_test))
knn = KNeighborsClassifier(1+knnscore.index(np.max(knnscore)))

knn.fit(X_train,y_train)

pred3 = knn.predict(X_test)

print(classification_report(y_test,pred3))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

pred4 = dt.predict(X_test)

print(classification_report(y_test,pred4))
from sklearn.ensemble import RandomForestClassifier

rfcscore = []

for i,k  in enumerate(range(100,300,20)):

    rfc = RandomForestClassifier(n_estimators=k)

    rfc.fit(X_train,y_train)

    rfcscore.append(rfc.score(X_test,y_test))
rfc = RandomForestClassifier(n_estimators=(1+rfcscore.index(np.max(rfcscore))))

rfc.fit(X_train,y_train)

pred5= rfc.predict(X_test)

print(classification_report(y_test,pred5))


