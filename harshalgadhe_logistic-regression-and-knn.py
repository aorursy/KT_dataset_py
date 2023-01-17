# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/logistic-regression/Social_Network_Ads.csv')

df.head()
df.info()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Gender']=le.fit(df['Gender']).transform(df['Gender'])
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='plasma')
df[['Purchased']].value_counts()
sns.pairplot(df)
from sklearn.model_selection import train_test_split

X=df.drop(['Purchased','User ID'],axis=1)

Y=df['Purchased']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=4)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)
yhat=lr.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

ax=confusion_matrix(yhat,y_test)

sns.heatmap(ax,annot=True,cmap='plasma')

plt.xlabel('Predict')

plt.ylabel('Actual')
print("Model Score : ",accuracy_score(yhat,y_test))
from sklearn.neighbors import KNeighborsClassifier
def Kneigh(X_train,X_test,y_train,y_test):

    

    score=[]

    

    for i in range(1,10):

        KN=KNeighborsClassifier(n_neighbors=i)

        KN.fit(X_train,y_train)

        KN_pred=KN.predict(X_test)

        score.append(accuracy_score(KN_pred,y_test))

    

    max_score=max(score)

    max_score_index=score.index(max_score)+1

    print(f"maximum score is {max_score} for neighbors ={max_score_index}")
Kneigh(X_train,X_test,y_train,y_test)