import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/suv-data/suv_data.csv')
df.head()
df.info()
df.isnull().sum()
sns.countplot(df['Gender'])
plt.show()
sns.distplot(df['Age'],color='darkred',kde=True)
plt.show()
plt.hist(df['EstimatedSalary'])
plt.show()
sns.boxplot(x='Gender',y='Age',data=df)
sns.countplot(x='Purchased',data=df)
sns.boxplot(x='Purchased',y='EstimatedSalary',data=df,hue='Gender')
X=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
import seaborn as sns
sns.heatmap(cm, annot=True)