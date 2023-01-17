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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('/kaggle/input/parkinsons-data-set/parkinsons.data')
pd.set_option('display.max_columns', None)
df
df.shape
df.info()
df.describe()
df['status'].value_counts(normalize=True)
df.isnull().sum()
sns.pairplot(df,hue='status')
plt.show()
fig,axes=plt.subplots(5,5,figsize=(15,15))
axes=axes.flatten()

for i in range(1,len(df.columns)-1):
    sns.boxplot(x='status',y=df.iloc[:,i],data=df,orient='v',ax=axes[i])
plt.tight_layout()
plt.show()
plt.figure(figsize=(25,15))
sns.heatmap(df.corr(),annot=True,linewidths=0.8)
plt.show() 
X=df.drop(['status','name'],axis=1)
y=df['status']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
pred=dtc.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
print(confusion_matrix(y_test,pred_rf))
print('\n')
print(classification_report(y_test,pred_rf))
print('\n')
print("The Train score of Random Forest is :",rf.score(X_train,y_train))
print("The Test score of Random Forest is :",rf.score(X_test,y_test))
