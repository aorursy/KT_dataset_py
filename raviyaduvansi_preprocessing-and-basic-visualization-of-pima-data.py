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
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
pima=pd.read_csv('/kaggle/input/pimaindiansdiabetescsv/pima-indians-diabetes.csv')
pima.head()
pima.rename(columns = {'6': 'Pregnancies','148':'Glucose', '72':'BloodPressure','35':'SkinThickness', '0':'Insulin','33.6':'BMI'

                       ,'0.627':'DiabetesPedigreeFunction','50': 'Age','1':'Outcome'}, inplace = True) 
pima.columns
pima.info()
pima.describe()
pima.hist(figsize=(12,12),bins=50)

plt.tight_layout()
import seaborn as sns
sns.countplot(x='Outcome', data=pima)
sns.countplot(x="Pregnancies", data=pima)
pimacor=pima.corr()
pimacor
sns.heatmap(pimacor,linewidths=0.5,annot=True)
plt.figure(figsize=(10,7))

sns.boxplot(x="Outcome",y="Age",data=pima)
sns.boxplot(x="Outcome",y="BMI",data=pima)
X=pima.drop("Outcome",axis=1)
y=pima["Outcome"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print("\n")

print(classification_report(y_test,predictions))
from sklearn import tree
Dtree=dtree.fit(X_train,y_train)
plt.figure(figsize=(200,150))

tree.plot_tree(Dtree,filled=True,fontsize=70)
dtree=DecisionTreeClassifier(criterion="entropy",max_depth=10)
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print("\n")

print(classification_report(y_test,predictions))