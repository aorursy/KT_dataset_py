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
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

df.head()
import matplotlib.pyplot as plt

import seaborn as sns
df1 = df.copy()

df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI',]] =  df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI',]].replace(0,np.NaN)

df1.info()
df1['Glucose'].fillna(df1['Glucose'].mean(),inplace = True)

df1['BloodPressure'].fillna(df1['BloodPressure'].mean(),inplace = True)

df1['SkinThickness'].fillna(df1['SkinThickness'].mean(),inplace = True)

df1['Insulin'].fillna(df1['Insulin'].mean(),inplace = True)

df1['BMI'].fillna(df1['BMI'].mean(),inplace = True)
c = df1.hist(figsize = (20,20))
plt.figure(figsize = (12,10))

sns.heatmap(df1.corr(),annot = True,cmap = 'RdYlGn')
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()

X =  pd.DataFrame(Scaler.fit_transform(df1.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()
from sklearn.model_selection import train_test_split

y = df1['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y  )
from sklearn.linear_model import LogisticRegression

my_model = LogisticRegression()

my_model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

pred = my_model.predict(X_test)

print(accuracy_score(pred,y_test))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred)

pd.crosstab(y_test, pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_p = knn.predict(X_test)

print(accuracy_score(y_p,y_test))
from sklearn.metrics import classification_report

print(classification_report(y_test,y_p))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred)

pd.crosstab(y_test, y_p, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_p)