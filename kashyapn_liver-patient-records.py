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
df = pd.read_csv(r'/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
df['Albumin_and_Globulin_Ratio'].unique()
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(0.930)
df['Albumin_and_Globulin_Ratio'].unique()
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].astype(float)
df.info()
df['Dataset'].value_counts()
df1 = pd.get_dummies(df['Gender'])
df2 = pd.concat([df,df1],axis=1)
df2.head()
df2.drop('Gender',axis=1,inplace=True)
df2.head()
import seaborn as sns
import matplotlib.pyplot as plt
corr = df2.corr()

plt.figure(figsize=(20,20))
sns.heatmap(df2.corr(),annot=True)
y = df2['Dataset']
X = df2.drop('Dataset',axis=1)
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()
X.head()
X.drop(['Direct_Bilirubin','Alamine_Aminotransferase','Albumin'],axis=1,inplace=True)
corr = X.corr()

plt.figure(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)
X.head()
from sklearn.model_selection import train_test_split,GridSearchCV

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state= 42)
from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier(random_state=6)
rand_clf.fit(X_train,y_train)
rand_clf.score(X_test,y_test)
prediction=rand_clf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,prediction))
print(accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))
