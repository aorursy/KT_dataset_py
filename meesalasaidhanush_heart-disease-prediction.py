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
df=pd.read_csv(r'/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isnull().any()
import seaborn as sns
df.describe()
sns.boxplot(df['creatinine_phosphokinase'])
df['creatinine_phosphokinase']=np.where(df['creatinine_phosphokinase']>=582,582,df['creatinine_phosphokinase'])
sns.boxplot(df['creatinine_phosphokinase'])
sns.boxplot(df['platelets'])
df['platelets']=np.where(df['platelets']>=303500,303500,df['platelets'])
sns.boxplot(df['platelets'])
df['platelets']=np.where(df['platelets']<=212500,212500,df['platelets'])
sns.boxplot(df['platelets'])
cor=df.corr()
cor
sns.heatmap(cor)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
y=df.iloc[:,[12]]
x
y
x=sc.fit_transform(x)
# since the data is small iam not spliting into train and test
from sklearn import svm
s=svm.SVC()
s.fit(x,y)
pred=s.predict(x)
pred
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(pred,y))
print(classification_report(pred,y))
# lets try it with different algorithms
import xgboost
xgb=xgboost.XGBClassifier()
xgb.fit(x,y)
pre=xgb.predict(x)
print(confusion_matrix(pre,y))
print(classification_report(pre,y))
from sklearn.ensemble import RandomForestClassifier
forest= RandomForestClassifier()
forest.fit(x,y)
pr=forest.predict(x)
print(confusion_matrix(pr,y))
print(classification_report(pr,y))
