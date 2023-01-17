# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.info()
bins_ = (2,6.5,8)

labels_ = ['bad','good']

df['quality']=pd.cut(df['quality'],bins=bins_,labels=labels_)

print(df['quality'])
sns.countplot(df['quality'],data=df)
x1=df[df['quality']=='bad']['quality'].count()

x2=df[df['quality']=='good']['quality'].count()

ratio =x1/(x1+x2)

print("Percentage of Bad Quality data in the dataset:",ratio*100)
plt.figure(figsize=(10,5))

sns.heatmap(df.corr(),annot=True)
from sklearn.preprocessing import StandardScaler,LabelEncoder



label_enc = LabelEncoder()

df['quality'] = label_enc.fit_transform(df['quality'])

df.head(10)
y = df['quality']

X = df.drop('quality',axis=1)
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(random_state=0,replacement = True)

rus.fit(X,y)

X_resampled, y_resampled = rus.fit_resample(X,y)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,random_state=0,train_size=0.85) 
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report,mean_absolute_error,accuracy_score



model1 = RandomForestClassifier(n_estimators=500)

model1.fit(X_train,y_train)

pred1 = model1.predict(X_test)

print(mean_absolute_error(y_test,pred1))

print(accuracy_score(y_test,pred1))
print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
model2 = LogisticRegressionCV(cv=5,random_state=0)

model2.fit(X_train,y_train)

pred2 = model2.predict(X_test)

print(mean_absolute_error(y_test,pred2))

print(accuracy_score(y_test,pred2))
print(classification_report(y_test,pred2))
print(confusion_matrix(y_test,pred2))