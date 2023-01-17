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
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
df=pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df.head()
#in order to check if its balanced data
sns.countplot(df['target'])
sns.pairplot(df)
sns.distplot(df['age'])
plt.figure(figsize=(12,8))
df['age'][df['target']==1].hist(label='Patients predicted to have attacks',alpha=.6,bins=20)
df['age'][df['target']==0].hist(label='Patients predicted not to have attacks',alpha=.6,bins=20)
plt.legend()
plt.figure(figsize=(12,8))
df['trestbps'][df['target']==1].hist(label='Patients predicted to have attacks',alpha=.6,bins=20)
df['trestbps'][df['target']==0].hist(label='Patients predicted not to have attacks',alpha=.6,bins=20)
plt.legend()
sns.heatmap(df.corr())
df.corr()
df.isnull().sum()
df.info()
#using knn for classification . First we need to scale all the features by isng standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('target',axis=1))
df_scaled=scaler.transform(df.drop('target',axis=1))
df.columns
final_df=pd.DataFrame(df_scaled,columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])
final_df.head()
df.target.head()
# now split the data 
from sklearn.model_selection import train_test_split
X=final_df
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# train the model 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
predictknn=knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictknn))
print(confusion_matrix(y_test,predictknn))
# to find the right n_neighbor
error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test))

plt.figure(figsize=(12,8))
plt.plot(range(1,40),error_rate,color='blue',marker='o')
knn1=KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train,y_train)
predictknn1=knn1.predict(X_test)
print(classification_report(y_test,predictknn1))
print(confusion_matrix(y_test,predictknn1))
