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

import pandas as pd 



df=pd.read_csv('..//input//covid19-patient-precondition-dataset//covid.csv')
df=df.drop(columns=['id','entry_date','date_symptoms','date_died'],axis=1)

df
df=df.replace({97:np.nan,98:np.nan,99:np.nan})

df=df.dropna()
df['icu'].value_counts()
df1=df[df['icu']==2][1:3000]

df2=df[df['icu']==1][1:2695]

frames=[df1,df2]

df_new=pd.concat(frames)

df_new_s=df_new.sample(frac=1)

df_new_s
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



def Knc(X_train,y_train,X_test,y_test):

    neigh = KNeighborsClassifier(n_neighbors=3)

    neighb=neigh.fit(X_train,y_train)

    y_pred=neighb.predict(X_test)

    print("Accuracy is:",accuracy_score(y_test,y_pred))
X=df_new_s.iloc[:,:-1]

y=df_new_s.iloc[:,-1]

X.shape,y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
Knc(X_train,y_train,X_test,y_test)
from sklearn.linear_model import LogisticRegression

def LR(X_train,y_train,X_test,y_test):

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy is:",accuracy_score(y_test,y_pred))
y=df_new_s.icu

data=['sex', 'patient_type', 'intubed', 'pneumonia', 'pregnancy',

       'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension',

       'other_disease', 'cardiovascular', 'obesity', 'renal_chronic',

       'tobacco', 'contact_other_covid', 'covid_res']

df_d=pd.get_dummies(df_new_s, columns = data)

df_d=df_d.drop(columns=['icu'],axis=1)

df_d.shape,y.shape

df_d

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df_d,y,test_size=0.2,random_state=0,stratify=y)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
LR(X_train,y_train,X_test,y_test)
Knc(X_train,y_train,X_test,y_test)