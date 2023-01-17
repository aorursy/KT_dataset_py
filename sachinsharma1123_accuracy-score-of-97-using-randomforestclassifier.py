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
df=pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df
df.columns
df.isnull().sum()
del_cols=['job_id','location','department','salary_range','description','title','company_profile','benefits','requirements']
df=df.drop(del_cols,axis=1)
df
col_list=list(df.columns)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in col_list:

    if df[i].dtype=='object':

        df[i]=df[i].replace(np.nan,df[i].mode()[0],regex=True)

    
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
for i in col_list:

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
df
y=df['fraudulent']
x=df.drop(['fraudulent'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression()
lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)
score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_y=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_y)

    list_1.append(scores)
import matplotlib.pyplot as plt

plt.plot(range(1,11),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy scores')

plt.show()
#from the figure k=3 gives the best accuracy
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

pred_2=knn.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred_2))
from sklearn.svm import SVC

model=SVC()
model.fit(x_train,y_train)

pred_3=model.predict(x_test)

score_3=accuracy_score(y_test,pred_3)
score_3
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)
pred_4=rfc.predict(x_test)

score_4=accuracy_score(y_test,pred_4)
score_4
#so from all the classification models RandomForestClassifier gives the best accuracy score
new_df=pd.DataFrame({'actual':y_test,

                    'predicted':pred_4})
new_df