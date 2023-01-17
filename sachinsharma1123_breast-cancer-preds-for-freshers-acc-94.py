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
df=pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(df['diagnosis'])
#here we can observe that people diagnosed with breast cancer are more in numbers
sns.barplot(x=df['diagnosis'],y=df['mean_radius'],data=df)
#people diagnosed with disease have mean radius less than 12.5
sns.barplot(x=df['diagnosis'],y=df['mean_texture'],data=df)
#here we can observe that people diagnosed with disease have mean_texture less than 17
sns.barplot(x=df['diagnosis'],y=df['mean_perimeter'],data=df)
# from  here it is clear that diagnosed people have mean_perimeter og about 70
sns.barplot(x=df['diagnosis'],y=df['mean_area'],data=df)
#diagnosed peoples have mean area of about 400
sns.barplot(x=df['diagnosis'],y=df['mean_smoothness'],data=df)
#here in both classes mean_smoothness seems to be almost equal
sns.lineplot(x=df['mean_radius'],y=df['mean_perimeter'],data=df)
#here we can clearly say that mean_perimeter increases with mean_radius
sns.lineplot(x=df['mean_perimeter'],y=df['mean_area'],data=df)
#there is also a proportional relation bw them
y=df['diagnosis']

x=df.drop(['diagnosis'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

list_models=[]

list_scores=[]

x_train=sc.fit_transform(x_train)

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(sc.transform(x_test))

score_1=accuracy_score(y_test,pred_1)

list_scores.append(score_1)

list_models.append('LogisticRegression')
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(sc.transform(x_test))

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)

    
sns.lineplot(x=list(range(1,21)),y=list_1)
list_scores.append(max(list_1))

list_models.append('KNeighbors Classifier')
print(max(list_1))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(sc.transform(x_test))

score_2=accuracy_score(y_test,pred_2)

list_models.append('Randomforest Classifier')

list_scores.append(score_2)
score_2
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(sc.transform(x_test))

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('Support vector machines')
score_3
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_4=xgb.predict(sc.transform(x_test))

score_4=accuracy_score(y_test,pred_4)

list_models.append('XGboost')

list_scores.append(score_4)
score_4
plt.figure(figsize=(12,5))

plt.bar(list_models,list_scores)

plt.xlabel('classifiers')

plt.ylabel('accuracy scores')

plt.show()