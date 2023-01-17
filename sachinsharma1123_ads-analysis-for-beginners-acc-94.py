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
df=pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
df
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(df.corr())
#lets drop the id columns as it is unnecessary 
df=df.drop(['User ID'],axis=1)
df
sns.countplot(x=df['Purchased'],hue='Gender',data=df)
#females purchased more than males
sns.lineplot(x=df['Age'],y=df['EstimatedSalary'],data=df)
#from this figure we can conclude that salary slightly increases with increase in age
sns.distplot(df['EstimatedSalary'])
#now preprocess categorical features i
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Gender']=le.fit_transform(df['Gender'])
y=df['Purchased']

x=df.drop(['Purchased'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

list_scores=[]

list_models=[]

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
list_scores.append(score_1)

list_models.append('logistic regression')
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_1.append(scores)
sns.barplot(x=list(range(1,21)),y=list_1)
print(max(list_1))
list_scores.append(max(list_1))

list_models.append('kneighbors classifier')

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
list_scores.append(score_2)

list_models.append('random forest classifier')
from sklearn.ensemble import GradientBoostingClassifier

gbr=GradientBoostingClassifier()

gbr.fit(x_train,y_train)

pred_3=gbr.predict(x_test)

score_3=accuracy_score(y_test,pred_3)
score_3
list_scores.append(score_3)

list_models.append('gradient boosting classifier')
plt.figure(figsize=(10,5))

plt.bar(list_models,list_scores,width=0.4)
#from the above figure we can conclude that random forest gives the best accuracy score