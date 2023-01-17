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
#loading dataset

import pandas as pd

import numpy as np

#visualisation

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#EDA

from collections import Counter

import pandas_profiling as pp

# data preprocessing

from sklearn.preprocessing import StandardScaler

# data splitting

from sklearn.model_selection import train_test_split

# data modeling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

#ensembling

from mlxtend.classifier import StackingCVClassifier
df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df.head()
df.info()
df.isnull().sum()
pp.ProfileReport(df)
df= df.drop('Serial No.', axis=1)

df
df['ChanceAdmit'] = df['Chance of Admit '].map(lambda x : 1 if x>0.80 else 0)
df.groupby(['ChanceAdmit'])['ChanceAdmit'].count().plot.pie(autopct='%.f%%' , shadow=True)

plt.title('% of Students admitted')

plt.legend(['Not Admitted','Admitted'])

plt.show()
fig,ax = plt.subplots(1,3,figsize=(20,10))

sns.scatterplot(df['GRE Score'] , df['Chance of Admit ']  , hue=df['ChanceAdmit'], ax=ax[0])

sns.scatterplot(df['TOEFL Score'] , df['Chance of Admit '] ,hue=df['ChanceAdmit'] , ax=ax[1])

sns.scatterplot(df['CGPA'] , df['Chance of Admit '] ,hue=df['ChanceAdmit'] , ax=ax[2])

ax[0].set_title('GRE Score vs Chance of Admit')

ax[1].set_title('TOEFL Score vs Chance of Admit')

ax[2].set_title('CGPA vs Chance of Admit')

plt.show()
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(df['Research'] , df['GRE Score'] ,ax=ax[0])

sns.barplot(df['Research'] , df['GRE Score'] , ax=ax[1])

ax[0].set_title('GRE Score vs Research experience')

ax[1].set_title('TOEFL Score vs Research experience')

plt.show()
fig,ax = plt.subplots(1,2,figsize=(10,5))

df.groupby(['Research'])['Research'].count().plot.pie(autopct='%.f%%' ,ax=ax[0] ,shadow=True )

ax[0].set_title('% of students having Research experience')

df.groupby(['University Rating'])['University Rating'].count().plot.pie(autopct='%.f%%' , ax=ax[1],shadow=True)

ax[1].set_title('University Rating of students')

plt.show()
X = df.drop(["Chance of Admit "], axis=1)

y = df["Chance of Admit "]
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
lr = LinearRegression()

model = lr.fit(X_train,y_train)



def get_cv_scores(model):

    scores = cross_val_score(model,

                             X_train,

                             y_train,

                             cv=5,

                             scoring='r2')

    

    print('CV Mean: ', np.mean(scores))

    print('STD: ', np.std(scores))

    print('\n')



# get cross val scores

get_cv_scores(model)
score1 = model.score(X_test, y_test)

print("Accuracy of Logistic Regression:",score1*100,'\n')