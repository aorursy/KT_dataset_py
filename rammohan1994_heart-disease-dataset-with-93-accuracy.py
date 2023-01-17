# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in Attribute Information:



#Only 14 attributes used:



#age : Patient age in years

#sex : sex (1 = male; 0 = female)

#cp : cp: chest pain type

    #     -- Value 1: typical angina 

   #      -- Value 2: atypical angina 

  #       -- Value 3: non-anginal pain 

 #        -- Value 4: asymptomatic

#trestbps : resting blood pressure (in mm Hg on admission to the hospital)

#chol : serum cholestoral in mg/dl

#fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

#restecg : resting electrocardiographic results

   #      -- Value 0: normal 

  #       -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 

 #        -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 

#thalach : maximum heart rate achieved

#exang : exercise induced angina (1 = yes; 0 = no)

#oldpeak : ST depression induced by exercise relative to rest

#slope : the slope of the peak exercise ST segment

  #     -- Value 1: upsloping 

 #      -- Value 2: flat 

#       -- Value 3: downsloping 

#ca : number of major vessels (0-3) colored by flourosopy

#thal : 3 = normal; 6 = fixed defect; 7 = reversable defect

#target  (the predicted attribute) : diagnosis of heart disease (angiographic disease status)

 #  -- Value 0: < 50% diameter narrowing  No Disease

#   -- Value 1: > 50% diameter narrowing  With Disease



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")

df.head()
df.columns
df.describe()
df.info()
## EDA
sns.set_style('whitegrid')

%matplotlib inline

plt.figure(figsize = (20,10))

sns.heatmap(df.corr(),annot= True, linewidth = 3,)
sns.countplot(x ='target', data= df, hue = 'sex')

plt.legend(['Female ','Male'])

plt.xlabel ( 'No Heart disease                 Heart Disease')
df.nunique()
sns.pairplot(df, hue = 'target')

import warnings

warnings.filterwarnings("ignore")
df['age'].describe()
df['Age_Category']= pd.cut(df['age'],bins=list(np.arange(25, 85, 5)))
df.head()
plt.figure(figsize=(20,5))



plt.subplot(121)

df[df['target']==1].groupby('Age_Category')['age'].count().plot(kind='bar')

plt.title('Age Distribution of Patients with +ve Heart Diagonsis')



plt.subplot(122)

df[df['target']==0].groupby('Age_Category')['age'].count().plot(kind='bar')

plt.title('Age Distribution of Patients with -ve Heart Diagonsis')
sns.countplot(x = 'cp' ,data = df, hue = 'target' )

plt.xlabel('typical angina     atypical angina     non-anginal pain     asymptomatic')

plt.ylabel(' Chest pain Type')

plt.legend(['No disease' ,         ' disease'])

sns.countplot(x = 'fbs' ,data = df, hue = 'target' )

plt.xlabel('< 120mm/Hg                   >120 mm/Hg')

plt.ylabel('Fasting blood sugar')

plt.legend(['No disease' ,         ' disease'])
sns.countplot(x = 'exang' ,data = df, hue = 'target' )

plt.xlabel('No ex                                     Exercise')

plt.title(' Excercise effect on Heart diease')

plt.legend(['No disease' ,         ' disease'])
df = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal','Age_Category'])
df.head()
df.drop(['age'],axis=1,inplace=True)
df.head()
df.shape
y = df['target']

x = df.drop(['target'],  axis =1)
df.shape
x.shape
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 5)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

rad = RandomForestClassifier()

rad.fit(X_train,y_train)
y_pred = rad.predict(X_test)
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))