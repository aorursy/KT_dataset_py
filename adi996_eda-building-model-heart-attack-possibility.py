# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv")
df.head()
df.isnull().sum()
df.describe()
def categorize_age(age):

    if (age>60):

        return 'S'

    elif (age>45 and age<=60):

        return 'MA'

    elif (age>1 and age<=12):

        return 'C'

    elif (age>12 and age<=19):

        return 'T'

    elif (age>20 and age<=30):

        return 'YA'

    elif (age>=30 and age<=45):

        return 'A'
def categorize_sex(sex):

    if (sex==0):

        return "F"

    else:

        return "M"
df['age_category'] = df['age'].apply(lambda x: categorize_age(x))
df['gender'] = df['sex'].apply(lambda x:categorize_sex(x))
df.head()
df['gender'].value_counts().plot(kind='bar').set_title("Total records of Male & Female")
df['gender'].loc[df['target']==1].value_counts().plot(kind='bar').set_title("Gender has risk of heart disease")
df['gender'].loc[df['target']==0].value_counts().plot(kind='bar').set_title("Genders having no risk of heart disease")
df.head()
df['age_category'].loc[(df['target']==1) & (df['gender']=='M')].value_counts().plot(

    kind='bar').set_title("Age categories having high risk - For Male")
df['age_category'].loc[(df['target']==1) & (df['gender']=='F')].value_counts().plot(

    kind='bar').set_title("Age categories having high risk - For Female")
sns.scatterplot(x='thalach' , y='chol' , data=df , hue='target').set_title("Relationship between the cholestrol level and the highest heart rate recorded")
sns.violinplot(x='cp',y='chol',data=df.loc[df['target']==1] , 

               hue='age_category').set_title("Plot for different kind of chest pain's with respect to the cholestrol level & age")
sns.violinplot(x='gender',y='thalach',data=df.loc[df['target']==1] , hue='age_category').set_title("Highest heart rate with respect to the Age Categories")
sns.distplot(df['thalach'].loc[(df['target']==1) & (df['gender']=='M')],

             color='r').set_title("Distribution for the highest heart rate recorded for Males")
sns.distplot(df['thalach'].loc[(df['target']==1) & (df['gender']=='F')] 

             , color='y').set_title("Distribution for the highest heart rate recorded for Females")
sns.countplot(x="age_category" , data=df.loc[(df["gender"]=="M") & 

                                             (df["exang"]==1) &

                                             (df["target"]==1)]).set_title("Male category who are in high risk with Exercised induced Angina")
sns.countplot(x="age_category" , data=df.loc[(df["gender"]=="F") & 

                                             (df["exang"]==1) &

                                             (df["target"]==1)]).set_title("Female category who are in high risk with Exercise induced Angina")
sns.distplot(df["thalach"].loc[(df["target"]==1)& (df["exang"]==0) & (df["gender"]=="M")]

            ,color='r').set_title("Highest heart rate recorded for Male(Red) & Female(Yellow) (No Risk)")

sns.distplot(df["thalach"].loc[(df["target"]==1)& (df["exang"]==0) & (df["gender"]=="F")]

            ,color='y')
sns.distplot(df["thalach"].loc[(df["target"]==1)& (df["exang"]==1) & (df["gender"]=="M")]

            ,color='r').set_title("Highest heart rate recorded for Male(red) & Female(yellow) (High Risk)")

sns.distplot(df["thalach"].loc[(df["target"]==1)& (df["exang"]==1) & (df["gender"]=="F")]

            ,color='y')
sns.distplot(df["thalach"].loc[(df["target"]==0)& (df["exang"]==1) & (df["gender"]=="M")]

            ,color="r").set_title("Highest heart rate recorded for Male(Red) & Female(Yellow) who is Low risk but has Angina")

sns.distplot(df["thalach"].loc[(df["target"]==0)& (df["exang"]==1) & (df["gender"]=="F")]

            ,color="y")
train = df.drop(['target','gender','age_category'],axis=1)
target=df['target'].values
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

x_train , x_test , y_train, y_test = train_test_split(train,target,test_size=0.1)
print (f"X Train : {x_train.shape} \nY Train : {y_train.shape} \nX Test : {x_test.shape} \nY Test : {y_test.shape}")
classifiers = {"randomforest":RandomForestClassifier(),

              "xgboost":XGBClassifier(),

              "catboost":CatBoostClassifier(),

              "decisiontree": DecisionTreeClassifier()

              }
for model_name , model in classifiers.items():

    print (f"For : {model_name}")

    model.fit(x_train,y_train)

    prediction = model.predict(x_test)

    print (f"Classification Report for : {model_name}")

    print (classification_report(y_test,prediction))