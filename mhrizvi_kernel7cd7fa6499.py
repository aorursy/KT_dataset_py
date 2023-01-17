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
#Importing  the required libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns
#Loading  the train and test data

train_df1=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df1=pd.read_csv("/kaggle/input/titanic/test.csv")
#Getting data summary

train_df1.shape

test_df1.shape

train_df1.shape
#Merging two dataframes

titanic_df1=pd.concat([train_df1,test_df1])
titanic_df1.head()
titanic_df1.describe()
titanic_df1.shape
#Building  Hypothesis related to Cabin Sibsp and Parch features

#1.The persons living in higher class have more chances of survival

#2The person travelling in family or  with mates have large chances of survival.

#3 Younger people are having chances of survival.
#Exploratory Data Analysis

#Checking for Missing Values

print(titanic_df1.isnull().sum())

#Handling Age, Cabin and Embarked Missing Values

# Taking just first letter of Cabin like as numbers won't make much sense and assigning  the rankings  based on frequency.

titanic_df1['Cabin']=titanic_df1['Cabin'].astype(object)

titanic_df1['Cabin_new']=titanic_df1['Cabin'].str[0]

cabin_map=titanic_df1['Cabin_new'].value_counts()

titanic_df1['Cabin_new']=titanic_df1['Cabin_new'].map(cabin_map)

titanic_df1['Cabin_new'].value_counts()

titanic_df1['Cabin_new']=titanic_df1['Cabin_new'].fillna(0)

titanic_df1.drop('Cabin',axis=1,inplace=True)

titanic_df1.rename({'Cabin_new':'Cabin'},axis=1,inplace=True)

titanic_df1['Age_new']=titanic_df1['Age']

    

Age_new=titanic_df1['Age'].dropna().sample(titanic_df1['Age'].isnull().sum(),random_state=0)

                                         

Age_new.index=titanic_df1[titanic_df1['Age'].isnull()].index

titanic_df1.loc[titanic_df1['Age'].isnull(),'Age_new']=Age_new

#Dropping the Age column

titanic_df1.drop('Age',axis=1,inplace=True)

titanic_df1.rename({'Age_new':'Age'},axis=1,inplace=True)
x=titanic_df1['Fare'].mean()
#Imputing the Embarked feature with the mode value and Fare with mean value.

titanic_df1['Embarked']=titanic_df1['Embarked'].fillna(str(titanic_df1['Embarked'].value_counts().index[0]))

titanic_df1['Fare'].fillna(x,inplace=True)

titanic_df1.isnull().sum()

#Validating are Hypothesis

sns.barplot(x='Pclass',y='Survived',data=titanic_df1)

#Cearly passenger living in high class have have higher chances of survival
titanic_df1['isALone']=np.where((titanic_df1['SibSp']==0) | (titanic_df1['Parch']==0),1,0)

    



titanic_df1.head()
sns.set(style='whitegrid')

sns.barplot(x='isALone',y='Survived',data=titanic_df1)

#Person  not tavelling alone have higher chances of survival
titanic_df1['Age']=pd.to_numeric(titanic_df1['Age']).astype(int)

titanic_df1['isYoung']=np.where((titanic_df1['Age']>20) & (titanic_df1['Age']<35),1,0 )

sns.set(style='whitegrid')



sns.barplot(x=titanic_df1['isYoung'],y=titanic_df1['Survived'])

#We see that the younger people have nearly  same chances of survival as  old ones.So neglecting this hypothesis.
#Extracting the name length and using it as a new feature.

titanic_df1['Name']=titanic_df1['Name'].astype(object)
titanic_df1['NameLength']=titanic_df1['Name']

titanic_df1['NameLength']=titanic_df1['Name'].apply(len)

titanic_df1['NameLength'].hist(bins=20)

titanic_df1.drop('Name',axis=1,inplace=True)

titanic_df1.rename({'NameLength':'Name'},axis=1,inplace=True)
#Finally collaborating the hypothesis results

titanic_df1.head()

#Dropping Passenger Id ,Ticket isYoung features.

titanic_df1.drop(columns=['PassengerId','Ticket','isYoung'],axis=0,inplace=True)
titanic_df1.head()
#Outlier detection

#Checking whether the our data follows a Gaussian distribution

sns.distplot(titanic_df1['Fare'])

#The Fare feature is right skewed



sns.boxplot(titanic_df1['Fare'])

# there are some extreme outliers.

titanic_df1['Fare'].describe()
#Using Inter Quantile Range mehtod to remove the outliers

IQR=titanic_df1['Fare'].quantile(0.75)-titanic_df1['Fare'].quantile(0.25)

Lower_Bound=titanic_df1['Fare'].quantile(0.25)-(3*IQR)

Upper_Bound=titanic_df1['Fare'].quantile(0.75)+(3*IQR)

print("Lower_Bund,Upper_Bound,IQR",Lower_Bound,Upper_Bound,IQR)
titanic_df1.loc[titanic_df1['Fare']>101 ,'Fare']=101
# Age column outlier detection.

titanic_df1.boxplot(column='Age')
titanic_df1.Age.describe()
sns.distplot(titanic_df1['Age'],bins=50)

#The Age nearly follows a Normal Distribution so we just replace extreme outliers values by following the Central 

#Limit Theorem property.

uppper_fence=train_df1['Age'].mean() + 3* train_df1['Age'].std()

lower_fence=train_df1['Age'].mean() - 3* train_df1['Age'].std()

print(lower_fence), print(uppper_fence),print(train_df1['Age'].mean())
titanic_df1.loc[titanic_df1['Age']> 71 ,'Age']=71
#Encoding Categorical Variables Sex,Embarked  to using one hot enconding

titanic_df1=pd.get_dummies(data=titanic_df1,columns=['Sex','Embarked'],drop_first=True)
titanic_df1.head()
titanic_df1.isnull().sum()
#We know we have 891 samples intially in our training dataset.So spltting based on that our train and test data



X_train=titanic_df1.iloc[:891,1:]

Y_train=titanic_df1.iloc[:891,0]

X_test=titanic_df1.iloc[891:,1:]
#Feature Scaling 

from sklearn.preprocessing import StandardScaler

st=StandardScaler()

X_train=st.fit_transform(X_train)

X_test=st.transform(X_test)

#Model Creation.

from sklearn.linear_model import LogisticRegression

LR_model=LogisticRegression()

LR_model.fit(X_train,Y_train)
Y_pred=LR_model.predict(X_test)

print(LR_model.score(X_train,Y_train))
Y_pred.shape
#XGBosst Classifier

import xgboost as xgb



XGBoost_Model  =xgb.XGBClassifier(n_estimators=300,objective="binary:logistic", eval_metric="auc",max_depth=3,gamma=1,

learning_rate=0.01)

XGBoost_Model.fit(X_train,Y_train)

XGBoost_Model.score(X_train,Y_train)

from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=300,

                         learning_rate=1)

# Train Adaboost Classifer

Adamodel = adaboost.fit(X_train, Y_train)



#Predict the response for test dataset

print(Adamodel.score(X_train,Y_train))

y_pred_AdaBoost=Adamodel.predict(X_test)
y_pred_AdaBoost=Y_pred.astype(int)
Sub_df=pd.DataFrame({'PassengerId':test_df1['PassengerId'].values,'Survived':y_pred_AdaBoost})

Sub_df
Sub_df.to_csv('Submission.csv',index=False)