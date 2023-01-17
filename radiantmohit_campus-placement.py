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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
dataset = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
#Remove Serial Number
dataset.drop("sl_no", axis=1, inplace=True)
dataset.head()
dataset.profile_report(title='Campus Placement Data - Report')

sns.set(style="darkgrid")
sns.countplot(x = 'gender' , hue = 'status' , data = dataset)
plt.show()


sns.kdeplot(dataset.ssc_p[ dataset.status=="Placed"])
sns.kdeplot(dataset.ssc_p[ dataset.status=="Not Placed"])
plt.legend(["placed" , "not placed"])
plt.xlabel("Percentage for Secondary school")
plt.show()

sns.set(style="darkgrid")
sns.countplot(x = 'ssc_b' , hue = 'status' , data = dataset)
plt.show()

sns.kdeplot(dataset.hsc_p[ dataset.status=="Placed"])
sns.kdeplot(dataset.hsc_p[ dataset.status=="Not Placed"])
plt.legend(["placed" , "not placed"])
plt.xlabel("Higher school Percentage")
plt.show()
sns.set(style="darkgrid")
sns.countplot(x = 'hsc_b' , hue = 'status' , data = dataset)
plt.show()
sns.set(style="darkgrid")
sns.countplot(x = 'hsc_s' , hue = 'status' , data = dataset)
plt.show()
sns.kdeplot(dataset.degree_p[ dataset.status=="Placed"])
sns.kdeplot(dataset.degree_p[ dataset.status=="Not Placed"])
plt.legend(["placed" , "not placed"])
plt.xlabel("Degree Percentage")
plt.show()

sns.set(style="darkgrid")
sns.countplot(x = 'degree_t' , hue = 'status' , data = dataset)
plt.show()

sns.set(style="darkgrid")
sns.countplot(x = 'workex' , hue = 'status' , data = dataset)
plt.show()
sns.kdeplot(dataset.etest_p[dataset.status=="Placed"])
sns.kdeplot(dataset.etest_p[dataset.status=="Not Placed"])
plt.legend(["placed" , "not placed"])
plt.xlabel("ET Percentage")
plt.show()
sns.set(style="darkgrid")
sns.countplot(x = 'specialisation' , hue = 'status' , data = dataset)
plt.show()
sns.kdeplot(dataset.mba_p[dataset.status=="Placed"])
sns.kdeplot(dataset.mba_p[dataset.status=="Not Placed"])
plt.legend(["placed" , "not placed"])
plt.xlabel("MBA Percentage")
plt.show()


y = dataset.iloc[: ,[12]].values
dataset.drop(['salary' , 'ssc_b' , 'hsc_b' , 'mba_p' ,'status' ,'etest_p',
                 ]  
             , axis = 1 , inplace = True)
dataset.head()
df1 = pd.get_dummies(dataset['specialisation'], drop_first = True)
df2 = pd.get_dummies(dataset['workex'], drop_first = True)
df3 = pd.get_dummies(dataset['degree_t'], drop_first = True)
df4 = pd.get_dummies(dataset['gender'], drop_first = True)
df5 = pd.get_dummies(dataset['hsc_s'], drop_first = True)


dataset = pd.concat([df1,df2,df3,df4,df5,dataset] , axis = 1)
dataset.drop(['workex' , 'specialisation' ,'hsc_s','gender','degree_t'] , inplace  = True , axis = 1)
dataset.head()

X = dataset.iloc[:, [0,1,2,4,3,5,6,7,8,9]].values
print(X)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
ac = accuracy_score(y_test, y_pred)
print(ac)
print(classification_report(y_test, y_pred))

