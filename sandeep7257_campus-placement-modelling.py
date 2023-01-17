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
df= pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head(5)
top10_percent = df.nlargest(round(len(df)/10),['mba_p'])
top10_percent['mba_p'].min()

def top10_perc(x):
    if x>=top10_percent['mba_p'].min():
        return 1
    else:
        return 0
df['top10_percent'] = df['mba_p'].apply(top10_perc)
df.drop('sl_no',axis=1,inplace=True)
# Importing libraries for feature engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Checking missing data
df.isnull().sum()
plt.figure(figsize=(6,4))
sns.countplot(x='status',data=df)
df.head(5)

fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(25,5))

sns.boxplot(x='status',y='ssc_p',data=df,ax=ax[0])
ax[0].set_title('ssc_p')

sns.boxplot(x='status',y='hsc_p',data=df,ax=ax[1])
ax[1].set_title('hsc_p')

sns.boxplot(x='status',y='degree_p',data=df,ax=ax[2])
ax[2].set_title('degree_p')

sns.boxplot(x='status',y='etest_p',data=df,ax=ax[3])
ax[3].set_title('etest_p')

sns.boxplot(x='status',y='mba_p',data=df,ax=ax[4])
ax[4].set_title('mba_p')
#finding columns with categorical variables

df.select_dtypes(include='object').columns
sns.countplot(x='status',data=df,hue='gender')
#Showing the effect of ctaegorical variables of placement

fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(25,10))

data = df[df['status']=='Placed'].groupby('gender').count()/df.groupby('gender').count()
sns.barplot(x=data.index,y='status',data=data,ax=ax[0][0])

data = df[df['status']=='Placed'].groupby('ssc_b').count()/df.groupby('ssc_b').count()
sns.barplot(x=data.index,y='status',data=data,ax=ax[0][1])

data = df[df['status']=='Placed'].groupby('hsc_s').count()/df.groupby('hsc_s').count()
sns.barplot(x=data.index,y='status',data=data,ax=ax[0][2])

data = df[df['status']=='Placed'].groupby('degree_t').count()/df.groupby('degree_t').count()
sns.barplot(x=data.index,y='status',data=data,ax=ax[1][0])

data = df[df['status']=='Placed'].groupby('workex').count()/df.groupby('workex').count()
sns.barplot(x=data.index,y='status',data=data,ax=ax[1][1])

data = df[df['status']=='Placed'].groupby('specialisation').count()/df.groupby('specialisation').count()
sns.barplot(x=data.index,y='status',data=data,ax=ax[1][2])
df.select_dtypes(include='object')
sns.distplot(df['salary'],kde=True)
sns.distplot(df[df['top10_percent']==1]['salary'],kde=True,bins=4)
sns.heatmap(df.corr())
# Determining effect on salary
df_salary = df.dropna()
sns.distplot(df_salary['salary'],bins=10)
fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,4))
sns.violinplot(x='gender',y='salary',data =df,hue='specialisation',ax=ax[0])
sns.violinplot(x='workex',y='salary',data =df,hue='degree_t',ax=ax[1])
sns.violinplot(x='hsc_s',y='salary',data =df,hue='hsc_b',ax=ax[2])
df.head(5)
gender = pd.get_dummies(df['gender'],drop_first=True)
df.drop('gender',axis=1,inplace=True)

ssc_b = pd.get_dummies(df['ssc_b'],drop_first=True)
df.drop('ssc_b',axis=1,inplace=True)

hsc_b = pd.get_dummies(df['hsc_b'],drop_first=True)
df.drop('hsc_b',axis=1,inplace=True)

hsc_s = pd.get_dummies(df['hsc_s'],drop_first=True)
df.drop('hsc_s',axis=1,inplace=True)

degree_t = pd.get_dummies(df['degree_t'],drop_first=True)
df.drop('degree_t',axis=1,inplace=True)

specialisation = pd.get_dummies(df['specialisation'],drop_first=True)
df.drop('specialisation',axis=1,inplace=True)

workex = pd.get_dummies(df['workex'],drop_first=True)
df.drop('workex',axis=1,inplace=True)

df = pd.concat([df,gender,ssc_b,hsc_b,degree_t,specialisation,workex],axis=1)
def change_status(x):
    if x=='Placed':
        return 1
    else:
        return 0
    
df['status']= df['status'].apply(change_status)
df.head(5)
from sklearn.model_selection import train_test_split
X=df.drop(['salary','status'],axis=1).values
y=df['status'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(12,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss ='binary_crossentropy' ,optimizer ='adam')

model.fit(X_train,y_train,epochs=500,validation_data=(X_test,y_test))
loss = pd.DataFrame(model.history.history)
plt.plot(loss)
pred = model.predict_classes(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# Logistic Regression Model 
from sklearn.model_selection import train_test_split
X=df.drop(['salary','status'],axis=1)
y=df['status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
df.head(5)
