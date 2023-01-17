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
import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/diabetes-data/diabetes-data-for-R.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(10,6))
sns.countplot(x='age',data=df)
sns.countplot(x='smoking',hue='dm',data=df)
df.head()
df.tail()
plt.figure(figsize=(10,7))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop(df.columns[df.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
df.head()
df.drop('id',axis=1,inplace=True)
df.head()
plt.figure(figsize=(10,7))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.head(4)
sns.jointplot(x='age',y='chol',data=df)
df[df['chol'].isnull()]['age']
df[df['age']==48]['chol'].mean()
def chol_cleaner(col):
    chol =col[0]
    age = col[1]
    if pd.isnull(chol):
        if age==27:
            return 204
        else:
            return 198
    else:
        return chol
df['chol'] = df[['chol','age']].apply(chol_cleaner,axis=1)
df.head()
plt.figure(figsize=(8,7))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.head(5)
plt.figure(figsize=(10,8))
df['hdl'].hist(bins=30)
plt.figure(figsize=(10,8))
sns.boxplot(x='hdl',y='age',data=df)
df[df['chol'].isnull()]['age']
df[df['hdl'].isnull()]['age']
import cufflinks as cf
cf.go_offline()
df['hdl'].iplot(kind='hist',bins=30)
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.head()
sns.distplot(df['hdl'],kde=False,bins=30)
plt.figure(figsize=(10,10))
sns.boxplot(x='hdl',y='age',data=df)
df
df[df['age']==27]['hdl']
df[df['age']==48]['hdl'].mean()
def clean_hdl(col):
    hdl = col[0]
    age = col[1]
    if pd.isnull(hdl):
        if age==27:
            return 54
        else:
            return 54
    else:
        return hdl
df['hdl'] = df[['hdl','age']].apply(clean_hdl,axis=1)
df.head()
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.head()
df[df['age']==48]['ratio'].mean()
def clean_ratio(col):
    ratio = col[0]
    age = col[1]
    if pd.isnull(ratio):
        if age == 27:
            return 4.0
        else:
            return 3.9
    else:
        return ratio
df['ratio'] = df[['ratio','age']].apply(clean_ratio,axis=1)
df.head()
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
sns.countplot(x='glyhb',data=df)
df[df['glyhb'].isnull()]['age']
df[df['age']>60] ['glyhb'].mean()
#sns.jointplot(x='glyhb',y='age',data=df)
def clear_gly_hub(col):
    glyhb = col[0]
    age = col[1]
    if pd.isnull(glyhb):
        if age>30 and age<40:
            return 5.8
        elif age>40 and age<50:
            return 6.1
        else:
            return 6.6
    else:
        return glyhb
df['glyhb'] = df[['glyhb','age']].apply(clear_gly_hub,axis=1)
df.head()
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df[df['height'].isnull()]['weight']
df[df['weight']>125]['height']
def clean_height(col):
    height = col[0]
    age = col[1]
    if pd.isnull(height):
        return 66.0
    else:
        return height
    
df['height'] = df[['height','age']].apply(clean_height,axis=1)
df.head()
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df[df['weight'].isnull()]['height']
df[df['height']==69]['weight'].mean()
def clean_weight(col):
    weight = col[0]
    height = col[1]
    if pd.isnull(weight):
        return 186
    else:
        return weight
df['weight'] = df[['weight','height']].apply(clean_weight,axis=1)
df.head()
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df[df['waist'].isnull()]['height'].mean()
def clean_waist(col):
    waist = col[0]
    if pd.isnull(waist):
        return 63.5
    else:
        return waist
    
df['waist'] = df[['waist','height']].apply(clean_waist,axis=1)
df.head()
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df[df['hip'].isnull()]['waist']
def clean_hip(col):
    hip = col[0]
    if pd.isnull(hip):
        return 63.5
    else:
        return hip
df['hip'] = df[['hip']].apply(clean_hip,axis=1)
df.head()
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.columns
df.info()
df['frame'].hist(bins=30)
sns.jointplot(x = 'time.ppn',y='age',data=df)
df[df['time.ppn'].isnull()]['gender']
df[df['age']==41]['time.ppn']
df[df['time.ppn'].isnull()]['age']
df.head()
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.drop(['bp.2s','bp.2s'],axis=1)
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
new_df = df.drop(['bp.2s','bp.2d','frame'],axis=1)
new_df.head()
sns.heatmap(new_df.isnull(),cbar=False,cmap='viridis')
new_df
sns.heatmap(new_df.isnull(),cbar=False,cmap='viridis')
target = pd.get_dummies(new_df['dm'],drop_first=True)
target
new_df = pd.concat([new_df,target],axis=1)
new_df.head()
sns.heatmap(new_df.isnull(),cbar=False,cmap='viridis')
new_df[new_df['dm'].isnull()].count()
new = new_df.dropna()
gender = pd.get_dummies(new['gender'],drop_first=True)
new = pd.concat([new,gender],axis=1)
new.info()
sns.heatmap(new.isnull(),cbar = False,cmap='viridis')
new.head()
final = new.drop(['gender','location','dm'],axis=1)
final.head()
final.info()
from sklearn.preprocessing import StandardScaler
myscaler = StandardScaler()
final.head()
myscaler.fit(final.drop('yes',axis=1))
feat_scaler = myscaler.transform(final.drop('yes',axis=1))

scaled_df = pd.DataFrame(feat_scaler,columns = final.columns[:-1])
scaled_df.columns
scaled_df.columns = ['chol', 'stab.glu', 'hdl', 'ratio', 'glyhb', 'age', 'height', 'weight',
       'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn', 'insurance', 'fh',
       'smoking', 'gender']
scaled_df.head()
X = scaled_df
y = final['yes']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


