# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df
df_copy = df.copy()
df.head()
df = df.set_index('sl_no')
df.info()
df.describe().transpose()
print(Fore.CYAN+'Total number of students',len(df['gender']))
print(Fore.CYAN+'Number of males:',len(df[df['gender']=='M']))
print(Fore.CYAN+'Number of females:',len(df[df['gender']=='F']))
box1 = sns.color_palette(['#0ead69','#003049'])
sns.countplot(df['gender'],palette=box1)
figure, axes = plt.subplots(nrows=3, ncols=2,figsize=(8,10))
swarmColor = sns.color_palette(['#da1e37','#0466c8'])
sns.swarmplot(x='gender',y='ssc_p',data=df,ax=axes[0,0],palette=swarmColor)
sns.swarmplot(x='gender',y='hsc_p',data=df,ax=axes[0,1],palette=swarmColor)
sns.swarmplot(x='gender',y='degree_p',data=df,ax=axes[1,0],palette=swarmColor)
sns.swarmplot(x='gender',y='etest_p',data=df,ax=axes[1,1],palette=swarmColor)
sns.swarmplot(x='gender',y='mba_p',data=df,ax=axes[2,0],palette=swarmColor)
axes[2,1].axis('off')
figure.tight_layout()
figure, axes = plt.subplots(nrows=4, ncols=2,figsize=(10,12))
box2 = sns.color_palette(['#390099','#ff0054'])
sns.countplot(hue='gender',x='ssc_b',data=df,ax=axes[0,0],palette=box2)
sns.countplot(hue='gender',x='hsc_b',data=df,ax=axes[0,1],palette=box2)
sns.countplot(hue='gender',x='degree_t',data=df,ax=axes[1,0],palette=box2)
sns.countplot(hue='gender',x='workex',data=df,ax=axes[1,1],palette=box2)
sns.countplot(hue='gender',x='specialisation',data=df,ax=axes[2,0],palette=box2)
sns.countplot(hue='gender',x='status',data=df,ax=axes[2,1],palette=box2)
sns.countplot(hue='gender',x='hsc_s',data=df,ax=axes[3,0],palette=box2)
axes[3,1].axis('off')
figure.tight_layout()
print(Back.RED+'Remeber there are sum null values in  salary which we will cover in data processing.')
plt.figure(figsize=(14,8))
box3 = sns.color_palette(['#00171f','#f7567c'])
sns.boxenplot(x='gender',y='salary',data=df,palette=box3)
forViolin = sns.color_palette(['#e29578','#ffddd2'])
figure, axes = plt.subplots(nrows=3, ncols=2,figsize=(12,14))
sns.violinplot(x='ssc_b',y='salary',data=df,ax=axes[0,0],hue='gender',palette=forViolin)
sns.violinplot(x='hsc_b',y='salary',data=df,ax=axes[0,1],hue='gender',palette="Paired")
sns.violinplot(x='hsc_s',y='salary',data=df,ax=axes[1,0],hue='gender',palette=forViolin)
sns.violinplot(x='degree_t',y='salary',data=df,ax=axes[1,1],hue='gender',palette="Paired")
sns.violinplot(x='workex',y='salary',data=df,ax=axes[2,0],hue='gender',palette=forViolin)
sns.violinplot(x='specialisation',y='salary',data=df,ax=axes[2,1],hue='gender',palette="Paired")
figure.tight_layout()
print(Fore.BLUE+'Observe the ssc percentage has good correlation with hsc and degree percentage.')
print(Fore.RED+'Observe the degree percentage has good correlation (not much) with MBA and hsc,ssc percentages.')
print(Fore.GREEN+'They actually make sense.Like wise observe every feature relation with other.')
plt.figure(figsize=(12,6))
sns.heatmap(df_copy.corr(),annot=True,cmap='RdPu')
print(Back.BLUE+'Observe, the ssc percentage has linear relation with hsc and degree percentage which make sense.',Back.RESET)
print(Back.BLACK+'Also, the hsc percentage has linear relation with degree percentage which also make sense.',Back.RESET)
print(Back.CYAN+'Also, the degree percentage has linear relation with hsc and ssc percentage.',Back.RESET)
print(Back.MAGENTA+'But, the MBA percentage has linear relation with degree,ssc and hsc which make sense too.',Back.RESET)
print(Back.RED+'etest percentage have positive relation with all percentage features.',Back.RESET)
sns.set_palette(sns.color_palette(['#660033']))
sns.pairplot(df.drop(['salary'],axis=1),kind='reg',markers='+')
figure, axes = plt.subplots(nrows=3, ncols=2,figsize=(12,10))
sns.distplot(df['ssc_p'], color="b", ax=axes[0, 0])
sns.distplot(df['hsc_p'], color="b", ax=axes[0, 1])
sns.distplot(df['degree_p'], color="r", ax=axes[1, 0])
sns.distplot(df['etest_p'], color="r", ax=axes[1, 1])
sns.distplot(df['mba_p'], color="g", ax=axes[2, 0])
axes[2,1].axis('off')
figure.tight_layout()
print(Fore.BLUE+'Notice ssc,hsc,degree and etest has more distribution between 60-70 percentage.More people scored between 60-70%.But its not same with MBA.')
print('We have 67 Null values in salary.')
df.isnull().sum()
df['status'].value_counts()
df = df.fillna(0)
print('0 null values.')
df.isnull().sum()
print(df.select_dtypes('object').columns)
for col in df.select_dtypes('object').columns:
    print(Fore.YELLOW+'-'*35,Fore.RESET)
    print(Fore.RED+col+Fore.RESET)
    print(Back.LIGHTYELLOW_EX+str(df[col].value_counts())+Back.RESET)

# NOTE: You can also pass drop_first=True into get_dummies function to drop first column of any two dummies generated
# To reduce the size of dataframe. How ever its not recommended if you have more that 2 catogeries in one feature.
# I will remove one column from the new columns which have only 2 catogeries we get after applying get_dummies  
# which now indicates 1 for one catogery and 0 for another. I will leave the columns that have more that 2 catogeries.

# Example: Consider gender, we have two catogeries, M and F
# Before applying get_dummies: Gender
#                                M
#                                F
#                                M
# After applying get_dummies: Gender_M Gender_F
#                                1        0
#                                0        1
#                                1        0
# After applying get_dummies, i said we can drop any one column now, observe both the columns above.If we remove Gender_F
# Then we can use Gender_M for both M and F as 1 mean 'Yes this is male' and 0 means 'No this isn't male' which is female.

a = pd.get_dummies(df['gender']).drop('M',axis=1).rename(columns={'F':'Gender'})
b = pd.get_dummies(df['ssc_b']).drop('Others',axis=1).rename(columns={'Central':'SSC_b'})
c = pd.get_dummies(df['hsc_b']).drop('Others',axis=1).rename(columns={'Central':'HSC_b'})
d = pd.get_dummies(df['hsc_s'])
e = pd.get_dummies(df['degree_t'])
f = pd.get_dummies(df['workex']).drop('No',axis=1).rename(columns={'Yes':'WorkExp'})
g = pd.get_dummies(df['specialisation']).drop('Mkt&Fin',axis=1).rename(columns={'Mkt&HR':'Mkt&HR and Mkt&Fin'})
h = pd.get_dummies(df['status']).drop('Not Placed',axis=1).rename(columns={'Placed':'Status'})

df = df.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status'],axis=1)
df = pd.concat([df,a,b,c,d,e,f,g,h],axis=1)
df
print(Back.RED+'Below are the features that are positively correlated with Placed column.Hence,these are the factor influenced a candidate in getting placed ')
df.corr()['Status'].sort_values(ascending=False)[1:7]
print(Back.RED+'In 40th cell we can see the ssc_p,hsc_p,degree_p has highest correlation compared to others which make sense and yes percentage matters.',Back.RESET)
print(Back.RED+'Seems like Comm & Mgmt degree specialization is much demanded by corporate. ')
figure,axes = plt.subplots(figsize=(12,10),ncols=2,nrows=2)
box4 = sns.color_palette(['#e8505b','#f9d56e'])
sns.swarmplot(y='salary',x='degree_t',data=df_copy,ax=axes[0,0],palette=box4)
sns.countplot(hue='gender',x='degree_t',data=df_copy,ax=axes[0,1],palette=box4)
sns.countplot(hue='specialisation',x='degree_t',data=df_copy,ax=axes[1,0],palette=box4)
sns.countplot(hue='status',x='degree_t',data=df_copy,ax=axes[1,1],palette=box4)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
# Notice: we are not dropping the 'salary' column which is highly correlated.
X = df.drop('Status',axis=1)
y = df['Status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
salary = [[0,0,0,0,0,19982,0,0,0,0,0,0,0,0,0,0,0]]
print(model.predict(salary)[0])
print('See we got 1 just by passing the salary column, 1 means Placed.')
X1 = df.drop(['Status','salary'],axis=1)
y1 = df['Status']
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.2,random_state=101)
model1 = LogisticRegression(max_iter=10000)
model1.fit(X_train,y_train)
y_pred_1 = model1.predict(X_test)
print(classification_report(y_test,y_pred_1))
print(confusion_matrix(y_test,y_pred_1))
print(accuracy_score(y_test,y_pred_1))