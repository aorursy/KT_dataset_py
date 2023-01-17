# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
raw_data=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
raw_data.head()
raw_data.info()
raw_data.describe()
raw_data['salary']=raw_data['salary'].fillna(0.00)
sns.countplot(x='gender',data=raw_data,hue='status')
#placemet ratio gender wise

male_ratio_placed=raw_data[(raw_data['gender']=='M') & (raw_data['status']=='Placed')]['sl_no'].count()/raw_data[raw_data['gender']=='M']['sl_no'].count()
male_ratio_placed
#female Ratio placed

female_ratio_placed=raw_data[(raw_data['gender']=='F') & (raw_data['status']=='Placed')]['sl_no'].count()/raw_data[raw_data['gender']=='F']['sl_no'].count()
female_ratio_placed
#Gender ratio
gender_ratio=raw_data[raw_data['gender']=='F']['sl_no'].count()/raw_data[raw_data['gender']=='M']['sl_no'].count()
gender_ratio
#gender 
print("gender \n",raw_data['gender'].value_counts())
#Types of SCC_boards/HSC Boards/hsc subjects/degree techs
print("ssc_b \n",raw_data['ssc_b'].value_counts())
print("\n \n")
print("hsc_b \n",raw_data['hsc_b'].value_counts())
print("\n \n")
print("hsc_s \n",raw_data['hsc_s'].value_counts())
print("\n \n")
print("degree_t \n",raw_data['degree_t'].value_counts())
print("\n \n")
print("specialisation \n",raw_data['specialisation'].value_counts())
print("\n \n")
raw_data.head()
sns.countplot(x='gender',data=raw_data,hue='status')
raw_data.columns
sns.countplot(x='ssc_b',data=raw_data,hue='status')
sns.countplot(x='hsc_b',data=raw_data,hue='status')
sns.countplot(x='degree_t',data=raw_data,hue='status')
sns.countplot(x='workex',data=raw_data,hue='status')
#This shows that more chances are there if you have a work experience

raw_data.columns

sns.distplot(raw_data[raw_data['status']=='Placed']['ssc_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['ssc_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Secondary Education Percentage")
sns.distplot(raw_data[raw_data['status']=='Placed']['hsc_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['hsc_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Higher Education Percentage")
sns.distplot(raw_data[raw_data['status']=='Placed']['degree_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['degree_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Degree Education Percentage")
sns.distplot(raw_data[raw_data['status']=='Placed']['mba_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['mba_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("MBA Education Percentage")
sns.distplot(raw_data[raw_data['status']=='Placed']['etest_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['etest_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("etest_p Percentage")
sns.countplot(x='degree_t',data=raw_data,hue='status')
sns.countplot(x='specialisation',data=raw_data,hue='status')
#mapping the columns statyus to 0/1

raw_data['Placement_status']=raw_data['status'].map({'Placed':1,'Not Placed':0})
df=raw_data.copy()
df=raw_data.drop(['sl_no','status'],axis=1).copy()
gender=pd.get_dummies(df.gender,drop_first=True)
ssc_b=pd.get_dummies(df.ssc_b,drop_first=True,prefix='ssc_b')
hsc_b=pd.get_dummies(df.hsc_b,drop_first=True,prefix='hsc_b')
hsc_s=pd.get_dummies(df.hsc_s,drop_first=True,prefix='hsc_s')
degree_t=pd.get_dummies(df.degree_t,drop_first=True,prefix='degree_t')
workex=pd.get_dummies(df.workex,drop_first=True,prefix='workex')
spec=pd.get_dummies(df.specialisation,drop_first=True,prefix='mba_spec')
df_b4_dummies=df.copy()
df_after_dummies=df_b4_dummies.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'],axis=1)
df_after_dummies=pd.concat([df_after_dummies,gender,ssc_b,hsc_b,hsc_s,degree_t,workex,spec],axis=1)
df_after_dummies.head() # Data before Dropping salary
data=df_after_dummies.drop('salary',axis=1).copy()
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True)
data.columns
data_affected=data[['ssc_p', 'hsc_p', 'degree_p','mba_p','etest_p','etest_p','workex_Yes', 'mba_spec_Mkt&HR','Placement_status']].copy()
data_affected.head()
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
X=data_affected.drop('Placement_status',axis=1)
y=data_affected.Placement_status
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred_dtree=dtree.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(y_test,pred_dtree))
print("\n \n")
print(classification_report(y_test,pred_dtree))
print("\n \n")
print(accuracy_score(y_test,pred_dtree))
list(dtree.feature_importances_)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(confusion_matrix(y_test,predictions))
print("\n \n")
print(classification_report(y_test,predictions))
print("\n \n")
print(accuracy_score(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
RandForest=RandomForestClassifier()
RandForest.fit(X_train,y_train)
pred_rf=RandForest.predict(X_test)
print(confusion_matrix(y_test,pred_rf))
print("\n \n")
print(classification_report(y_test,pred_rf))
print("\n \n")
print(accuracy_score(y_test,pred_rf))
#importing a scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
X_train=MinMaxScaler().fit_transform(X_train)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
print(confusion_matrix(y_test,knn_pred))
print("\n \n")
print(classification_report(y_test,knn_pred))
print("\n \n")
print(accuracy_score(y_test,knn_pred))
df_b4_dummies.head()
df_salary=df_b4_dummies[df_after_dummies['salary']>0.0]
#df_salary=df_b4_dummies.copy()
df_salary.head()
#Distribution of Salary
plt.hist(df_salary.salary)
sns.kdeplot(df_salary.salary)
#Analysis gender wise 

sns.kdeplot(df_salary[df_salary['gender']=="M"]['salary'])

sns.kdeplot(df_salary[df_salary['gender']=="F"]['salary'])
plt.legend(["Male","female"])
plt.xlabel("Salary")
sns.boxplot(x="salary",
    y="gender",
    data=df_salary)
df_salary.ssc_b.value_counts()
sns.kdeplot(df_salary[df_salary['ssc_b']=="Central"]['salary'])

sns.kdeplot(df_salary[df_salary['ssc_b']=="Others"]['salary'])
plt.legend(["Central","Others"])
sns.boxplot(x="salary",
    y="ssc_b",
    data=df_salary)
#Analysing hsc_b
df_salary.hsc_b.value_counts()
sns.kdeplot(df_salary[df_salary['hsc_b']=="Central"]['salary'])

sns.kdeplot(df_salary[df_salary['hsc_b']=="Others"]['salary'])
plt.legend(["Others",'Central'])
sns.boxplot(x="salary",
    y="hsc_b",
    data=df_salary,hue='gender')
df_salary.hsc_s.value_counts()
sns.kdeplot(df_salary[df_salary['hsc_s']=="Commerce"]['salary'])
sns.kdeplot(df_salary[df_salary['hsc_s']=="Science"]['salary'])
sns.kdeplot(df_salary[df_salary['hsc_s']=="Arts"]['salary'])
plt.legend(["Commerce","Science","Arts"])
sns.boxplot(x='hsc_s',y='salary',data=df_salary,hue='gender')
#degree_t	workex	etest_p	specialisation	
df_salary.degree_t.value_counts()
sns.kdeplot(df[df.degree_t=='Comm&Mgmt']["salary"])
sns.kdeplot(df[df.degree_t=='Sci&Tech']["salary"])
sns.kdeplot(df[df.degree_t=='Others']["salary"])
plt.legend(['Comm&Mgmt','Sci&Tech','Others'])
sns.boxplot(x='degree_t',y='salary',data=df_salary)
sns.kdeplot(df_salary[df_salary['workex']=='Yes']['salary'])
sns.kdeplot(df_salary[df_salary['workex']=='No']['salary'])
plt.legend(["workexp yes","workexp No"])
sns.boxplot(x='workex',y='salary',data=df_salary)
df_salary['specialisation'].value_counts()
sns.kdeplot(df_salary[df_salary['specialisation']=="Mkt&Fin"]['salary'])
sns.kdeplot(df_salary[df_salary['specialisation']=="Mkt&HR"]['salary'])
plt.legend(["Mkt&Fin","Mkt&HR"])
#Average is almost same but Mkt&Fin can get more higer salary 
df_salary.head()
sns.pairplot(df_salary.drop('Placement_status',axis=1),kind='reg')
df_salary.head()
dfsal_b4dummy=df_salary.drop('Placement_status',axis=1).copy()
dfsal_b4dummy.head()
plt.figure(figsize=(10,10))
sns.heatmap(dfsal_b4dummy.corr(),annot=True)
data=dfsal_b4dummy.copy()
data.head()
data["gender"] = data.gender.map({"M":0,"F":1})
data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})
data["ssc_b"]=data.ssc_b.map({"Others":0, "Central":1})
data["hsc_b"]=data.hsc_b.map({"Others":0, "Central":1})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
data.head()
df_sal=data.copy()
plt.hist(df_sal['salary'])
len(df_sal[df_sal.salary>400000])# this means only 10 kids are outleiars we can remove it.
#df_sal=df_sal[df_sal.salary<400000]
df_sal.info()
df_sal.corr().transpose()['salary']
X=df_sal.drop(["salary",'hsc_p','ssc_p','ssc_b','workex','specialisation','degree_p','hsc_b'],axis=1)
#this is ourfinal feature selection on basis of chceking metrics by adding of removing features
y=df_sal['salary']
X.head()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=X_scaled[y <= 400000]
y=y[y <= 400000]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,random_state=41)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
from sklearn.metrics import accuracy_score,mean_absolute_error,r2_score
mean_absolute_error(y_test,reg.predict(X_test))
plt.scatter(y_test,reg.predict(X_test))
plt.xlabel('salary')
plt.ylabel('pred_sal')
