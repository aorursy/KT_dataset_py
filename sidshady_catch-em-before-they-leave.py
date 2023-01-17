# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 



import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





df = pd.read_csv("../input/HR_comma_sep.csv")



df.head()

# Any results you write to the current directory are saved as output.
df['sales'] = df['sales'].astype("category")



df['salary'] = df['salary'].astype("category")
sns.boxplot(x=df['left'],y=df['average_montly_hours'],data=df)
sns.boxplot(x=df['left'],y=df['time_spend_company'],data=df)
sns.countplot(x=df['left'],data=df)
sns.boxplot(x=df['left'],y=df['average_montly_hours'],data=df)
sns.countplot(x=df['number_project'],data=df,hue=df['left'])
df.groupby("left").mean()['last_evaluation']
sns.boxplot(x=df['left'],y=df['satisfaction_level'],data=df)
df['sales'].value_counts()
df.groupby(["sales"]).mean()['satisfaction_level'].sort_values(ascending=False)
plt.figure(figsize=(12,8))

sns.countplot(x=df['left'],hue=df['sales'],data=df)
df.head()
plt_df = df.groupby("sales",as_index=False).mean()[['sales','number_project','satisfaction_level']].sort_values(by='number_project',ascending=False)
plt_df.head()
sns.countplot(x=df['number_project'],data=df,hue=df['left'])
df[df['sales']=='sales'].groupby("left").mean()
df.groupby(["sales","left"]).mean()
df['promotion_last_5years'].value_counts()
df['number_project'].value_counts()
def project_counts(x) : 

    if x == 2 :

        return 2

    

    elif (x>2 and x<6) : 

        return "between 3 & 5"

    else :

        return "more than 6"

    
df['project_count'] = df['number_project'].map(project_counts)



df.head()
df.drop(["last_evaluation","number_project"],axis=1,inplace=True)
project_no_var = pd.get_dummies(df['project_count'],drop_first=True)
df = pd.concat([df,project_no_var],axis=1)
df.drop(["project_count"],axis=1,inplace=True)
df.head()
df['salary'].value_counts()
plt.figure(figsize=(12,8))



sns.boxplot(x=df['salary'],y=df['satisfaction_level'],data=df,hue=df['left'])
df.groupby(["sales","left"]).mean()[['average_montly_hours','time_spend_company','satisfaction_level']].sort_values(by="average_montly_hours",ascending=False)
df.drop("time_spend_company",axis=1,inplace=True)
df.head()
df.groupby(["salary","left"]).mean()
salaries = pd.get_dummies(df['salary'],drop_first=True)
df.drop("salary",axis=1,inplace=True)
df = pd.concat([df,salaries],axis=1)
df.head()
df.groupby(['sales','left']).size()
def combine_depts(x) : 

    

    if (x == 'IT' or x=='technical' or x=='support') : 

        return "tech"

    elif (x=='marketing' or x=='sales') : 

        return "sales & markt"

    elif ((x=='management') or (x=='product_mng') or (x=='hr')):

        return 'management'

    else :

        return x 

    
df['department'] = df['sales'].map(combine_depts)
df.drop("sales",axis=1,inplace=True)
dept_values = pd.get_dummies(df['department'],drop_first=True)
df.drop("department",axis=1,inplace=True)



df = pd.concat([df,dept_values],axis=1)
df.head()
plt.figure(figsize=(12,10))

sns.distplot(df['satisfaction_level'],kde=False)
df['satisfaction_level'].describe()
pd.cut(df['satisfaction_level'],6,include_lowest=True).value_counts()
df['satisfaction_band'] = pd.cut(df['satisfaction_level'],[0.08,0.5,0.84,1.0])
df.head()
df.groupby(['left',"satisfaction_band"]).size()
#df.drop("satisfaction_level",axis=1,inplace=True)
df.head()
def satisfaction_map(x) : 

    

    if (x >= 0.08) and (x <= 0.5) : 

        return 1 

    elif (x >0.5) and (x<=0.84) : 

        return 2

    else : 

        return 3
df['satisfaction_level'] = df['satisfaction_level'].apply(satisfaction_map)
df.drop("satisfaction_band",axis=1,inplace=True)
df.head()
from sklearn.cross_validation import train_test_split

X = df.drop("left",axis=1)

y = df['left']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression 



lmodel = LogisticRegression()
lmodel.fit(X_train,y_train)
logistic_reg_y_preds = lmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



print(confusion_matrix(y_test,logistic_reg_y_preds))

print(classification_report(y_test,logistic_reg_y_preds))
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()



dtree.fit(X_train,y_train)
dtree_y_preds = dtree.predict(X_test)
print(confusion_matrix(y_test,dtree_y_preds))

print(classification_report(y_test,dtree_y_preds))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(X_train,y_train)



rfc_y_preds = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_y_preds))

print(classification_report(y_test,rfc_y_preds))
from sklearn.svm import SVC 



svc_model = SVC()



svc_model.fit(X_train,y_train)

svc_y_preds = svc_model.predict(X_test)
print(confusion_matrix(y_test,svc_y_preds))

print(classification_report(y_test,svc_y_preds))
from sklearn import metrics

score = metrics.f1_score(y_test, svc_y_preds)

score
y_test.shape
svc_y_preds.shape
metrics.f1_score(y_test,rfc_y_preds)
metrics.f1_score(y_test,dtree_y_preds)
metrics.f1_score(y_test,svc_y_preds)