import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv',index_col='EmployeeNumber')
df.head()
df.info()
plt.figure(figsize=(10,6))

sns.heatmap(df.isna(),cmap='viridis',cbar=False, yticklabels=False);
#no null value present as seen from above heatmap.
for col in df.columns:

    print("{}:{}".format(col,df[col].nunique()))

    print("=======================================")
df.drop(columns=['Over18','StandardHours','EmployeeCount'],inplace=True)
df['Attrition']=df['Attrition'].map({'Yes':1, 'No':0})
categorical_col=[]

for col in df.columns:

    if df[col].dtype== object and df[col].nunique()<=50:

        categorical_col.append(col)

print(categorical_col)
for col in categorical_col:

    print("{}:\n{}".format(col,df[col].value_counts()))

    print("=======================================")
df.columns
sns.countplot(x='Attrition',data=df)
sns.countplot(x='Attrition',hue='PerformanceRating',data=df)
sns.countplot(x='Attrition',hue='JobInvolvement',data=df)
sns.scatterplot(x='Age',y='MonthlyIncome',data=df)
sns.kdeplot(df['Age'],df['MonthlyIncome'],shade=True,cbar=True)
plt.figure(figsize=(18,12))

sns.heatmap(df.corr(),cmap='RdYlGn',annot=True,fmt='.2f')
df.corr()['Attrition'].sort_values(ascending=False)
sns.set(font_scale=2)

plt.figure(figsize=(30,30))

for i,col in enumerate(categorical_col,1):

    plt.subplot(3,3,i)

    sns.barplot(x=f"{col}",y='Attrition',data=df)

    plt.xticks(rotation=90)

plt.tight_layout()

sns.set(font_scale=1)

sns.boxplot(x='JobRole',y='MonthlyIncome',data=df)

plt.xticks(rotation=90);
sns.boxplot(x='EducationField',y='MonthlyIncome',data=df)

plt.xticks(rotation=90);
sns.violinplot(x='EducationField',y='MonthlyIncome',data=df,hue='Attrition',color='Yellow',split=True)

plt.legend(bbox_to_anchor=(1.2,0.65))

plt.xticks(rotation=45);
plt.subplots(figsize=(15,5))

sns.countplot(x='TotalWorkingYears',data=df)
plt.figure(figsize=(6,6))

plt.pie(df['EducationField'].value_counts(),labels=df['EducationField'].value_counts().index,autopct='%.2f%%');
df['EducationField'].value_counts()
df.groupby(by='JobRole')["PercentSalaryHike","YearsAtCompany","TotalWorkingYears","YearsInCurrentRole","WorkLifeBalance"].mean()
plt.figure(figsize=(6,6))

plt.pie(df['JobRole'].value_counts(),labels=df['JobRole'].value_counts().index,autopct='%.2f%%');

plt.title('Job Role Distribution',fontdict={'fontsize':22});
plt.figure(figsize=(14,5))

sns.countplot(x='Age',data=df)
sns.barplot(x='Education',y='MonthlyIncome',hue='Attrition',data=df)

plt.legend(bbox_to_anchor=(1.2,0.6))
sns.barplot(y='DistanceFromHome',x='JobRole',hue='Attrition',data=df,dodge=False,alpha=0.4,palette='twilight')

plt.xticks(rotation=90);

plt.legend(bbox_to_anchor=(1.2,0.6));
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
for col in categorical_col:

    df[col]=le.fit_transform(df[col])
from sklearn.model_selection import train_test_split
data= df.copy()
X= data.drop('Attrition',axis=1)

y=data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()
model.fit(X_train,y_train)
pred= model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.model_selection import RandomizedSearchCV
params={"criterion":("gini", "entropy"),

        "splitter":("best", "random"), 

        "max_depth":(list(range(1, 20))), 

        "min_samples_split":[2, 3, 4], 

        "min_samples_leaf":list(range(1, 20))}
tree_random= RandomizedSearchCV(model,params,n_iter=100,n_jobs=-1,cv=3,verbose=2)
tree_random.fit(X_train,y_train)
tree_random.best_estimator_
model=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',

                       max_depth=8, max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=6, min_samples_split=4,

                       min_weight_fraction_leaf=0.0, presort='deprecated',

                       random_state=None, splitter='random')
model.fit(X_train,y_train)

pred=model.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred= rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))