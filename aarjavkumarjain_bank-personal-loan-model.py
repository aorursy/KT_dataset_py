import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
#path of your csv file

path='../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx'

df=pd.read_excel(path,'Data')

df.head()
df.shape
df.describe()
df.isnull().sum()
df.info()
df.nunique()
categorical_variables=[col for col in df.columns if df[col].nunique()<=5]

print(categorical_variables)

continuous_variables=[col for col in df.columns if df[col].nunique()>5]

print(continuous_variables)
categorical_variables.remove("Personal Loan")

print(categorical_variables)

continuous_variables.remove("ID")

print(continuous_variables)
fig=plt.figure(figsize=(20,10))

#fig.subplots_adjust(wspace=0.4,hspace=0.4)

for i,col in enumerate(continuous_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.distplot(df[col])
fig=plt.figure(figsize=(20,10))

#fig.subplots_adjust(wspace=0.4,hspace=0.4)

for i,col in enumerate(categorical_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.countplot(df[col])
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(continuous_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.boxplot(y=df[col],x=df['Personal Loan'])
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(continuous_variables):

    ax=fig.add_subplot(2,3,i+1)

    ax1=sns.distplot(df[col][df['Personal Loan']==0],hist=False,label='No Personal Lone')

    sns.distplot(df[col][df['Personal Loan']==1],hist=False,ax=ax1,label='Personal Lone')
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(categorical_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.barplot(x=col,y='Personal Loan',data=df,ci=None)
con=continuous_variables.copy()

con.remove('Income')
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(con):

    ax=fig.add_subplot(2,3,i+1)

    sns.scatterplot('Income',col,hue='Personal Loan',data=df)
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(categorical_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.scatterplot(col,'Income',hue='Personal Loan',data=df)
con.remove('CCAvg')
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(con):

    ax=fig.add_subplot(2,2,i+1)

    sns.scatterplot('CCAvg',col,hue='Personal Loan',data=df)
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(categorical_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.scatterplot(col,'CCAvg',hue='Personal Loan',data=df)
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(categorical_variables):

    ax=fig.add_subplot(2,3,i+1)

    sns.countplot(x=col,hue='Personal Loan',data=df)
df.drop_duplicates(inplace=True)
df.shape
df.set_index("ID",inplace=True)
df.drop('ZIP Code',axis=1,inplace=True)
corr=df.corr()

plt.figure(figsize=(10,10))

plt.title('Correlation')

sns.heatmap(corr > 0.90, annot=True, square=True)
df[['Age','Experience','Personal Loan']].corr()
df.drop('Experience',axis=1,inplace=True)
df['Account']=df['CD Account']+df['Securities Account']
df[['CD Account','Securities Account','Account','Personal Loan']].corr()
df.drop('Account',axis=1,inplace=True)
df['Facilities']=df['Online']+df['CreditCard']
df[['Facilities','Online','CreditCard','Personal Loan']].corr()
df.drop(['Online','CreditCard'],axis=1,inplace=True)
df.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaled_df=scaler.fit_transform(df.drop('Personal Loan',axis=1))
scaled_df=pd.DataFrame(scaled_df)
scaled_df.columns=df.drop('Personal Loan',axis=1).columns

scaled_df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
X=scaled_df

y=df['Personal Loan']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)
model_list=[]

model_f1_score=[]

model_accuracy_score=[]
model_list.append('LogisticRegression')

lm=LogisticRegression()
lm.fit(x_train,y_train)
yhat_lm=lm.predict(x_test)
lm_score=f1_score(y_test,yhat_lm)

model_f1_score.append(lm_score)

lm_score
lm_accuracy=accuracy_score(y_test,yhat_lm)

model_accuracy_score.append(lm_accuracy)

lm_accuracy
print(classification_report(y_test,yhat_lm))
sns.heatmap(confusion_matrix(y_test,yhat_lm),annot=True,fmt='',cmap='YlGnBu')
model_list.append('DecisionTreeClassifier')

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
yhat_tree=tree.predict(x_test)
tree_score=f1_score(y_test,yhat_tree)

model_f1_score.append(tree_score)

tree_score
tree_accuracy=accuracy_score(y_test,yhat_tree)

model_accuracy_score.append(tree_accuracy)

tree_accuracy
print(classification_report(y_test,yhat_tree))
sns.heatmap(confusion_matrix(y_test,yhat_tree),annot=True,fmt='',cmap='YlGnBu')
model_list.append('RandomForestClassifier')

forest=RandomForestClassifier()
forest.fit(x_train,y_train)
yhat_forest=forest.predict(x_test)
forest_score=f1_score(y_test,yhat_forest)

model_f1_score.append(forest_score)

forest_score
forest_accuracy=accuracy_score(y_test,yhat_forest)

model_accuracy_score.append(forest_accuracy)

forest_accuracy
print(classification_report(y_test,yhat_forest))
sns.heatmap(confusion_matrix(y_test,yhat_forest),annot=True,fmt='',cmap='YlGnBu')
model_list.append('SVC')

svc=SVC()
svc.fit(x_train,y_train)
yhat_svc=svc.predict(x_test)
svc_score=f1_score(y_test,yhat_svc)

model_f1_score.append(svc_score)

svc_score
svc_accuracy=accuracy_score(y_test,yhat_svc)

model_accuracy_score.append(svc_accuracy)

svc_accuracy
print(classification_report(y_test,yhat_svc))
sns.heatmap(confusion_matrix(y_test,yhat_svc),annot=True,fmt='',cmap='YlGnBu')
model_list.append('KNeighborsClassifier')

neighbour=KNeighborsClassifier()
neighbour.fit(x_train,y_train)
yhat_neighbour=neighbour.predict(x_test)
neighbour_score=f1_score(y_test,yhat_neighbour)

model_f1_score.append(neighbour_score)

neighbour_score
neighbour_accuracy=accuracy_score(y_test,yhat_neighbour)

model_accuracy_score.append(neighbour_accuracy)

neighbour_accuracy
print(classification_report(y_test,yhat_neighbour))
sns.heatmap(confusion_matrix(y_test,yhat_neighbour),annot=True,fmt='',cmap='YlGnBu')
fig,ax=plt.subplots(figsize=(10,8))

sns.barplot(model_list,model_f1_score)

ax.set_title("F1 Score of  Test Data",pad=20)

ax.set_xlabel("Models",labelpad=20)

ax.set_ylabel("F1_Score",labelpad=20)

plt.xticks(rotation=90)



for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))
fig,ax=plt.subplots(figsize=(10,6))

sns.barplot(model_list,model_accuracy_score)

ax.set_title("Accuracy of Models on Test Data",pad=20)

ax.set_xlabel("Models",labelpad=20)

ax.set_ylabel("Accuracy",labelpad=20)

plt.xticks(rotation=90)



for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))