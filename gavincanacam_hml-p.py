import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
print(os.listdir("../input"))
HomeLoan_Test=pd.read_csv('../input/Test_Loan_Home.csv') #Reading the test dataset

HomeLoan_Test.head()
HomeLoan_Train=pd.read_csv('../input/Train_Loan_Home.csv') #Reading the train dataset

HomeLoan_Train.head()
#1. Identification of variables

HomeLoan_Train_list=HomeLoan_Train.columns.tolist()

for i in range(0,len(HomeLoan_Train_list)):

    print(HomeLoan_Train_list[i])
#Identifying the datatypes:

HomeLoan_Train.dtypes

HomeLoan_Train.info() #gives detailed information on dtypes
HomeLoan_Train.isnull().sum()  #To check the complete list of missing values in each column.
HomeLoan_Train.fillna(HomeLoan_Train.mean(), inplace=True)
most_common=pd.get_dummies(HomeLoan_Train.Gender).sum().sort_values(ascending=False).index[0] #created dummies for the Gender column



def replace_most_common(x):

    if pd.isnull(x):

        return most_common

    else:

        return x



x1=HomeLoan_Train.Gender.map(replace_most_common)

print(x1)
HomeLoan_Train['Gender_Updated']=x1
HomeLoan_Train.drop('Gender', axis=1, inplace=True)
HomeLoan_Train.head()
most_common=pd.get_dummies(HomeLoan_Train.Self_Employed).sum().sort_values(ascending=False).index[0] #created dummies for the Self_Employed column



def replace_most_common(x):

    if pd.isnull(x):

        return most_common

    else:

        return x



x2=HomeLoan_Train.Self_Employed.map(replace_most_common)

print(x2)
HomeLoan_Train['Self_Employed_updated']=x2
HomeLoan_Train.head()
HomeLoan_Train.drop('Self_Employed', axis=1, inplace=True)
HomeLoan_Train.head()
most_common=pd.get_dummies(HomeLoan_Train.Dependents).sum().sort_values(ascending=False).index[0] #Created dummies for Dependents column



def replace_most_common(x):

    if pd.isnull(x):

        return most_common

    else:

        return x



x3=HomeLoan_Train.Dependents.map(replace_most_common)

print(x3)
HomeLoan_Train['Dependents_Updated']=x3
HomeLoan_Train.drop('Dependents',axis=1, inplace=True)
HomeLoan_Train.replace('3+','3', inplace=True)
HomeLoan_Train.head()
most_common=pd.get_dummies(HomeLoan_Train.Married).sum().sort_values(ascending=False).index[0] #Created dummies for Married column



def replace_most_common(x):

    if pd.isnull(x):

        return most_common

    else:

        return x



x4=HomeLoan_Train.Married.map(replace_most_common)

print(x4)
HomeLoan_Train['Married_updated']=x4
HomeLoan_Train.drop('Married',axis=1,inplace=True)
HomeLoan_Train.isnull().sum()
x2=HomeLoan_Train.Married_updated #Plotting Married_Updated column for preliminary analysis

print(x2)

x2.value_counts().plot(kind='bar')

plt.xlabel('Marital Status', fontsize=16)

plt.ylabel('count', fontsize=16)

plt.title("HomeLoan_Marital Status")

plt.show()
x3=HomeLoan_Train.Gender_Updated #Plotting Gender_Updated column for preliminary analysis

print(x3)

x3.value_counts().plot(kind='bar')

plt.xlabel('Gender', fontsize=16)

plt.ylabel('count', fontsize=16)

plt.title('Gender As Primary Applicant')

plt.show()
x5=HomeLoan_Train.Dependents_Updated #Plotting Dependents_Updated column for preliminary analysis

depend=['No_Dependents','1_Dependents','2_Dependents','3+_Dependents']

print(x5)

x5.value_counts().plot(kind='bar')

plt.xlabel('Dependents', fontsize=16)

plt.ylabel('count', fontsize=16)

plt.title('Dependents on Primary Applicant')

plt.show()
#Relationship between Property area and loan status

x12=HomeLoan_Train.groupby(['Property_Area','Loan_Status']).Loan_Status.value_counts()

print(x12)



Property_Area=['Rural','Semiurban','Urban']

Loan_Status=['Yes', 'No']

pos=np.arange(len(Property_Area))

bar_width=0.35

Loan_Status_Yes=[110,179,133]

Loan_Status_NO=[69,54,69]



plt.bar(pos,Loan_Status_Yes,bar_width,color='blue',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, Property_Area)

plt.xlabel('Property Area', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Property Area vs Loan Status',fontsize=18)

plt.legend(Loan_Status,loc=1)

plt.show()
#Relationship between Credit History and Loan Status: 

x6=HomeLoan_Train.groupby(['Credit_History','Loan_Status']).Loan_Status.value_counts()

print(x6)



Credit_History=['Bad','Medium','Good']

Loan_Status=['Yes', 'No']

pos=np.arange(len(Credit_History))

bar_width=0.35

Loan_Status_Yes=[7,37,378]

Loan_Status_NO=[82,13,97]



plt.bar(pos,Loan_Status_Yes,bar_width,color='navy',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, Credit_History)

plt.xlabel('Credit History', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Credit History vs Loan Status',fontsize=18)

plt.legend(Loan_Status,loc=2)

plt.show()
#Relationship between Gender and Loan Status:

x7=HomeLoan_Train.groupby(['Gender_Updated','Loan_Status']).Loan_Status.value_counts()

print(x7)



Gender_Updated=['Male', 'Female']

Loan_Status=['Yes', 'No']

pos=np.arange(len(Gender_Updated))

bar_width=0.30

Loan_Status_Yes=[347,75]

Loan_Status_NO=[155,37]



plt.bar(pos,Loan_Status_Yes,bar_width,color='blue',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, Gender_Updated)

plt.xlabel('Gender', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Gender vs Loan status',fontsize=18)

plt.legend(Loan_Status,loc=1)

plt.show()
#Relationship between education vs Loan status:

x8=HomeLoan_Train.groupby(['Education','Loan_Status']).Loan_Status.value_counts()

print(x8)



Education=['Graduate', 'Non-Graduate']

Loan_Status=['Yes', 'No']

pos=np.arange(len(Education))

bar_width=0.30

Loan_Status_Yes=[340,82]

Loan_Status_NO=[140,52]



plt.bar(pos,Loan_Status_Yes,bar_width,color='navy',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, Education)

plt.xlabel('Education', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Education vs Loan status',fontsize=18)

plt.legend(Loan_Status,loc=1)

plt.show()
#Relationship between Self-Employed vs Loan_Status:

x9=HomeLoan_Train.groupby(['Self_Employed_updated','Loan_Status']).Loan_Status.value_counts()

print(x9)

Self_Employed=['Yes', 'No']

Loan_Status=['Yes', 'No']

pos=np.arange(len(Self_Employed))

bar_width=0.30

Loan_Status_Yes=[56,366]

Loan_Status_NO=[26,166]



plt.bar(pos,Loan_Status_Yes,bar_width,color='navy',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, Self_Employed)

plt.xlabel('Self Employed', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Self Employed vs Loan status',fontsize=18)

plt.legend(Loan_Status,loc=1)

plt.show()
#Relationship between Dependents vs Loan status

x10=HomeLoan_Train.groupby(['Dependents_Updated','Loan_Status']).Loan_Status.value_counts()

print(x10)



Dependents=['Dpdnt_No', 'Dpdnt_1', 'Dpdnt_2', 'Dpdnt_3']

Loan_Status=['Yes', 'No']

pos=np.arange(len(Dependents))

bar_width=0.30

Loan_Status_Yes=[247,66,76,33]

Loan_Status_NO=[113,36,25,18]



plt.bar(pos,Loan_Status_Yes,bar_width,color='navy',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, Dependents)

plt.xlabel('Dependents', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Dependents vs Loan status',fontsize=18)

plt.legend(Loan_Status,loc=1)

plt.show()
#Relationship between marital status vs loan status

x11=HomeLoan_Train.groupby(['Married_updated','Loan_Status']).Loan_Status.value_counts()

print(x11)



MaritalStatus=['Yes', 'No']

Loan_Status=['Yes', 'No']

pos=np.arange(len(MaritalStatus))

bar_width=0.30

Loan_Status_Yes=[288,134]

Loan_Status_NO=[113,79]



plt.bar(pos,Loan_Status_Yes,bar_width,color='navy',edgecolor='black')

plt.bar(pos+bar_width,Loan_Status_NO,bar_width,color='red',edgecolor='black')

plt.xticks(pos, MaritalStatus)

plt.xlabel('Marital Status', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.title('Marital Status vs Loan status',fontsize=18)

plt.legend(Loan_Status,loc=1)

plt.show()
HomeLoan_Train.Education=HomeLoan_Train.Education.map({'Not Graduate':0,'Graduate':1})
HomeLoan_Train.Property_Area=HomeLoan_Train.Property_Area.map({'Rural':0,'Semiurban':1,'Urban':2})
HomeLoan_Train.Loan_Status=HomeLoan_Train.Loan_Status.map({'N':0,'Y':1})
HomeLoan_Train.Self_Employed_updated=HomeLoan_Train.Self_Employed_updated.map({'No':0,'Yes':1})
HomeLoan_Train.Married_updated=HomeLoan_Train.Married_updated.map({'No':0,'Yes':1})
HomeLoan_Train.Gender_Updated=HomeLoan_Train.Gender_Updated.map({'Female':0,'Male':1})
HomeLoan_Train.head()
HomeLoan_Train.drop(['Loan_ID'], axis=1, inplace=True) #Drop the column Loan_ID
HomeLoan_Train.head()
X=HomeLoan_Train.drop(['Loan_Status'], axis=1)
y=HomeLoan_Train.Loan_Status
import seaborn as sns
plt.figure(figsize=(10,10))

import math

cor = abs(HomeLoan_Train.corr())

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
max_accu=0 #Importing the model

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

estimator = LogisticRegression()

for  i  in range(1,len(X.iloc[0])+1):

    selector =RFE(estimator, i, step=1)

    selector = selector.fit(X,y)

    accuracy = selector.score(X,y)

    if max_accu < accuracy:

        sel_features = selector.support_

        max_accu =accuracy

 

X_sub = X.loc[:,sel_features]
#Data Preprocessing

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_scaled = pd.DataFrame(sc_X.fit_transform(X), columns=X.columns)
#split train and test sets



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
#import classifier

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

classifier.score(X_test,y_test) #classifier performance on test set
# importing performance measuring tools

from sklearn.metrics import accuracy_score, confusion_matrix,recall_score,precision_score,classification_report



recall_score(y_test,y_pred,average='macro')
cr=classification_report(y_test,y_pred)

print(cr)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred,average='macro')
#import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=21,weights='distance',p=1)

model.fit(X_train,y_train)

model.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(X_train,y_train)

model.score(X_test,y_test)

param_dict=({'n_neighbors':range(3,11,2),'weights':['uniform','distance'],'p':[1,2,3,4,5]})

from sklearn.model_selection import GridSearchCV

best_model=GridSearchCV(model,param_dict,cv=5)

best_model.fit(X_scaled,y)

best_model.best_params_

best_model.best_score_
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(max_depth=25)

model2.fit(X_train,y_train)

model2.score(X_test,y_test)

param_dict_2=({'n_estimators':range(2,50)})

from sklearn.model_selection import GridSearchCV

best_model=GridSearchCV(model2,param_dict_2,cv=5)

best_model.fit(X_scaled,y)

best_model.best_params_

best_model.best_score_
from sklearn.ensemble import AdaBoostClassifier

model3 = AdaBoostClassifier(n_estimators=20)

model3.fit(X_train,y_train)

model3.score(X_test,y_test)

param_dict_3=({'n_estimators':range(2,50)})

from sklearn.model_selection import GridSearchCV

best_model=GridSearchCV(model3,param_dict_3,cv=5)

best_model.fit(X_scaled,y)

best_model.best_params_

best_model.best_score_
#Support vector Machine model

from sklearn.svm import SVC

model_svc = SVC(kernel='linear',gamma=0.001,C=1.0)

model_svc.fit(X_train,y_train)

model_svc.score(X_test,y_test)
#Estimating the best model using Cross-validation

new_model=best_model.best_estimator_ #gives the best model estimation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

cross_val_score(new_model, X_scaled,y,cv=5).mean()
print(new_model)