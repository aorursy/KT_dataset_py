import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

df=pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv',encoding='utf-8', engine='c')

df.head()

b=df.isnull()

sns.heatmap(b,annot=True)

df.isnull().sum()

df.describe().transpose()

df.info()

sns.barplot('Education','MonthlyIncome',data=df,hue='Attrition')



#categorical cols

categorical_col=[]

for col,value in df.iteritems():

    if value.dtype=='object':

        categorical_col.append(col)

df_col=df[categorical_col]

df_col.head()



for col in categorical_col:

    print(col,df[col].unique())





df.hist(figsize=(20,20))



#checking for correlation

plt.figure(figsize=(25,25))

sns.heatmap(df.corr(),annot=True)



#Relationship between Attrition and different features

plt.figure(figsize=(10,10))

plt.subplot(4,2,1)

sns.countplot('Age',data=df,hue='Attrition')

plt.subplot(4,2,2)

sns.countplot('MaritalStatus',data=df,hue='Attrition')

plt.subplot(4,2,3)

sns.countplot('JobLevel',data=df,hue='Attrition')

plt.subplot(4,2,4)

sns.countplot('JobRole',data=df,hue='Attrition')

plt.subplot(4,2,5)

sns.countplot('JobSatisfaction',data=df,hue='Attrition')

plt.subplot(4,2,6)

sns.countplot('OverTime',data=df,hue='Attrition')

plt.subplot(4,2,7)

sns.countplot('TotalWorkingYears',data=df,hue='Attrition')

plt.subplot(4,2,8)

sns.countplot('Gender',data=df,hue='Attrition')



#converting categorical variables into numerical vales

df=pd.get_dummies(df,drop_first=True)

df.head()



df.drop(['EmployeeNumber','EmployeeCount','StandardHours'],axis=1,inplace=True)#they all have same values



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import accuracy_score



x=df.drop('Attrition_Yes',axis=1)

y=df['Attrition_Yes']



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)



model=RandomForestClassifier(n_estimators=1000)

model.fit(x_train,y_train)

pred=model.predict(x_test)

ac=accuracy_score(y_test,pred)

print(ac)



#feature selection, optional

imp_feat=pd.DataFrame(model.feature_importances_,index=x.columns)

print(imp_feat)

imp_feat.plot(kind='bar')





cm=confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,cmap='Blues_r')

plt.xlabel('Predicted')

plt.ylabel('Original')

title='Accuracy:{} '.format(ac)

plt.title(title)

print(classification_report(y_test,pred))



#using randomized search to tune up parameters

n_estimators=[int(x) for x in np.linspace(100,1000,10)]

criterion=['gini','entropy']

max_depth=[int(x) for x in np.linspace(5,20,4)]

max_features=['auto','sqrt']

min_samples_leaf=[1,2,5,10]

min_samples_split=[1,5,10,20,100]



params={'criterion':criterion,'max_depth':max_depth,'max_features':max_features,'min_samples_leaf':min_samples_leaf,

        'min_samples_split':min_samples_split,'n_estimators':n_estimators}

print(params)



new_model=RandomizedSearchCV(RandomForestClassifier(),params,verbose=2,n_iter=10,n_jobs=1,

                             scoring='neg_mean_squared_error')



new_model.fit(x_train,y_train)

pred1=new_model.predict(x_test)



print(new_model.best_params_)

print(new_model.best_score_)



ac1=accuracy_score(y_test,pred1)

print(ac1)



cm1=confusion_matrix(y_test,pred1)

sns.heatmap(cm1,annot=True)