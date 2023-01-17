import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import plotly.io as pio
pio.renderers.default='notebook'
df=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.head()
df.shape
df.info()
100*df.isnull().sum()/df.shape[0]
pd.get_dummies(df['Gender'].dropna(), drop_first=True).median()
df['Gender']=df['Gender'].fillna('Male')   
df['Dependents'].unique()
df['Dependents'].dtypes
le=LabelEncoder()
le.fit(df['Dependents'].dropna())
pd.Series(le.transform(df['Dependents'].dropna())).median()
df['Dependents']=df['Dependents'].fillna('0')   
df['Self_Employed'].unique()
pd.get_dummies(df['Self_Employed'].dropna(), drop_first=True).median()
df['Self_Employed']=df['Self_Employed'].fillna('No')   
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())   
df['Loan_Amount_Term'].unique()
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())   
df['Credit_History'].unique()
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median())   
df.dropna(inplace=True)
df.isnull().sum()
df.columns=df.columns.str.lower()
df.columns=['loan_id', 'gender', 'married', 'dependents', 'education','self_employed', 'applicant_income', 'co-applicant_income', 'loan_amount', 'loan_amount_term', 'credit_history', 'property_area', 'loan_status']
df.head(2)
df.shape
df.describe()
df[df['loan_status']=='Y'].count()['loan_status']
df[df['loan_status']=='N'].count()['loan_status']
plt.figure(figsize=(10,10))
plt.pie(x=[419,192], labels=['Yes','No'], autopct='%1.0f%%', pctdistance=0.5,labeldistance=0.7,shadow=True, startangle=90, colors=['limegreen', 'deeppink'], textprops={'fontsize':14})
plt.title('Distribution of Loan Status', fontsize=18)
plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.countplot(x='gender' ,hue='loan_status', data=df,palette='plasma')
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Number', fontsize=14)

plt.subplot(2,3,2)
sns.countplot(x='married',hue='loan_status',data=df,palette='viridis')
plt.ylabel(' ')
plt.yticks([ ])
plt.xlabel('Married', fontsize=14)
plt.title('The Impacts Of Different Factors On Loan Status\n', fontsize=18)

plt.subplot(2,3,3)
sns.countplot(x='education',hue='loan_status',data=df,palette='copper')
plt.xlabel('Education', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,4)
sns.countplot(x='credit_history', data=df,hue='loan_status',palette='summer')
plt.xlabel('Credit History', fontsize=14)
plt.ylabel('Number', fontsize=14)

plt.subplot(2,3,5)
sns.countplot(x='self_employed',hue='loan_status',data=df,palette='autumn')
plt.xlabel('Self Employed', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,6)
sns.countplot(x='property_area',hue='loan_status',data=df,palette='PuBuGn')
plt.xlabel('Property Area', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])
plt.show()
df['married_revised']=df['married'].apply(lambda x: 'Married' if x=='Yes' else 'Single')
df['loan_status_revised']=df['loan_status'].apply(lambda x: 'Receive Loan' if x=='Y' else 'Not Receive Loan')
fig=px.sunburst( data_frame=df,path=['gender','married_revised','education','loan_status_revised'], color='loan_amount', color_continuous_scale='rainbow', height=800, width=800)
fig.update_layout(
    title={
        'text': 'Loan Status Versus Gender, Marrital Status And Education \n',
        'y':0.92,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
df.drop(['married_revised','loan_status_revised'], axis=1, inplace=True)
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.violinplot(x='gender', y='loan_amount',hue='loan_status', data=df,palette='plasma')
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Loan Amount', fontsize=14)

plt.subplot(2,3,2)
sns.violinplot(x='married',y='loan_amount',hue='loan_status',data=df,palette='viridis')
plt.xlabel('Married', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])
plt.title('The Impacts Of Different Factors On The Amount Of Loans\n', fontsize=18)

plt.subplot(2,3,3)
sns.violinplot(x='education',y='loan_amount',hue='loan_status',data=df,palette='copper')
plt.xlabel('Education', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,4)
sns.violinplot(x='credit_history',y='loan_amount', data=df,hue='loan_status',palette='summer')
plt.xlabel('Credit History', fontsize=14)
plt.ylabel('Loan Amount', fontsize=14)

plt.subplot(2,3,5)
sns.violinplot(x='self_employed',y='loan_amount',hue='loan_status',data=df,palette='autumn')
plt.xlabel('Self Employed', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,6)
sns.violinplot(x='property_area', y='loan_amount',data=df,hue='loan_status',palette='PuBuGn')
plt.xlabel('Property Area', fontsize=14)
plt.ylabel(' ')
plt.yticks([ ])
plt.show()
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.distplot(df['applicant_income'],bins=30,color='deeppink',hist_kws=dict(edgecolor='black'))
plt.xlabel('Applicant Income', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.subplot(1,3,2)
sns.distplot(df['co-applicant_income'],bins=30,color='cyan',hist_kws=dict(edgecolor='black'))
plt.xlabel('Co-Applicant Income', fontsize=14)
plt.title('Distribution Of Applicant Income, Co-Applicant Income and Loan Ammount\n', fontsize=18)

plt.subplot(1,3,3)
sns.distplot(df['loan_amount'],bins=30,color='lime',hist_kws=dict(edgecolor='black'))
plt.xlabel('Loan Ammount', fontsize=14)
plt.show()
fig=px.scatter_3d(data_frame=df,x='applicant_income',y='co-applicant_income',z='loan_amount',color='loan_status')
fig.update_layout(
    title={
        'text': 'Relationship Between Applicant Income, Co-Applicant Income and Loan Ammount',
        'y':0.92,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
le1=LabelEncoder()
le1.fit(df['gender'])
df['gender']=le1.transform(df['gender'])
le2=LabelEncoder()
le2.fit(df['married'])
df['married']=le2.transform(df['married'])
le3=LabelEncoder()
le3.fit(df['education'])
df['education']=le3.transform(df['education'])
le4=LabelEncoder()
le4.fit(df['self_employed'])
df['self_employed']=le4.transform(df['self_employed'])
le5=LabelEncoder()
le5.fit(df['property_area'])
df['property_area']=le5.transform(df['property_area'])
le6=LabelEncoder()
le6.fit(df['dependents'])
df['dependents']=le6.transform(df['dependents'])
X=df.drop(['loan_id','loan_status'],axis=1)
y=df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
lr=LogisticRegression()
lr.fit(X,y)
predictions_lr=lr.predict(X_test)
print(confusion_matrix(y_test,predictions_lr))
print('\n')
print(classification_report(y_test,predictions_lr))
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
predictions_dtc=dtc.predict(X_test)
print(confusion_matrix(y_test,predictions_dtc))
print("\n")
print(classification_report(y_test,predictions_dtc))
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
predictions_rfc=rfc.predict(X_test)
print(confusion_matrix(y_test,predictions_rfc))
print('\n')
print(classification_report(y_test,predictions_rfc))
error_rate=[]
for n in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    predictions_knn=knn.predict(X_test)
    error_rate.append(np.mean(predictions_knn!=y_test))
plt.figure(figsize=(10,7))
sns.set_style('whitegrid')
plt.plot(list(range(1,40)),error_rate,color='royalblue', marker='o', linewidth=2, markersize=12, markerfacecolor='deeppink', markeredgecolor='deeppink')
plt.xlabel('Number of Neighbors', fontsize=14)
plt.ylabel('Error Rate', fontsize=14)
plt.title('Elbow Method', fontsize=18)
plt.show()
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
predictions_knn=knn.predict(X_test)
print(confusion_matrix(y_test,predictions_knn))
print('\n')
print(classification_report(y_test,predictions_knn))
svc=SVC()
svc.fit(X_train,y_train)
predictions_svc=svc.predict(X_test)
print(confusion_matrix(y_test,predictions_svc))
print('\n')
print(classification_report(y_test,predictions_svc))
param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}
gs=GridSearchCV(SVC(),param_grid, verbose=3)
gs.fit(X_train,y_train)
predictions_gs=gs.predict(X_test)
print(confusion_matrix(y_test,predictions_gs))
print('\n')
print(classification_report(y_test,predictions_gs))
xgbc=xgb.XGBClassifier(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)
xgbc.fit(X_train,y_train)
predictions_xgbc=xgbc.predict(X_test)
print(confusion_matrix(y_test,predictions_xgbc))
print('\n')
print(classification_report(y_test,predictions_xgbc))
print('Accuracy Score, Logistic Regression: ', round(accuracy_score(y_test,predictions_lr),ndigits=4))
print('Accuracy Score, Decision Tree Classifier: ', round(accuracy_score(y_test,predictions_dtc),ndigits=4))
print('Accuracy Score, Random Forest Classifier: ', round(accuracy_score(y_test,predictions_rfc),ndigits=4))
print('Accuracy Score, K-Nearest Neighbors Classifier: ', round(accuracy_score(y_test,predictions_knn),ndigits=4))
print('Accuracy Score, Support Vector Classifier: ', round(accuracy_score(y_test,predictions_gs),ndigits=4))
print('Accuracy Score, XGBoost Classifier: ', round(accuracy_score(y_test,predictions_xgbc), ndigits=4))