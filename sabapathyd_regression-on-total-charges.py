import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsRegressor as Knn

from sklearn.metrics import mean_squared_error as mse

from sklearn.tree import DecisionTreeRegressor

import math as m
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()

data.shape
data.isna().sum()

data.dtypes
data.drop(columns=['customerID','Churn'],axis=1,inplace=True)
data['SeniorCitizen'].replace([1],'Yes',inplace=True)

data['SeniorCitizen'].replace([0],'No',inplace=True)     

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')

data.drop(index=data.loc[data['TotalCharges'].isna()].index,axis=0,inplace=True)
data1=data.copy(deep=True)

data.head()
sc = data.groupby(['gender','SeniorCitizen']).mean()

sc.reset_index(inplace=True)

sc

#px.bar(sc,x='gender',y='TotalCharges',facet_col='SeniorCitizen',category_orders={'SeniorCitizen':['No','Yes']})
part = data.groupby('Partner').sum()

part.reset_index(inplace=True)

part

px.pie(names=part['Partner'],values=part['TotalCharges'])
dep = data.groupby('Dependents').sum()

dep.reset_index(inplace=True)

px.pie(names=dep['Dependents'],values=dep['TotalCharges'])
deppat = data.groupby(['Partner','Dependents']).mean()

deppat.reset_index(inplace=True)

px.bar(deppat,x='Partner',y='MonthlyCharges',facet_col='Dependents')

gen = data.groupby(['gender','SeniorCitizen','Partner','Dependents']).mean()

gen.reset_index(inplace=True)

px.bar(gen,y='TotalCharges',x='gender',facet_col='SeniorCitizen',facet_row='Partner',color='Dependents',barmode='group')
px.scatter(x=data['tenure'],y=data['MonthlyCharges'],labels={'x':'tenure','y':'MonthlyCharges'})
ps = data.groupby('PhoneService').sum()

ps.reset_index(inplace=True)

px.pie(names=ps['PhoneService'],values=ps['TotalCharges'])
ml=data.groupby('MultipleLines').sum()

ml.reset_index(inplace=True)

px.pie(names=ml['MultipleLines'],values=ml['TotalCharges'])
Is = data.groupby('InternetService').sum()

Is.reset_index(inplace=True)

px.pie(names=Is['InternetService'],values=Is['TotalCharges'])
npi = data.groupby(['PhoneService','InternetService']).sum()

#npi.get_group(('No','Fiber optic'))['Total Charges'].sum()

npi.reset_index(inplace=True)

npi
data['Contract'].value_counts()
cont = data.groupby('Contract').mean()

cont.reset_index(inplace=True)

px.scatter(data,x='TotalCharges',y='tenure',facet_col='Contract')
for x in data.columns:

    print(x ,' : ' ,data[x].unique())

    print(data[x].value_counts())

    print('\n')
data.shape
noweb = data[data['InternetService'] =='No']

nophone = data[data['MultipleLines'] =='No phone service']



data.drop(index=noweb.index,axis=0,inplace=True)

data.drop(index=nophone.index,axis=0,inplace=True)

data.drop(columns=['MonthlyCharges'],inplace=True,axis=1)
data.head()
data_mod = pd.get_dummies(data,columns=['InternetService','Contract','PaymentMethod'],drop_first=True)

#'Gender','Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security'

data_mod.replace(to_replace='No',value='0',inplace=True)

data_mod.replace(to_replace='Yes',value='1',inplace=True)

data_mod['gender'].replace(to_replace='Female',value='1',inplace=True)

data_mod['gender'].replace(to_replace='Male',value='0',inplace=True)

data_mod.head()

#cor = 

data_mod.corr()

#plt.figure(figsize=(40,40))

#sns.heatmap(cor, annot=True)

#plt.rcParams['figure.figsize'] = [40, 40]

#plt.rcParams['figure.dpi'] = 100

#data_mod.corr()
data_mod.drop(index=data.loc[data_mod['TotalCharges'].isna()].index,axis=0,inplace=True)

x = data_mod.drop(columns='TotalCharges',axis=1)

y = data_mod['TotalCharges']
x.isna().sum()

y.isna().sum()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)
model = sm.OLS(y_train.astype(float),sm.add_constant(x_train.astype(float))).fit()

print(model.summary()) 
def backward_regression(X, y,

                           initial_list=[], 

                           threshold_in=0.01, 

                           threshold_out = 0.05, 

                           verbose=True):

    included=list(X.columns)

    while True:

        changed=False

        model = sm.OLS(y.astype(float), sm.add_constant(pd.DataFrame(X[included].astype(float)))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.idxmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {} with p-value {} '.format(worst_feature, worst_pval))

        if not changed:

            break

    return included
backward_regression(x,y)
data_mod.drop(columns='PaymentMethod_Credit card (automatic)',axis=1,inplace=True)

data_mod.drop(columns='Contract_One year',axis=1,inplace=True) 

data_mod.drop(columns='PaymentMethod_Electronic check',axis=1,inplace=True) 

data_mod.drop(columns='SeniorCitizen',axis=1,inplace=True) 

data_mod.drop(columns='Partner',axis=1,inplace=True) 

data_mod.drop(columns='PaperlessBilling',axis=1,inplace=True) 

data_mod.drop(columns='Dependents',axis=1,inplace=True)
x = data_mod.drop(columns='TotalCharges',axis=1)

y = data_mod['TotalCharges']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)
model = sm.OLS(y_train.astype(float),sm.add_constant(x_train.astype(float))).fit()

print(model.summary()) 
model1 = LinearRegression()

model1.fit(x_train,y_train)

predict = model1.predict(x_test)



print(r2_score(y_test,predict))



noweb.drop(noweb.iloc[:,7:14],axis=1,inplace=True)

noweb.drop(columns='MonthlyCharges',axis=1,inplace=True)

noweb
noweb.replace(to_replace='No',value='0',inplace=True)

noweb.replace(to_replace='Yes',value='1',inplace=True)



noweb.replace(to_replace='Male',value='1',inplace=True)

noweb.replace(to_replace='Female',value='0',inplace=True)

noweb_mod = pd.get_dummies(noweb,columns=['Contract','PaymentMethod'],drop_first=True)

noweb_mod
corr1 = noweb_mod.corr()

sns.heatmap(corr1,annot=True)
x9 = noweb_mod.drop(columns='TotalCharges')

y9 = noweb_mod['TotalCharges']
x_train5,x_test5,y_train5,y_test5=train_test_split(x9,y9,test_size=0.3,random_state=125)
model2 = sm.OLS(y_train5.astype(float),sm.add_constant(x_train5.astype(float))).fit()

print(model2.summary())
model5 = LinearRegression()

model5.fit(x_train5,y_train5)

predict = model5.predict(x_test5)



print(r2_score(y_test5,predict))
data_mod1 = pd.get_dummies(data1,columns=['InternetService','Contract','PaymentMethod','MultipleLines','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])

#'Gender','Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security'

data_mod1.replace(to_replace='No',value='0',inplace=True)

data_mod1.replace(to_replace='Yes',value='1',inplace=True)

data_mod1['gender'].replace(to_replace='Female',value='1',inplace=True)

data_mod1['gender'].replace(to_replace='Male',value='0',inplace=True)

data_mod1.head()
x1 = data_mod1.drop(columns=['TotalCharges'],axis=1)

y1 = data_mod1['TotalCharges']
x_train1,x_test1,y_train1,y_test1 = train_test_split(x1,y1,test_size=0.3,random_state=122)

x_train1.shape

x_test1.shape

y_train1.shape

y_test1.shape
scaler = MinMaxScaler()

x_train_stand = scaler.fit_transform(x_train1)

x_test_stand = scaler.fit_transform(x_test1)
mse1 = []

r2 = []

for x in range(1,27):

  kNN = Knn(n_neighbors=x,p=2,metric='minkowski')

  kNN.fit(x_train_stand,y_train1)

  predictKnn = kNN.predict(x_test_stand)

  #mse1.append(mse(y_test1,predictKnn))

  r2.append(kNN.score(x_test_stand,y_test1))


values = pd.Series(r2)

#knnvalue = np.hstack((index,values))

#values = np.array(values)

print(values)

plt.plot(values.index,values)

plt.xticks(range(0,28))

plt.show()
kNN = Knn(n_neighbors=11,p=2)

kNN.fit(x_train_stand,y_train1)

predictKnn = kNN.predict(x_test_stand)
r2_score(y_test1,predictKnn)
data_mod2 = pd.get_dummies(data,columns=['InternetService','Contract','PaymentMethod','MultipleLines','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])

data_mod2.replace(to_replace='No',value='0',inplace=True)

data_mod2.replace(to_replace='Yes',value='1',inplace=True)

data_mod2['gender'].replace(to_replace='Female',value='1',inplace=True)

data_mod2['gender'].replace(to_replace='Male',value='0',inplace=True)

data_mod2.head()

#data_mod2.drop(columns=['Monthly Charges'],axis=1,inplace=True)
x3= data_mod2.drop(columns='TotalCharges',axis=1)

y3 = data_mod2['TotalCharges']
x_train3,x_test3,y_train3,y_test3 = train_test_split(x3,y3,test_size=0.3,random_state=121)
tree1 = DecisionTreeRegressor()

tree1.fit(x_train3,y_train3)

predict3 = tree1.predict(x_test3)

r2_score(y_test3,predict3)
x = pd.DataFrame(data_mod2.columns)

x.columns=['Feature']

x.reset_index(inplace=True)

x.drop(columns='index',axis=1,inplace=True)

x.drop(index=5,inplace=True)

tree1.feature_importances_.shape

x['Importance']=tree1.feature_importances_

x.sort_values(by='Importance',ascending=False,inplace=True)

px.bar(x=x.Feature,y=x.Importance,labels={'y':'Score'},title="Feature importance with Tenure included")
data_mod3 = pd.get_dummies(data,columns=['InternetService','Contract','PaymentMethod','MultipleLines','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])

data_mod3.replace(to_replace='No',value='0',inplace=True)

data_mod3.replace(to_replace='Yes',value='1',inplace=True)

data_mod3['gender'].replace(to_replace='Female',value='1',inplace=True)

data_mod3['gender'].replace(to_replace='Male',value='0',inplace=True)

data_mod3.head()

data_mod3.drop(columns='tenure',axis=1,inplace=True)
x4= data_mod3.drop(columns='TotalCharges',axis=1)

y4 = data_mod3['TotalCharges']
x_train4,x_test4,y_train4,y_test4 = train_test_split(x4,y4,test_size=0.3,random_state=121)

x_train4.shape

x_test4.shape

y_train4.shape

y_test4.shape
tree2 = DecisionTreeRegressor()

tree2.fit(x_train4,y_train4)

predict4 = tree2.predict(x_test4)

r2_score(y_test4,predict4)
x1 = pd.DataFrame(data_mod3.columns)

x1.columns=['Feature']

x1.reset_index(inplace=True)

x1.drop(columns='index',axis=1,inplace=True)

x1.drop(index=5,inplace=True)

tree2.feature_importances_.shape

x1['Importance']=tree2.feature_importances_

x1.sort_values(by='Importance',ascending=False,inplace=True)

px.bar(x=x1.Feature,y=x1.Importance,labels={'y':'Score'},title="Feature importance without Tenure")