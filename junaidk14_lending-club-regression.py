import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/lending-club-loan/loan.csv')
df.isnull().sum()
df1=df.drop(['id','member_id','title','zip_code','emp_title','issue_d',

             'earliest_cr_line','last_pymnt_d','last_credit_pull_d','url','desc','tax_liens',

             'delinq_amnt','acc_now_delinq'],axis=1)
df2 = df1[[column for column in df1 if ((df1[column].isnull().sum())/len(df1))*100 <= 60]]

df2.shape
df2.isnull().sum()/len(df1)*100
#Dropping rows with all NaN values

df2.dropna(how='all',inplace=True,axis=0)
df2.reset_index(drop=True,inplace=True)
df2.shape
df2.isnull().sum()
#Dropping columns with one unique value

df3 = df2[[column for column in df2 if df1[column].nunique()>1]]

print("List of dropped columns:", end=" ")

for c in df2.columns:

    if c not in df3.columns:

        print(c, end=", ")
df3.shape
df3.isnull().sum()
intr=[]

for i in df3['int_rate'].values:

    x=i[:-1]

    intr.append(x)
df3['int_rate']=np.array(intr)
pd.set_option('display.max_columns',150)

df3.head()
df3 = df3.dropna(subset=['revol_util'],axis = 0)
rev=[]

for i in df3['revol_util'].values:

    x=i[:-1]

    rev.append(x)
df3['revol_util']=np.array(rev)
# Variable Broadcasting

df3=df3.astype({'int_rate':float})
# Variable Broadcasting

df3=df3.astype({'revol_util':float})
df3.info()
ter=[]

for i in df3['term'].values:

    x=i.split()[0]

    ter.append(x)
df3['term']=np.array(ter, dtype = float)        
df3.replace({'10+ years':10,'< 1 year':1,'1 year':1,'3 years':3, '8 years':8, '9 years':9,

       '4 years':4, '5 years':5, '6 years':6, '2 years':2, '7 years':7},inplace=True)
df3=df3.astype({'emp_length':float})
df3.shape
df3.dropna(inplace=True,subset=['annual_inc'],axis=0)
df3.isnull().sum()
df3.reset_index(inplace=True)
df3.replace({'Does not meet the credit policy. Status:Fully Paid':'Fully Paid',

             'Does not meet the credit policy. Status:Charged Off':'Charged Off'},inplace=True)
df3['loan_status'].value_counts()
df3['emp_length']=df3['emp_length'].fillna(df3['emp_length'].median())
df3['pub_rec_bankruptcies']=df3['pub_rec_bankruptcies'].fillna(df3['pub_rec_bankruptcies'].median())
df3.shape
df3.drop('index',axis = 1, inplace=True)
corrmat  = df3.corr()

plt.subplots(figsize=(20,20))

sns.heatmap(corrmat, annot = True)
pd.set_option('display.max_columns',150)

df3.head()
df4 = pd.get_dummies(df3,drop_first=True)
pd.set_option('display.max_columns',1000)

df4.shape
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X = df4.drop('int_rate',axis = 1)

X.shape
y = df4['int_rate']

y.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 2)
sc1=StandardScaler()

X_train_sc=sc1.fit_transform(X_train)

X_test_sc=sc1.transform(X_test)
from sklearn.linear_model import Lasso,LinearRegression,Ridge

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.metrics import max_error,mean_absolute_error,mean_squared_error,r2_score
Las = Lasso()

LinR = LinearRegression()

Rid = Ridge()

Rfc = RandomForestRegressor(random_state=2)

Boost_Lin = AdaBoostRegressor(base_estimator=LinR,random_state=2)

Boost_las = AdaBoostRegressor(base_estimator=Las,random_state=2)

Boost_rid = AdaBoostRegressor(base_estimator=Rid,random_state=2)

svr = SVR()
for model, name in zip([Las,LinR,Rid,Rfc,Boost_Lin,Boost_las,Boost_rid,svr], 

     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Boosted Linear','Boosted Lasso','Boosted Ridge','SVR']):

    model1 = model.fit(X_train_sc,y_train)

    Y_predict=model1.predict(X_test_sc)

    print(name)

    print('Mean Absolute Error:', mean_absolute_error(y_test, Y_predict))  

    print('Mean Squared Error:', mean_squared_error(y_test, Y_predict))  

    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, Y_predict)))

    print('R2 : ',r2_score(y_test, Y_predict))

    print()

    

    
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[0,1])

X_sc = scaler.fit_transform(X)

pca = PCA().fit(X_sc)
#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Loan Dataset Explained Variance')

plt.show()
pca_var = pd.DataFrame((np.cumsum(pca.explained_variance_ratio_)))

pca_var = pca_var.T

pca_var.shape
pd.set_option('display.max_columns',140)

pca_var.head()
for model, name in zip([Las,LinR,Rid,Rfc,Boost_Lin,Boost_las,Boost_rid], 

     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Boosted Linear','Boosted Lasso','Boosted Ridge']):

    model1 = model.fit(X_train_sc,y_train)

    Y_predict=model1.predict(X_test_sc)

    print(name)

    plt.scatter(y_test, Y_predict)

    plt.title("Model Analysis")

    plt.xlabel("Truth")

    plt.ylabel("Prediction")

    plt.show()
n = np.arange(52,130)

for i in n:

    pca_comp = PCA(n_components=n)

    X_sc_pca = pca_comp.fit_transform(X_sc)

     
pca_95 = PCA(n_components=0.95)

X_sc_pca = pca_95.fit_transform(X_sc)

pca95_var = np.cumsum(pca_95.explained_variance_ratio_)

X_sc_pca.shape
X_train_pca,X_test_pca,y_train_pca,y_test_pca = train_test_split(X_sc_pca,y,test_size=0.3,random_state = 2)
for model, name in zip([Las,LinR,Rid,Rfc,Boost_Lin,Boost_las,Boost_rid,svr], 

     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Boosted Linear','Boosted Lasso','Boosted Ridge','SVR']):

    model1 = model.fit(X_train_pca,y_train_pca)

    Y_predict=model1.predict(X_test_pca)

    print(name)

    print('Mean Absolute Error:', mean_absolute_error(y_test_pca, Y_predict))  

    print('Mean Squared Error:', mean_squared_error(y_test_pca, Y_predict))  

    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_pca, Y_predict)))

    print('R2 : ',r2_score(y_test_pca, Y_predict))

    print()

    