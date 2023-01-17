
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from   statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import Lasso, Ridge

import warnings 
warnings.filterwarnings('ignore')


df=pd.read_csv('../input/auto-insurance-customerlifetimevalue/data.csv')
df.head()
df.shape
df.isnull().sum()
df.rename(columns={'Customer Lifetime Value':'CLV'},inplace=True)

df['CLV']
df.info()
df['Customer'].count
df['Number of Open Complaints'].value_counts()
df['Number of Policies'].value_counts()

df['Number of Open Complaints'] = df['Number of Open Complaints'].astype('object',copy=False)
df.info()
list(df.select_dtypes(exclude=['object']).columns)

sns.boxplot(df['Income'])
sns.boxplot(df['Monthly Premium Auto'])
sns.boxplot(df['Months Since Last Claim'])
sns.boxplot(df['Months Since Policy Inception'])
sns.boxplot(df['Total Claim Amount'])
df['Total Claim Amount'].describe()
fig,ax=plt.subplots(1,3,figsize=(15,5))
sns.boxplot(df['Total Claim Amount'],ax=ax[1])

sns.distplot(df['Total Claim Amount'],ax=ax[0])
sns.scatterplot(df['Total Claim Amount'],df['CLV'],ax=ax[2])
plt.axvline(1800,color='r')
sns.scatterplot(df['Total Claim Amount'],df['CLV'])
plt.axvline(1800,color='r')
df[df['Total Claim Amount']>1800]['Total Claim Amount'].count()
# df=df[df['Total Claim Amount']<1800]

print('Data point remain',len(df))
# q1=df['Total Claim Amount'].quantile(0.25)
# q2=df['Total Claim Amount'].quantile(0.50)
# q3=df['Total Claim Amount'].quantile(0.75)
# IQR=q3-q1


# Upper_whiskers=q3+1.5*IQR
# Upper_whiskers
# df[df['Total Claim Amount']>961]['Total Claim Amount'].count()
# a=df[~((df['Total Claim Amount']<(q1-1.5*IQR))|(df['Total Claim Amount']>(q3+1.5*IQR)))]

# a['Total Claim Amount'].describe()
# sns.scatterplot(a['Total Claim Amount'],a['CLV'])
sns.distplot(df['Total Claim Amount'])
sns.boxplot(df['Total Claim Amount'])
sns.distplot(df['CLV'])
fig,ax=plt.subplots(1,3,figsize=(15,5))
sns.boxplot(df['Monthly Premium Auto'],ax=ax[1])

sns.distplot(df['Monthly Premium Auto'],ax=ax[0])
sns.scatterplot(df['Monthly Premium Auto'],df['CLV'],ax=ax[2])
plt.axvline(204,color='r')
Upper_whiskers=df['Monthly Premium Auto'].quantile(0.75)+1.5*df['Monthly Premium Auto'].quantile(0.75)-df['Monthly Premium Auto'].quantile(0.25)
Upper_whiskers
df[df['Monthly Premium Auto']>204.5]['Monthly Premium Auto'].count()
# q1=df['Monthly Premium Auto'].quantile(0.25)
# q2=df['Monthly Premium Auto'].quantile(0.75)
# IQR=q2-q1
# df=df[~((df['Monthly Premium Auto']<(q1-1.5*IQR))|(df['Monthly Premium Auto']>(q2+1.5*IQR)))]

fig,ax=plt.subplots(1,3,figsize=(15,5))
sns.boxplot(df['Monthly Premium Auto'],ax=ax[1])

sns.distplot(df['Monthly Premium Auto'],ax=ax[0])
sns.scatterplot(df['Monthly Premium Auto'],df['CLV'],ax=ax[2])
plt.axvline(204,color='r')


df['Effective To Date']=pd.to_datetime(df['Effective To Date'],infer_datetime_format=True)
df.info()
df['Months']=df['Effective To Date'].dt.month
df['Months'].value_counts()
df['Months'] = df['Months'].astype('object',copy=False)
df['Number of Policies'] = np.where(df['Number of Policies']>2,3,df['Number of Policies'])
df['Number of Policies'].value_counts()
df['Number of Policies'] = df['Number of Policies'].astype('object',copy=False)

no=df[df['Response']=='No']['CLV']
yes=df[df['Response']=='Yes']['CLV']

import scipy.stats as stats
stats.mannwhitneyu(no,yes)
import matplotlib.pyplot as plt
# plt.figure(figsize=(15,3))
sns.boxplot(df['CLV'])
df['CLV'].describe()
q1=df['CLV'].quantile(0.25)
q2=df['CLV'].quantile(0.50)
q3=df['CLV'].quantile(0.75)
IQR=q3-q1
print(IQR)
l1=q3+1.5*IQR
l1
l2=q1-1.5*IQR
l2
df[df['CLV']>l1]['CLV'].count()
df[df['CLV']>l1].head()
q1=df['CLV'].quantile(0.25)
q2=df['CLV'].quantile(0.75)
IQR=q2-q1
df=df[~((df['CLV']<(q1-1.5*IQR))|(df['CLV']>(q2+1.5*IQR)))]

df.shape
sns.boxplot(df['CLV'])
sns.boxplot(x=df['Coverage'],y=df['CLV'])
# sns.scatterplot(df['Customer Lifetime Value'])
sns.scatterplot(y='CLV',x='Income',data=df)

sns.scatterplot(y='CLV',x='Monthly Premium Auto',data=df)
sns.scatterplot(y='CLV',x='Total Claim Amount',data=df)
df.corr()
# df=df[df['Customer Lifetime Value']<df['Customer Lifetime Value'].max()]
df.head()
df['CLV'].max()
sns.distplot(df['CLV'])
df['State'].value_counts()
cl=df[df['State']=='California']['CLV']
org=df[df['State']=='Oregon']['CLV']
ar=df[df['State']=='Arizona']['CLV']
Nv=df[df['State']=='Nevada']['CLV']
Wa=df[df['State']=='Washington']['CLV']

a=stats.kruskal(cl,org,ar,Nv,Wa)

catg=df.select_dtypes(include=['object'])
catg.drop(['Customer'],1,inplace=True)
catg.head()
col=list(catg.columns)
print(len(col))
col
for i in col:
    print(i,'\n',df[i].value_counts(),'\n')
ttest=[]
anova=[]
for i in col:
    if len(df[i].unique())<3:
        ttest.append(i)
    else:
        anova.append(i)

print(ttest)
print(anova)
df1=df.iloc[1:1670,:]
df1.shape
print(stats.shapiro(df1['CLV']))
# print(stats.jarque_bera(df['Months Since Last Claim']))
num=df.select_dtypes(exclude=['object'])
num.columns
yes=df[df['Response']=='No']['CLV']
no=df[df['Response']=='Yes']['CLV']


b=stats.mannwhitneyu(yes,no)
b[1]
female=df[df['Gender']=='F']['CLV']
male=df[df['Gender']=='M']['CLV']
f=stats.mannwhitneyu(female,male)
anova
df['State'].value_counts()
ca=df[df['State']=='California']['CLV']
Or=df[df['State']=='Oregon']['CLV']
Ar=df[df['State']=='Arizona']['CLV']
Ne=df[df['State']=='Nevada']['CLV']
Wa=df[df['State']=='Washington']['CLV']
stats.kruskal(ca,Or,Ar,Ne,Wa)

df['Coverage'].value_counts()
Ba=df[df['Coverage']=='Basic']['CLV']
Ex=df[df['Coverage']=='Extended']['CLV']
Pr=df[df['Coverage']=='Premium']['CLV']
c=stats.kruskal(Ba,Ex,Pr)
df['Education'].value_counts()
Ba=df[df['Education']=='Bachelor']['CLV']
Co=df[df['Education']=='College']['CLV']
Hi=df[df['Education']=='High School or Below']['CLV']
Ma=df[df['Education']=='Master']['CLV']
Da=df[df['Education']=='Doctor']['CLV']
d=stats.kruskal(Ba,Co,Hi,Ma,Da)
sns.boxplot(y='CLV',x='Education',data=df)

    

df['EmploymentStatus'].value_counts()
Ba=df[df['EmploymentStatus']=='Employed']['CLV']
Co=df[df['EmploymentStatus']=='Unemployed']['CLV']
Hi=df[df['EmploymentStatus']=='Medical Leave']['CLV']
Ma=df[df['EmploymentStatus']=='Disabled']['CLV']
Da=df[df['EmploymentStatus']=='Retired']['CLV']
e=stats.kruskal(Ba,Co,Hi,Ma,Da)
df['Location Code'].value_counts()
Ba=df[df['Location Code']=='Suburban']['CLV']
Co=df[df['Location Code']=='Rural']['CLV']
Hi=df[df['Location Code']=='Urban']['CLV']
g=stats.kruskal(Ba,Co,Hi)
df['Marital Status'].value_counts()
Ba=df[df['Marital Status']=='Married']['CLV']
Co=df[df['Marital Status']=='Single']['CLV']
Hi=df[df['Marital Status']=='Divorced']['CLV']
h=stats.kruskal(Ba,Co,Hi)
df['Number of Open Complaints'].value_counts()
Ba=df[df['Number of Open Complaints']==0]['CLV']
Co=df[df['Number of Open Complaints']==1]['CLV']
Hi=df[df['Number of Open Complaints']==2]['CLV']
Ma=df[df['Number of Open Complaints']==3]['CLV']
Da=df[df['Number of Open Complaints']==4]['CLV']
As=df[df['Number of Open Complaints']==5]['CLV']
i=stats.kruskal(Ba,Co,Hi,Ma,Da,As)
e
df['Number of Policies'].value_counts()
Ba=df[df['Number of Policies']==1]['CLV']
Co=df[df['Number of Policies']==2]['CLV']
Hi=df[df['Number of Policies']==3]['CLV']

j=stats.kruskal(Ba,Co,Hi)
j
df['Policy Type'].value_counts()
Ba=df[df['Policy Type']=='Personal Auto']['CLV']
Co=df[df['Policy Type']=='Corporate Auto']['CLV']
Hi=df[df['Policy Type']=='Special Auto']['CLV']
k=stats.kruskal(Ba,Co,Hi)
df['Policy'].value_counts()
Ba=df[df['Policy']=='Personal L3']['CLV']
Co=df[df['Policy']=='Personal L2']['CLV']
Hi=df[df['Policy']=='Personal L1']['CLV']
Ma=df[df['Policy']=='Corporate L3']['CLV']
Da=df[df['Policy']=='Corporate L2']['CLV']
Ca=df[df['Policy']=='Corporate L1']['CLV']
s1=df[df['Policy']=='Special L2']['CLV']
s2=df[df['Policy']=='Special L3']['CLV']
s3=df[df['Policy']=='Special L1']['CLV']
l=stats.kruskal(Ba,Co,Hi,Ma,Da,Ca,s1,s2,s3)
df['Renew Offer Type'].value_counts()
Ba=df[df['Renew Offer Type']=='Offer1']['CLV']
Co=df[df['Renew Offer Type']=='Offer2']['CLV']
Hi=df[df['Renew Offer Type']=='Offer3']['CLV']
Ma=df[df['Renew Offer Type']=='Offer4']['CLV']
m=stats.kruskal(Ba,Co,Hi,Ma)
df['Sales Channel'].value_counts()
Ba=df[df['Sales Channel']=='Agent']['CLV']
Co=df[df['Sales Channel']=='Branch']['CLV']
Hi=df[df['Sales Channel']=='Call Center']['CLV']
Ma=df[df['Sales Channel']=='Web']['CLV']
n=stats.kruskal(Ba,Co,Hi,Ma)
anova
df['Vehicle Class'].value_counts()
Ba=df[df['Vehicle Class']=='Four-Door Car']['CLV']
Co=df[df['Vehicle Class']=='Two-Door Car']['CLV']
Hi=df[df['Vehicle Class']=='SUV']['CLV']
Ma=df[df['Vehicle Class']=='Sports Car']['CLV']
Da=df[df['Vehicle Class']=='Luxury SUV']['CLV']
Ca=df[df['Vehicle Class']=='Luxury Car']['CLV']
o=stats.kruskal(Ba,Co,Hi,Ma,Da,Ca)
o
df['Vehicle Size'].value_counts()
Ba=df[df['Vehicle Size']=='Medsize']['CLV']
Co=df[df['Vehicle Size']=='Small']['CLV']
Hi=df[df['Vehicle Size']=='Large']['CLV']
p=stats.kruskal(Ba,Co,Hi)
p[1]
df['Months'].value_counts()
Ba=df[df['Months']==1]['CLV']
Co=df[df['Months']==2]['CLV']
q=stats.kruskal(Ba,Co)
q
z=[a[1],b[1],c[1],d[1],e[1],f[1],g[1],h[1],i[1],j[1],k[1],l[1],m[1],n[1],o[1],p[1],q[1]]
print(len(z))
z
print(len(col))
col




df2=pd.DataFrame({'columns':col,'P_vlaue':z})
df2
df2['status']=df2['P_vlaue'].map(lambda x:'significant' if x<0.05 else 'not significant')
df2
sns.barplot(df['State'],df['CLV'])

sns.barplot(df['Response'],df['CLV'])


sns.barplot(df['Gender'],df['CLV'])
sns.barplot(df['Location Code'],df['CLV'])
sns.barplot(df['Policy Type'],df['CLV'])
sns.barplot(df['Policy'],df['CLV'])
sns.barplot(df['Renew Offer Type'],df['CLV'])
sns.barplot(df['Months'],df['CLV'])

num=df.select_dtypes(exclude=['object'])
num=num.drop('Effective To Date',1)
num.head()
X=num.drop(['CLV'],axis=1)
y=num['CLV']
# xc=sm.add_constant(x)
lin_reg=sm.OLS(y,X).fit()
lin_reg.summary()                                                                     


a={'Numerical_column':['Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Total Claim Amount'],
  'P_value':[0.011,0.000,0.102,0.483,0.057],'Status':['Significant','Significant','Not_Significant','Not_Significant','Significant']}
m=pd.DataFrame(a)
m


num.head()
catg=df.select_dtypes(include=['object'])
catg=catg.drop('Customer',1)
catg.head()
df1=pd.concat([catg,num],axis=1)
df.head()
catg=pd.get_dummies(catg,drop_first=True)
catg.head()


df=pd.concat([catg,num],axis=1)
df.head()
X=df.drop(['CLV'],axis=1)
y=num['CLV']
xc=sm.add_constant(X)
lin_reg=sm.OLS(y,xc).fit()
lin_reg.summary()
import statsmodels.api as sm
sm.stats.diagnostic.linear_rainbow(res=lin_reg, frac=0.5)
y_pred=lin_reg.predict()

fig,ax=plt.subplots(figsize=(7,5))
sns.regplot(x=y_pred,y=y,lowess=True,line_kws={'color':'red'})
ax.set_title('observed vs predicted')
ax.set(xlabel='Predicted',ylabel='Observed')
plt.show()
residuals = y_pred-y
mean_of_residuals = np.mean(residuals)
print(f"The mean of the residuals is {mean_of_residuals}")
y_pred=lin_reg.predict()
resids=lin_reg.resid
fig,ax=plt.subplots()
sns.regplot(y_pred,resids,lowess=True,line_kws={'color':'red'})  ## lowess kind of linear rerrg.
ax.set_title('residual vs predicted')
ax.set(xlabel='prediacted',ylabel='residual')
plt.show()
import statsmodels.stats.api as sms
name=['F-statistic','p-value']
test=sms.het_goldfeldquandt(lin_reg.resid,lin_reg.model.exog)

test

plt.figure(figsize=(7,5))
p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')
from scipy import stats
print(stats.jarque_bera(lin_reg.resid))
from statsmodels.stats.stattools import durbin_watson
durbin_watson(lin_reg.resid)
import statsmodels.tsa.api as smt
plt.figure(figsize=(7,5))
acf=smt.graphics.plot_acf(lin_reg.resid,lags=40)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
a=pd.DataFrame({'vif': vif}, index=X.columns)
a
df.head()

df.shape
X=df.drop(['CLV','Policy Type_Personal Auto','Policy Type_Special Auto','Policy_Personal L1','Policy_Personal L2',
'Policy_Personal L3',
'Policy_Special L1',
'Policy_Special L2',
'Policy_Special L3',],axis=1)
y=df['CLV']
# xc=sm.add_constant(x)
lin_reg=sm.OLS(y,X).fit()
lin_reg.summary()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
sns.distplot(df['CLV'])
a=np.log(df['CLV'])
sns.distplot(a)
b = df['CLV'].transform(lambda X: 1/X)
sns.distplot(b)
b

b = df['CLV'].transform(lambda X:X**0.10)
sns.distplot(b)
sns.boxplot(df['Monthly Premium Auto'])
sns.distplot(df['Monthly Premium Auto'])
a=np.log(df['Monthly Premium Auto'])
sns.distplot(a)
b = df['Monthly Premium Auto'].transform(lambda X:X**0.1)
sns.distplot(b)

X=df.drop(['CLV','Policy Type_Personal Auto','Policy Type_Special Auto','Policy_Personal L1','Policy_Personal L2','Policy_Personal L3',
'Policy_Special L1','Policy_Special L2','Policy_Special L3'],1)
y=df['CLV']          
y =np.log(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

print('-----Log---- ')
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred)) 
print('After Anti Log')
print('RMSE : ',np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred))))
print('r2_score : ',r2_score(np.exp(y_test),np.exp(y_pred)))

print('After Anti Log')
print('RMSE : ',np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred))))
print('r2_score : ',r2_score(np.exp(y_test),np.exp(y_pred)))
y_pred_train=lr.predict(x_train)
print('RMES:',np.sqrt(mean_squared_error(y_train,y_pred_train)))
print('R-squared:',r2_score(y_train,y_pred_train)) 
X.head()
from sklearn.feature_selection import RFE

model=LinearRegression()
X=df.drop(['CLV','Policy Type_Personal Auto','Policy Type_Special Auto','Policy_Personal L1','Policy_Personal L2','Policy_Personal L3',
'Policy_Special L1','Policy_Special L2','Policy_Special L3'],1)
y=df['CLV']

### initilize rfe
rfe=RFE(model,10)
rfe.fit(x_train,y_train)


print(rfe.support_)
print(rfe.ranking_)
#no of features
nof_list=np.arange(1,48)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


X=df.drop(['CLV','Policy Type_Personal Auto','Policy Type_Special Auto','Policy_Personal L1','Policy_Personal L2','Policy_Personal L3',
'Policy_Special L1','Policy_Special L2','Policy_Special L3'],1)
y=df['CLV']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
nof_cols=np.arange(1,49)

model=LinearRegression()
score_list=[]

for n in range(48):
    rfe=RFE(model,n+1)
    rfe.fit(x_train,y_train)
    y_pred=rfe.predict(x_test)
    score=r2_score(y_test,y_pred)
    score_list.append(score)
plt.plot(nof_cols,score_list)
plt.show()
# plt.plot(nof_cols,score_list)
rfe=RFE(model,43)
rfe.fit(x_train,y_train)
pd.DataFrame(list(zip(X.columns,rfe.support_,rfe.ranking_)),columns=['col','select','rank'])
y_pred=rfe.predict(x_test)
print('R2:',r2_score(y_test,y_pred))
print('Rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))



df1.head()
catg=df1.select_dtypes(include=['object'])
catg.head()
num1=df1.select_dtypes(exclude=['object'])
num1.head()
from sklearn.preprocessing import LabelEncoder

cat=catg.apply(LabelEncoder().fit_transform)
cat.head()
df1=pd.concat([cat,num1],axis=1)
df1.head()
X=df1.drop(['CLV','State','Response','Gender','Location Code','Policy Type','Months','Policy','Renew Offer Type','Months Since Policy Inception'],1)
y=df1['CLV']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)


y_pred=dt.predict(x_test)
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred)) 


from scipy.stats import zscore

num2=num1.apply(zscore)
num2.head()
sns.distplot(num2['Monthly Premium Auto'])
df2=pd.concat([cat,num2],axis=1)
df2.head()
X=df2.drop(['CLV','State','Response','Gender','Location Code','Policy Type','Months'],1)
y=df2['CLV']
# y=np.log(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred)) 




rf = RandomForestRegressor()

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred)) 
from sklearn.model_selection import GridSearchCV
rf = RandomForestRegressor()
params = {
        'max_depth' : [10,20,30],
        
        'n_estimators' : [100,200,50],
        
        "bootstrap" : [True, False],
    
        'max_features': ['auto', 'sqrt', 'log2']
        
        }

grid = GridSearchCV(estimator = rf, param_grid=params, cv = 5, n_jobs = -1, return_train_score = True )
grid.fit(x_train,y_train)
grid.best_params_
rf = RandomForestRegressor(bootstrap= True,max_depth= 30,max_features= 'auto',n_estimators= 300)

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred)) 

from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor 
br = BaggingRegressor()

br.fit(x_train,y_train)

y_pred=br.predict(x_test)
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred))

br = GradientBoostingRegressor(learning_rate=0.1,n_estimators=200)

br.fit(x_train,y_train)

y_pred=br.predict(x_test)
print('RMES:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred))

X=df1.drop(['CLV','State','Response','Gender','Location Code','Policy'],axis=1)
y=df1['CLV']
# xc=sm.add_constant(x)
lin_reg=sm.OLS(y,X).fit()
lin_reg.summary()


num1.head()

num3=num1[['Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Total Claim Amount','CLV']]
num3.head()
cat.head()
df4=pd.concat([cat,num3],axis=1)
df4.head()
n_iterations = 10
n_size = int(len(df4) * 0.50)
values = df4.values
from sklearn.utils import resample

# run bootstrap
stats = list()
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(values, n_samples=n_size)
    test = np.array([x for x in values if x.tolist() not in train.tolist()])
    # fit model
    model =RandomForestRegressor()
    model.fit(train[:,:-1], train[:,-1])
    # evaluate model
    predictions = model.predict(test[:,:-1])
    score = r2_score(test[:,-1], predictions)
    print(score)
    stats.append(score)
plt.hist(stats)
plt.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


