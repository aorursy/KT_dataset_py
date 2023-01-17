# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs.layout as go
from plotly import tools
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
# Any results you write to the current directory are saved as output.
application_train=pd.read_csv('../input/application_train.csv')
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
application_test = pd.read_csv('../input/application_test.csv')
print('Shape of our application_train data: ',application_train.shape)
print('How does our application_train data look: ')
application_train.head()
print('Shape of our pos_cash_balance data: ',pos_cash_balance.shape)
print('How does our pos_cash_balance data look: ')
pos_cash_balance.head()
print('Shape of our bureau_balance data: ',bureau_balance.shape)
print('How does our bureau_balance data look: ')
bureau_balance.head()
print('Shape of our previous_application data: ',previous_application.shape)
print('How does our previous_application data look: ')
previous_application.head()
print('Shape of our installments_payments data: ',installments_payments.shape)
print('How does our installments_payments data look: ')
installments_payments.head()
print('Shape of our credit_card_balance data: ',credit_card_balance.shape)
print('How does our credit_card_balance data look: ')
credit_card_balance.head()
print('Shape of our bureau data: ',bureau.shape)
print('How does our bureau data look: ')
bureau.head()
#application_train missing data
appmistotal=application_train.isnull().sum().sort_values(ascending=False)
appmisperc=(application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending=False)
appmissdatatrain=pd.concat([appmistotal,appmisperc],axis=1,keys=['Total','Percentage'])
appmissdatatrain.head()
#bureau missing data
appmistotal=bureau.isnull().sum().sort_values(ascending=False)
appmisperc=(bureau.isnull().sum()/bureau.isnull().count()*100).sort_values(ascending=False)
appmissdatatrain=pd.concat([appmistotal,appmisperc],axis=1,keys=['Total','Percentage'])
appmissdatatrain.head(20)
#pos_cash_balance missing data
posmistotal=pos_cash_balance.isnull().sum().sort_values(ascending=False)
posmisperc=(pos_cash_balance.isnull().sum()/pos_cash_balance.isnull().count()*100).sort_values(ascending=False)
posmissdata=pd.concat([posmistotal,posmisperc],axis=1,keys=['Total','Percentage'])
posmissdata.head()
#bureau_balance missing data
bbmistotal=bureau_balance.isnull().sum().sort_values(ascending=False)
bbmisperc=(bureau_balance.isnull().sum()/bureau_balance.isnull().count()*100).sort_values(ascending=False)
bbmissdata=pd.concat([bbmistotal,bbmisperc],axis=1,keys=['Total','Percentage'])
bbmissdata.head(20)
#previous_application missing data
pbmistotal=previous_application.isnull().sum().sort_values(ascending=False)
pbmisperc=(previous_application.isnull().sum()/previous_application.isnull().count()*100).sort_values(ascending=False)
pbmissdata=pd.concat([pbmistotal,pbmisperc],axis=1,keys=['Total','Percentage'])
pbmissdata.head(20)
#installments_payments missing data
ipmistotal=installments_payments.isnull().sum().sort_values(ascending=False)
ipmisperc=(installments_payments.isnull().sum()/installments_payments.isnull().count()*100).sort_values(ascending=False)
ipmissdata=pd.concat([ipmistotal,ipmisperc],axis=1,keys=['Total','Percentage'])
ipmissdata.head()
#credit_card_balance missing data
ccmistotal=credit_card_balance.isnull().sum().sort_values(ascending=False)
ccmisperc=(credit_card_balance.isnull().sum()/credit_card_balance.isnull().count()*100).sort_values(ascending=False)
ccmissdata=pd.concat([ccmistotal,ccmisperc],axis=1,keys=['Total','Percentage'])
ccmissdata.head()
#Sex ratio in Appication Train Data
sex={'M':'Male','F':'Female','XNA':'Not Defined'}
temp=application_train['CODE_GENDER'].map(sex).value_counts()
trace=go.Bar(x=temp.index,y=temp.values*100/temp.sum(),marker=dict(color=(temp / temp.sum())*100,
        colorscale = 'Blues',reversescale = True))
data=[trace]
layout=go.Layout(title='Sex ratio of Loan Applicant',xaxis=dict(title='People Count'),
                 yaxis=dict(title='Applicant\'s Sex'))
fig=go.Figure(data=data,layout=layout)
iplot(fig)
#alternative to above code
sex={'M':'Male','F':'Female','XNA':'Not Defined'}
temp=application_train['CODE_GENDER'].map(sex).value_counts()
plt.bar(temp.index,temp.values*100/temp.sum())
plt.title('Sex ratio of Loan Applicant')
plt.ylabel('Count')
#is data balanced?
repay={0:'Repayed',1:'Not Repayed'}
temp=application_train.TARGET.map(repay).value_counts()
tempo=pd.DataFrame({'labels':temp.index,'values':temp.values})
#plt.figure()
#plt.pie(tempo.values,labels=tempo.labels,autopct='%1.1f%%', shadow=True)
#plt.title('Data Balance Check')
tempo.iplot(kind='pie',labels='labels',values='values',title='Loan Repayed or not')
#person accompanied client during loan application
temp=application_train.NAME_TYPE_SUITE.value_counts()
tempo=pd.DataFrame({'labels':temp.index,'values':temp.values})
tempo.iplot(kind='pie',labels='labels',values='values',title='Person accompanied client during loan application')
#what kind of loans have been distributed
temp=application_train.NAME_CONTRACT_TYPE.value_counts()
tempo=pd.DataFrame({'values':(temp / temp.sum())*100}).reset_index()
plt.bar(tempo.index,tempo.values)
plt.title('Sex ratio of Loan Applicant')
plt.ylabel('Count')
#tempo.iplot(kind='bar',labels='index',values='values',  title='Loan Disbursal Percentage')

#amount credit distribution
cf.set_config_file(theme='pearl')
application_train['AMT_CREDIT'].iplot(kind='histogram')
#amount_income total distribution
cf.set_config_file(theme='pearl')
application_train['AMT_INCOME_TOTAL'].iplot(kind='scatter')
#amt_goods_price distribution
plt.figure(figsize=(10,6))
plt.title('AMOUNT_GOODS_PRICE\'s Distribution')
sns.distplot(application_train['AMT_GOODS_PRICE'].dropna(),color='r')
#what kind of loans were distributed
temp=application_train.NAME_CONTRACT_TYPE.value_counts()
plt.figure(figsize=(6,6))
plt.pie(temp.values,labels=temp.index,autopct='%1.1f%%', shadow=True)
plt.title('Loan Types')
#Assests owned by Loan applicants
temp=application_train.FLAG_OWN_CAR.value_counts()
temp1=application_train.FLAG_OWN_REALTY.value_counts()
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].pie(temp.values,labels=temp.index,autopct='%1.1f%%', shadow=True)
ax[0].set_title('Applicants owning Cars')
ax[1].pie(temp1.values,labels=temp1.index,autopct='%1.1f%%', shadow=True)
ax[1].set_title('Applicants owning Real Estate')

#Loan Appplicants Income Type
temp=application_train['NAME_INCOME_TYPE'].value_counts()
temp1=pd.DataFrame({'labels':temp.index,'values':temp.values})
temp1.iplot(kind='pie',labels='labels',values='values',title="Different Sources of Applicant's Income", hole=0.6)
#Loan Appplicants Income Type
temp=application_train['NAME_FAMILY_STATUS'].value_counts()
plt.figure(figsize=(10,7))
plt.pie(temp.values,labels=temp.index,autopct='%1.1f%%', shadow=True)
plt.title('Loan Applicants Family Status')
plt.legend(loc='best')
#temp1.iplot(kind='pie',labels='labels',values='values',title="Different Sources of Applicant's Income", hole=0.6)
#chinldren numbers in application 
ch=application_train['CNT_CHILDREN'].value_counts()
ch.iplot(kind='bar',xTitle='Number of Children',yTitle='Children\'s Count', title='Loan applicants children distribution')
#family members numbers in application 
ch=application_train['CNT_FAM_MEMBERS'].value_counts()
ch.iplot(kind='bar',xTitle='Number of family members',yTitle='Applicant\'s Count wrt members', title='Loan applicants family members distribution')
df=application_train['WALLSMATERIAL_MODE'].value_counts()
df.iplot(kind='bar',xTitle='Wall material type', yTitle='Count', title='Walls Material')
df=application_train['REGION_RATING_CLIENT'].value_counts()
df1=pd.DataFrame({'labels':df.index,'values':df.values})
df1.iplot(kind='pie',labels='labels',values='values',title='Client\'s Region Rating ')
#DAYS_BIRTH distribution
plt.figure(figsize=(10,5))
plt.hist(abs(application_train.DAYS_BIRTH/365), edgecolor='b',bins=30)
plt.xlabel('Client\'s Age in Years')
plt.ylabel('Count')
plt.title('Age of client DISTRIBUTION')
age=application_train[['TARGET','DAYS_BIRTH']]
age['DAYS_BIRTH']=abs(age['DAYS_BIRTH'])
age['YEARS_BIRTH']=age['DAYS_BIRTH']/365
age['YEARS_INTERVAL']=pd.cut(age['YEARS_BIRTH'],bins=np.linspace(20,70,num=11))
agegrp=age.groupby('YEARS_INTERVAL').mean()
plt.figure(figsize=(8,8))
plt.bar(agegrp.index.astype(str),100*agegrp['TARGET'])
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
                
#DAYS_EMPLOYED distribution
plt.figure(figsize=(10,5))
sns.distplot(application_train.DAYS_EMPLOYED)
plt.title('DAYS_EMPLOYED DISTRIBUTION')
#No of days for registration distribution
plt.figure(figsize=(10,5))
sns.distplot(application_train.DAYS_REGISTRATION)
plt.title('DAYS REGISTRATION DISTRIBUTION')
#loan applicants occupation & owning Car and Real estate
df=application_train[['FLAG_OWN_CAR','FLAG_OWN_REALTY','OCCUPATION_TYPE']]
df1=df[df['FLAG_OWN_CAR']=='Y']
df2=df1['OCCUPATION_TYPE'].value_counts().head(10)
df3=df[df['FLAG_OWN_REALTY']=='Y']
df4=df3['OCCUPATION_TYPE'].value_counts().head(10)
fig,ax=plt.subplots(2,1,figsize=(15,10))
plt.figure(figsize=(15,5))
ax[0].bar(df2.index,df2.values,width=0.5)
ax[0].set_title('Top ten occupation of Loan applicants owning CAR')
ax[0].set_xlabel('Occupation')
ax[0].set_ylabel('Count')
ax[1].bar(df4.index,df4.values,width=0.5)
ax[1].set_title('Top ten occupation of Loan applicants owning Realty')
ax[1].set_xlabel('Occupation')
ax[1].set_ylabel('Count')
#Applicant's Income sources for Loan Repayed or not
df=application_train[['TARGET','NAME_INCOME_TYPE']]
#applicants income sources who repaid
df1=df[df['TARGET']==0]['NAME_INCOME_TYPE'].value_counts()
#applicants income sources who could not repay
df2=df[df['TARGET']==1]['NAME_INCOME_TYPE'].value_counts()
trace1=go.Bar(x = df1.index,y = (df1.values / df1.sum()) * 100,name='YES')
trace2=go.Bar( x = df2.index, y = (df2.values / df2.sum()) * 100, name='NO')
data=[trace1,trace2]
layout=go.Layout(title='Applicant\'s Income sources for Loan Repayed or not',
                width=1000,xaxis=dict(title='Income Sources'),yaxis=dict(title='Count Percentage'))
fig=go.Figure(data=data,layout=layout)            
iplot(fig)
#Organizations who applied for Loan
df=application_train['ORGANIZATION_TYPE'].value_counts()
df.iplot(kind='bar', xTitle='Organization Type', yTitle='Count', title='Organization Types applied for loan', color='green')

#Applicant's family status showing Loan repayed or not in Percentage
df=application_train['NAME_FAMILY_STATUS'].value_counts()
df0=[]
df1=[]
for val in df.index:
    df1.append(np.sum(application_train['TARGET'][application_train['NAME_FAMILY_STATUS']==val]==1))
    df0.append(np.sum(application_train['TARGET'][application_train['NAME_FAMILY_STATUS']==val]==0))
trace1=go.Bar(x=df.index,y=(df1/df.sum())*100,name="YES")
trace2=go.Bar(x=df.index,y=(df0/df.sum())*100,name="NO")
layout=go.Layout(title='Applicant\'s family status showing Loan repayed or not in Percentage',
                      xaxis=dict(title='Family Status'), yaxis=dict(title='Count in Percentage'))
fig=go.Figure(data=[trace1, trace2],layout=layout)
iplot(fig)

#Applicant's occupation showing Loan repayed or not in Percentage
df=application_train['OCCUPATION_TYPE'].value_counts()
df0=[]
df1=[]
for val in df.index:
    df1.append(np.sum(application_train['TARGET'][application_train['OCCUPATION_TYPE']==val]==1))
    df0.append(np.sum(application_train['TARGET'][application_train['OCCUPATION_TYPE']==val]==0))
trace1=go.Bar(x=df.index,y=(df1/df.sum())*100,name="YES")
trace2=go.Bar(x=df.index,y=(df0/df.sum())*100,name="NO")
layout=go.Layout(title='Applicant\'s Occupation showing Loan repayed or not in Percentage',
                      xaxis=dict(title='Occupation'), yaxis=dict(title='Count in Percentage'))
fig=go.Figure(data=[trace1, trace2],layout=layout)
iplot(fig)
#Applicant's income source showing Loan repayed or not in Percentage
df=application_train['NAME_INCOME_TYPE'].value_counts()
df0=[]
df1=[]
for val in df.index:
    df1.append(np.sum(application_train['TARGET'][application_train['NAME_INCOME_TYPE']==val]==1))
    df0.append(np.sum(application_train['TARGET'][application_train['NAME_INCOME_TYPE']==val]==0))
trace1=go.Bar(x=df.index,y=(df1/df.sum())*100,name="YES")
trace2=go.Bar(x=df.index,y=(df0/df.sum())*100,name="NO")
layout=go.Layout(title='Applicant\'s sources of Income showing Loan repayed or not in Percentage',
                      xaxis=dict(title='Family Status'), yaxis=dict(title='Count in Percentage'))
fig=go.Figure(data=[trace1, trace2],layout=layout)
iplot(fig)
#Applicant's organization type showing Loan repayed or not in Percentage
df=application_train['ORGANIZATION_TYPE'].value_counts()
df0=[]
df1=[]
for val in df.index:
    df1.append(np.sum(application_train['TARGET'][application_train['ORGANIZATION_TYPE']==val]==1))
    df0.append(np.sum(application_train['TARGET'][application_train['ORGANIZATION_TYPE']==val]==0))
trace1=go.Bar(x=df.index,y=(df1/df.sum())*100,name="YES")
trace2=go.Bar(x=df.index,y=(df0/df.sum())*100,name="NO")
layout=go.Layout(title='Applicant\'s workplace showing Loan repayed or not in Percentage',
                      xaxis=dict(title='Organization Status'), yaxis=dict(title='Count in Percentage'))
fig=go.Figure(data=[trace1, trace2],layout=layout)
iplot(fig)
#Applicant's purpose for buying houses type showing Loan repayed or not in Percentage
df=application_train['NAME_HOUSING_TYPE'].value_counts()
df0=[]
df1=[]
for val in df.index:
    df1.append(np.sum(application_train['TARGET'][application_train['NAME_HOUSING_TYPE']==val]==1))
    df0.append(np.sum(application_train['TARGET'][application_train['NAME_HOUSING_TYPE']==val]==0))
trace1=go.Bar(x=df.index,y=(df1/df.sum())*100,name="YES")
trace2=go.Bar(x=df.index,y=(df0/df.sum())*100,name="NO")
layout=go.Layout(title='Applicant\'s house status after which Loan repayed or not in Percentage',
                      xaxis=dict(title='Housing Type'), yaxis=dict(title='Count in Percentage'))
fig=go.Figure(data=[trace1, trace2],layout=layout)
iplot(fig)
#Applicant's educational qualification type showing Loan repayed or not in Percentage
df=application_train['NAME_EDUCATION_TYPE'].value_counts()
df0=[]
df1=[]
for val in df.index:
    df1.append(np.sum(application_train['TARGET'][application_train['NAME_EDUCATION_TYPE']==val]==1))
    df0.append(np.sum(application_train['TARGET'][application_train['NAME_EDUCATION_TYPE']==val]==0))
trace1=go.Bar(x=df.index,y=(df1/df.sum())*100,name="YES")
trace2=go.Bar(x=df.index,y=(df0/df.sum())*100,name="NO")
layout=go.Layout(title='Applicant\'s educational status showing Loan repayed or not in Percentage',
                      xaxis=dict(title='Educational Qualification'), yaxis=dict(title='Count in Percentage'))
fig=go.Figure(data=[trace1, trace2],layout=layout)
iplot(fig)
#Loan Appplicants Previous Contract Product Type
temp=previous_application['NAME_CONTRACT_TYPE'].value_counts()
temp1=pd.DataFrame({'labels':temp.index,'values':temp.values})
temp1.iplot(kind='pie',labels='labels',values='values',title="Loan Appplicants Previous Contract Product Type", hole=0.6)
#Day on which Previous Loan Appplicants
df=previous_application['WEEKDAY_APPR_PROCESS_START'].value_counts()
plt.figure(figsize=(10,5))
plt.bar(df.index,(df/df.sum())*100)
plt.xlabel('Weekdays')
plt.ylabel('Count')
plt.title("Days on which Previous Loans applications started")
#purpose of previous cash loans
df=previous_application['NAME_CASH_LOAN_PURPOSE'].value_counts()
trace=go.Bar(x=df.index, y=(df/df.sum())*100)
layout=go.Layout(title='Cash loans purpose in previous application', xaxis=dict(title='Cash Loan Purpose'), 
                 yaxis=dict(title='Percentage'))
fig=go.Figure(data=[trace], layout=layout)
iplot(fig)
#whether previous loan was approved or not
sp1=previous_application['NAME_CONTRACT_STATUS'].value_counts()
trace=go.Bar(x=sp1.index,y=(sp1/sp1.sum())*100,text=(sp1/sp1.sum())*100,
            textposition = 'auto',
            marker=dict( color='rgb(158,202,225)', line=dict( color='rgb(8,48,107)',  width=1.5) ), opacity=0.6)
layout=dict(title='Approval Status of Previous Loan Application', xaxis=dict(title='Approval Status'),
           yaxis=dict(title='Percentage'))
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='ContractStatus')
#Payment methods chosen by clients to payback previous loans
sp=previous_application['NAME_PAYMENT_TYPE'].value_counts()
sp1=pd.DataFrame({'labels':sp.index,'values':sp.values})
sp1.iplot(kind='pie',labels='labels',values='values',title='Payment methods chosen by clients to payback previous loans')
#Loan Application Rejection by HomeCredit in previous loans
sp=previous_application['CODE_REJECT_REASON'].value_counts()
sp.iplot(kind='bar',xTitle='Reason',yTitle='Rejection Count',title='Loan Application Rejection by HomeCredit in previous loans')
#Client accompanied by, in previous applications
sp=previous_application['NAME_TYPE_SUITE'].value_counts()
sp1=pd.DataFrame({'labels':sp.index,'values':sp.values})
sp1.iplot(kind='pie',labels='labels',values='values',title='Who accompanied client during previous loan applications')
#Determing whether client is new or old
sp=previous_application['NAME_CLIENT_TYPE'].value_counts()
sp1=pd.DataFrame({'labels':sp.index,'values':sp.values})
sp1.iplot(kind='pie',labels='labels',values='values',title='Was client an old one or new during previous application??')
#Previous loans distributed to purchase different goods
sp=previous_application['NAME_TYPE_SUITE'].value_counts()
sp1=pd.DataFrame({'labels':sp.index,'values':(sp.values/sp.sum())*100})
sp1.iplot(kind='pie',labels='labels',values='values',title='Goods purchased by clients using previous loans')
#Previous application use case for POS or Car etc
sp=previous_application['NAME_PORTFOLIO'].value_counts()
sp1=pd.DataFrame({'labels':sp.index,'values':sp.values})
sp1.iplot(kind='pie',labels='labels',values='values',title='Was previous application for POS, CARs or CASH??')
#was customer given loan along with other product (cross sale) or did he directly wwalkin with his loan application
sp=previous_application['NAME_PRODUCT_TYPE'].value_counts()
sp1=pd.DataFrame({'labels':sp.index,'values':sp.values})
sp1.iplot(kind='pie',labels='labels',values='values',title='Was loan offered in X-sale or customer walk-in during previous application')

#ways for acquiring clients
sp1=previous_application['CHANNEL_TYPE'].value_counts()
trace=go.Bar(x=sp1.index,y=(sp1/sp1.sum())*100, opacity=0.6)
layout=dict(title='Client Acquisition ways in Previous Application', xaxis=dict(title='Channel Types'),
           yaxis=dict(title='Percentage'))
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='Channels')
#top SELLER_INDUSTRY
sp1=previous_application['NAME_SELLER_INDUSTRY'].value_counts()
trace=go.Bar(x=sp1.index,y=(sp1/sp1.sum())*100, opacity=0.6)
layout=dict(title='Top Industry of the seller', xaxis=dict(title='Seller Industry'),
           yaxis=dict(title='Percentage'))
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='Channels')
#Different loan yields for company in previous application
sp1=previous_application['NAME_YIELD_GROUP'].value_counts()
trace=go.Bar(x=sp1.index,y=(sp1/sp1.sum())*100,opacity=0.6)
layout=dict(title='Risk groups in terms of interest rate charged', xaxis=dict(title='Risk Groups'),
           yaxis=dict(title='Percentage'))
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='Channels')
#all product combinations of previous application
sp1=previous_application['PRODUCT_COMBINATION'].value_counts()
trace=go.Bar(x=sp1.index,y=(sp1/sp1.sum())*100,opacity=0.6)
layout=dict(title='Different Product Combination in Previous Application ', xaxis=dict(title='Product Combination'),
           yaxis=dict(title='Percentage'))
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='Channels')
#did client requested insurance 
yn={1:'Yes',0:'No'}
temp=previous_application['NFLAG_INSURED_ON_APPROVAL'].map(yn).value_counts()
#temp=pd.DataFrame({'labels':sp.index,'values':sp.values})
plt.figure(figsize=(5,5))
plt.pie(temp.values,labels=temp.index,autopct='%1.1f%%', shadow=True)
plt.title('Did client requested insurance ')
plt.legend(loc='best')






