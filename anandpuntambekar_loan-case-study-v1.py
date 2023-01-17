import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
file = '/kaggle/input/loan-case-study/Collections Case Study Data.xlsx'

xl = pd.ExcelFile(file)

print(xl.sheet_names)
Loan_Detail = xl.parse('Loan Details')

Loan_Stat_MartoMay= xl.parse('Loan_Status_MartoMay')

Loan_Stat_AprtoJun= xl.parse('Loan_Status_AprtoJun')

Hist_6_Mnth_Dtls= xl.parse('Historical 6 Months Details')

Loan_ID_map= xl.parse('Loan_ID mapping')

Call_Detl= xl.parse('Call Details')

Loan_Detail.head(2)
print(Loan_Detail['Loan_id'].nunique())

## How many rows are there in the Loan_Detais Data frame

print(Loan_Detail.shape[0])
print(Loan_Detail['Debt_to_burden_Ratio'].min())

print(Loan_Detail['Debt_to_burden_Ratio'].max())
print(Loan_Detail[Loan_Detail['Debt_to_burden_Ratio']>1].shape[0])

print(Loan_Detail[Loan_Detail['Debt_to_burden_Ratio']<1].shape[0])
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax= Loan_Detail[Loan_Detail['Debt_to_burden_Ratio']>1]['Debt_to_burden_Ratio'].hist()

ax.set_xlabel("Debt_to_burden_Ratio")

ax.set_ylabel("Frequeny")

ax
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax= Loan_Detail[Loan_Detail['Debt_to_burden_Ratio']<1]['Debt_to_burden_Ratio'].hist()

ax.set_xlabel("Debt_to_burden_Ratio")

ax.set_ylabel("Frequeny")

ax
Loan_Detail.drop(Loan_Detail['total_income'].idxmax(), inplace=True)

Loan_Detail.drop(Loan_Detail['total_income'].idxmax(), inplace=True)

Loan_Detail.drop(Loan_Detail['total_income'].idxmax(), inplace=True)

Loan_Detail.drop(Loan_Detail['total_income'].idxmax(), inplace=True)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



ax = Loan_Detail['total_income'].plot.box()
Loan_Detail = xl.parse('Loan Details')

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = Loan_Detail['TENURE'].plot.box()
Loan_Detail.rename(columns={'Sanctioned Amount':'Sanctioned_Amount'}, inplace=True)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = Loan_Detail['Sanctioned_Amount'].plot.box()
Loan_Detail_V0 = Loan_Detail.drop('Loan_id', 1)

Loan_Detail_V0.head(3)
corr = Loan_Detail.corr()

corr.style.background_gradient(cmap='coolwarm')
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(Loan_Detail.TENURE, Loan_Detail.Sanctioned_Amount)

plt.title("TENURE vs Sanctioned_Amount")

plt.xlabel("TENURE")

plt.ylabel("Sanctioned_Amount")

plt.show()
Loan_Stat_MartoMay.head(2)
### How many unique elements are there in the loan_id column

print(Loan_Detail['Loan_id'].nunique())

print(Loan_Stat_MartoMay['Loan_id'].nunique())
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



ax= Loan_Stat_MartoMay['Month'].value_counts().plot(kind='bar')

ax.set_xlabel("Month")

ax.set_ylabel("Count of Loans")

ax
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



ax = Loan_Stat_MartoMay['TENURE'].plot.box()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



ax = Loan_Stat_MartoMay['PRINBALANCE'].plot.box()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



ax = Loan_Stat_MartoMay['Months on Books'].plot.box()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

Loan_Stat_MartoMay.rename(columns={'Sanctioned Amount':'Sanctioned_Amount'}, inplace=True)

ax = Loan_Stat_MartoMay['Sanctioned_Amount'].plot.box()
corr = Loan_Stat_MartoMay.corr()

corr.style.background_gradient(cmap='coolwarm')
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

Loan_Stat_MartoMay.rename(columns={'Sanctioned Amount':'Sanctioned_Amount'}, inplace=True)

Loan_Stat_MartoMay.drop(Loan_Stat_MartoMay['Sanctioned_Amount'].idxmax(), inplace=True)

plt.scatter(Loan_Stat_MartoMay.PRINBALANCE, Loan_Stat_MartoMay.Sanctioned_Amount)

plt.xlabel("PRINBALANCE")

plt.ylabel("Sanctioned_Amount")

plt.show()
Loan_Stat_AprtoJun.head(2)
## How many unique elements are there in the loan_id column

print(Loan_Detail['Loan_id'].nunique())

print(Loan_Stat_MartoMay['Loan_id'].nunique())

print(Loan_Stat_AprtoJun['Loan_id'].nunique())
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = Loan_Stat_AprtoJun['Bucket'].value_counts().plot(kind='bar')

ax.set_xlabel("Bucket")

ax.set_ylabel("Count of Loans")

ax
Hist_6_Mnth_Dtls.head(2)
Loan_ID_map.head(2)
print(Loan_ID_map['Loanid'].nunique())

print(Loan_ID_map['Application_id'].nunique())
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax= Call_Detl['month'].value_counts().plot(kind='bar')

ax.set_xlabel("Month")

ax.set_ylabel("Frequeny")

ax
figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')

plt.rcParams.update({'font.size': 8})

ax = Call_Detl['Login_ID'].value_counts().plot(kind='bar')

ax.set_xlabel("Agent")

ax.set_ylabel("Frequeny")

ax


Loan_Stat_Mrg = pd.concat([Loan_Stat_MartoMay,Loan_Stat_AprtoJun], ignore_index=True).drop_duplicates()



Loan_Stat_March= Loan_Stat_Mrg [Loan_Stat_Mrg ['Month']== 'March']

Loan_Stat_March.reset_index(inplace = True,drop=True) 

print(Loan_Stat_March['Loan_id'].nunique())

print(Loan_Stat_March['Loan_id'].shape)





Loan_Stat_April= Loan_Stat_Mrg [Loan_Stat_Mrg ['Month']== 'April']

Loan_Stat_April.reset_index(inplace = False,drop=True) 

print(Loan_Stat_April['Loan_id'].nunique())

print(Loan_Stat_April['Loan_id'].shape)





Loan_Stat_May= Loan_Stat_Mrg [Loan_Stat_Mrg ['Month']== 'May']

Loan_Stat_May.reset_index(inplace = False,drop=True) 

print(Loan_Stat_May['Loan_id'].nunique())

print(Loan_Stat_May['Loan_id'].shape)



Loan_Stat_June= Loan_Stat_Mrg [Loan_Stat_Mrg ['Month']== 'June']

Loan_Stat_June.reset_index(inplace = False,drop=True) 

print(Loan_Stat_June['Loan_id'].nunique())

print(Loan_Stat_June['Loan_id'].shape)



Loan_Stat_March = Loan_Stat_March[['Loan_id','Bucket','PRINBALANCE']]

Loan_Stat_April= Loan_Stat_April[['Loan_id','Bucket','PRINBALANCE']]

Loan_Stat_May = Loan_Stat_May[['Loan_id','Bucket','PRINBALANCE']]

Loan_Stat_June= Loan_Stat_June[['Loan_id','Bucket','PRINBALANCE']]



Loan_Stat_March.rename(columns={"Bucket": "Bucket_March"},inplace= True)

Loan_Stat_April.rename(columns={"Bucket": "Bucket_April"},inplace= True)

Loan_Stat_May.rename(columns={"Bucket": "Bucket_May"},inplace= True)

Loan_Stat_June.rename(columns={"Bucket": "Bucket_June"},inplace= True)



Loan_Stat_March.rename(columns={"PRINBALANCE": "PRINBALANCE_March"},inplace= True)

Loan_Stat_April.rename(columns={"PRINBALANCE": "PRINBALANCE_April"},inplace= True)

Loan_Stat_May.rename(columns={"PRINBALANCE": "PRINBALANCE_May"},inplace= True)

Loan_Stat_June.rename(columns={"PRINBALANCE": "PRINBALANCE_June"},inplace= True)

Temp= pd.merge(Loan_Stat_March,Loan_Stat_April, on='Loan_id')



Loan_Stat= pd.merge(Temp,Loan_Stat_May, on='Loan_id')

Loan_Stat= pd.merge(Loan_Stat,Loan_Stat_June, on='Loan_id')

Loan_Stat[Loan_Stat['Bucket_March']=='TB0'].shape[0]



print("No of caes with TB0 to TBX (2,3,4,5,6):", Loan_Stat[ (Loan_Stat['Bucket_March']=='TB0') &(Loan_Stat['Bucket_April']!='TB0')&(Loan_Stat['Bucket_April']!='REGULAR')].shape[0])

print("No of caes with TB0 in March:",Loan_Stat[ (Loan_Stat['Bucket_March']=='TB0')].shape[0])

print("% of cases:",100*(Loan_Stat[ (Loan_Stat['Bucket_March']=='TB0') &(Loan_Stat['Bucket_April']!='TB0')&(Loan_Stat['Bucket_April']!='REGULAR')].shape[0])/ (Loan_Stat[ (Loan_Stat['Bucket_March']=='TB0')].shape[0]) )
Loan_Stat.head(2)


print("No of caes with TB0 to TBX (2,3,4,5,6):", Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0') &(Loan_Stat['Bucket_May']!='TB0')&(Loan_Stat['Bucket_May']!='REGULAR')].shape[0])

print("No of caes with TB0 in April:",Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0')].shape[0])

print("% of cases:",100*(Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0') &(Loan_Stat['Bucket_May']!='TB0')&(Loan_Stat['Bucket_April']!='REGULAR')].shape[0])/ (Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0')].shape[0]) )
print("No of caes with TB0 to TBX (2,3,4,5,6):", Loan_Stat[ (Loan_Stat['Bucket_May']=='TB0') &(Loan_Stat['Bucket_June']!='TB0')&(Loan_Stat['Bucket_June']!='REGULAR')].shape[0])

print("No of caes with TB0 in April:",Loan_Stat[ (Loan_Stat['Bucket_May']=='TB0')].shape[0])

print("% of cases:",100*(Loan_Stat[ (Loan_Stat['Bucket_May']=='TB0') &(Loan_Stat['Bucket_June']!='TB0')&(Loan_Stat['Bucket_April']!='REGULAR')].shape[0])/ (Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0')].shape[0]) )
#print(Loan_Stat[ (Loan_Stat['Bucket_March']=='TB0') &(Loan_Stat['Bucket_April']!='TB0')&(Loan_Stat['Bucket_April']!='REGULAR')]['PRINBALANCE_April'].sum())

#Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0') &(Loan_Stat['Bucket_May']!='TB0')&(Loan_Stat['Bucket_May']!='REGULAR')]['PRINBALANCE_May'].sum()





A = Loan_Stat[ (Loan_Stat['Bucket_March']=='TB0') &(Loan_Stat['Bucket_April']!='TB0')&(Loan_Stat['Bucket_April']!='REGULAR')]['PRINBALANCE_April'].sum()

B = Loan_Stat[ (Loan_Stat['Bucket_May']=='TB0')].shape[0]

C= (A/B)

C
Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0') &(Loan_Stat['Bucket_May']!='TB0')&(Loan_Stat['Bucket_May']!='REGULAR')]['PRINBALANCE_May'].sum()





A = Loan_Stat[ (Loan_Stat['Bucket_April']=='TB0') &(Loan_Stat['Bucket_May']!='TB0')&(Loan_Stat['Bucket_May']!='REGULAR')]['PRINBALANCE_May'].sum()

B = Loan_Stat[ (Loan_Stat['Bucket_May']=='TB0')].shape[0]

C= (A/B)

C


A = Loan_Stat[ (Loan_Stat['Bucket_May']=='TB0') &(Loan_Stat['Bucket_June']!='TB0')&(Loan_Stat['Bucket_May']!='REGULAR')]['PRINBALANCE_June'].sum()

B = Loan_Stat[ (Loan_Stat['Bucket_June']=='TB0')].shape[0]

C= (A/B)

C
print(Call_Detl['Application_Id'].nunique())

C= Call_Detl['Application_Id'].nunique()



D= Call_Detl[Call_Detl['month']==3]['total_contacts'].sum()

print(D)



D/C
Call_Detl['RPC']= Call_Detl['Right_Party_Contact']/Call_Detl['total_contacts']

Call_Detl['PTP']= Call_Detl['Promise_to_pay']/Call_Detl['total_contacts']



print("RPC for Month 3:",Call_Detl[Call_Detl['month']==3]['RPC'].mean())

print("RPC for Month 4:",Call_Detl[Call_Detl['month']==4]['RPC'].mean())

print("RPC for Month 5:",Call_Detl[Call_Detl['month']==5]['RPC'].mean())



print("PTP for Month 3:",Call_Detl[Call_Detl['month']==3]['PTP'].mean())

print("PTP for Month 4:",Call_Detl[Call_Detl['month']==4]['PTP'].mean())

print("PTP for Month 5:",Call_Detl[Call_Detl['month']==5]['PTP'].mean())
# PRINBALANCE : The principle outstanding balance of the account remaining to be paid by the account

# Tenure: The number of months the loan has to be repaid

# Months on Books: The number of months since the start of the loan



Loan_Stat_Mrg['Month_Remaining']= Loan_Stat_Mrg['TENURE']-Loan_Stat_Mrg['Months on Books']



Loan_Stat_Mrg['Bal_Remaining']= Loan_Stat_Mrg['Sanctioned Amount']-Loan_Stat_Mrg['PRINBALANCE']



Loan_Stat_Mrg['Prop_Bal_Remaining']= (Loan_Stat_Mrg['Sanctioned Amount']-Loan_Stat_Mrg['PRINBALANCE'])/Loan_Stat_Mrg['Sanctioned Amount']



Loan_Stat_Mrg['Prop_Month_Remaining']= (Loan_Stat_Mrg['TENURE']-Loan_Stat_Mrg['Months on Books'])/Loan_Stat_Mrg['TENURE']



Loan_Stat_Mrg[['Months on Books','Bal_Remaining']].corr()
#Loan_Stat_Mrg.plot(x='Months on Books', y='Prop_Bal_Remaining', style='o')

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.rcParams.update({'font.size': 8})

plt.scatter(Loan_Stat_Mrg.Prop_Month_Remaining, Loan_Stat_Mrg.Prop_Bal_Remaining)

plt.rcParams.update({'font.size': 8})

plt.xlabel("Months on Books")

plt.ylabel("Prop_Bal_Remaining" )

plt.show()
#Loan_Stat_Mrg.plot(x='Prop_Month_Remaining', y='Prop_Bal_Remaining', style='o')

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(Loan_Stat_Mrg.Prop_Month_Remaining, Loan_Stat_Mrg.Prop_Bal_Remaining)

plt.rcParams.update({'font.size': 8})

plt.xlabel("Prop_Month_Remaining")

plt.ylabel("Prop_Bal_Remaining" )

plt.show()
temp = pd.merge(Call_Detl[['Application_Id', 'Login_ID']], Loan_ID_map, left_on='Application_Id', right_on='Application_id').drop('Application_id', 1)



temp_agent= pd.merge(temp, Loan_Detail[['Loan_id','Sanctioned_Amount']], left_on='Loanid', right_on='Loan_id').drop('Loanid', 1)



KPI_Agent = pd.merge(temp_agent, Hist_6_Mnth_Dtls, left_on='Loan_id', right_on='Loan_id')



KPI_Agent['IDX']=KPI_Agent.index





KPI_Agent_grp=KPI_Agent.groupby(['Login_ID','Month']).agg({'Sanctioned_Amount': 'sum', 'paidcure': 'mean','Loan_id':'count',

                                            'paidcure':'median' ,

                                            'paiduncure':'median' ,

                                            'unpaid':'count' ,

                                            'rollb':'median' ,

                                            'rollf':'median' ,

                                            'num6mosdel':'count' ,

                                            'num3mosdel':'median' ,

                                            'num6mosdel_2plus':'median' ,

                                            'num3mosdel_2plus':'median' ,

                                            'max6del':'median' ,

                                            'max3del':'median'

                                            

                                            

                                           })

#KPI_Agent['Month'] = KPI_Agent['Month'].map({'March': 3, 'April': 4, 'May': 5})



#KPI_Agent.Month.replace(['March', 'April', 'May'], [3, 4,5], inplace=True)



KPI_Agent_grp['Agent']=KPI_Agent_grp.index

KPI_Agent_grp['Month']=KPI_Agent_grp.index



KPI_Agent_grp['IDX']=KPI_Agent_grp.index

for i in range (0,KPI_Agent_grp.shape[0]):

    KPI_Agent_grp['Agent'][i]=KPI_Agent_grp['IDX'][i][0]

    KPI_Agent_grp['Month'][i]=KPI_Agent_grp['IDX'][i][1]

#KPI_Agent_grp['IDX'][i][0]

#KPI_Agent_grp.shape[0]



KPI_Agent_grp=KPI_Agent.groupby(['Login_ID','Month']).agg({'Sanctioned_Amount': 'sum', 'paidcure': 'mean','Loan_id':'count',

                                            'paidcure':'median' ,

                                            'paiduncure':'median' ,

                                            'unpaid':'count' ,

                                            'rollb':'median' ,

                                            'rollf':'median' ,

                                            'num6mosdel':'count' ,

                                            'num3mosdel':'median' ,

                                            'num6mosdel_2plus':'median' ,

                                            'num3mosdel_2plus':'median' ,

                                            'max6del':'median' ,

                                            'max3del':'median'

                                            

                                            

                                           })

#KPI_Agent['Month'] = KPI_Agent['Month'].map({'March': 3, 'April': 4, 'May': 5})



#KPI_Agent.Month.replace(['March', 'April', 'May'], [3, 4,5], inplace=True)



KPI_Agent_grp['Agent']=KPI_Agent_grp.index

KPI_Agent_grp['Month']=KPI_Agent_grp.index



KPI_Agent_grp['IDX']=KPI_Agent_grp.index

for i in range (0,KPI_Agent_grp.shape[0]):

    KPI_Agent_grp['Agent'][i]=KPI_Agent_grp['IDX'][i][0]

    KPI_Agent_grp['Month'][i]=KPI_Agent_grp['IDX'][i][1]

    

    

KPI_Agent_grp['Month']=KPI_Agent_grp['Month'].replace(['March'], 3)

KPI_Agent_grp['Month']=KPI_Agent_grp['Month'].replace(['April'], 4)

KPI_Agent_grp['Month']=KPI_Agent_grp['Month'].replace(['May'], 5)
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



%matplotlib inline

plt.rcParams['figure.figsize']=15,50



ax = sns.lineplot(x="Month", y="Sanctioned_Amount", hue="Agent",

                  #units="subject", estimator=None, lw=1,

                  data=KPI_Agent_grp)
%matplotlib inline

plt.rcParams['figure.figsize']=15,50



ax = sns.lineplot(x="Month", y="Loan_id", hue="Agent",

                  #units="subject", estimator=None, lw=1,

                  data=KPI_Agent_grp)
%matplotlib inline

plt.rcParams['figure.figsize']=15,50



ax = sns.lineplot(x="Month", y="Loan_id", hue="Agent",

                  #units="subject", estimator=None, lw=1,

                  data=KPI_Agent_grp)
#df=pd.merge(Loan_Detail[['Loan_id','Debt_to_burden_Ratio', 'total_income','TENURE','Sanctioned_Amount']], Hist_6_Mnth_Dtls[['Loan_id', 'max6del']], on='Loan_id').drop('Loan_id', 1)

df=pd.merge(Loan_Detail[['Loan_id','Debt_to_burden_Ratio', 'total_income','TENURE','Sanctioned_Amount']], Hist_6_Mnth_Dtls[['Loan_id', 'max6del']], on='Loan_id')



df=pd.merge(Loan_ID_map,df,  left_on='Loanid',  right_on='Loan_id')



df=pd.merge(df,Call_Detl[['Login_ID', 'Application_Id','RPC','PTP' ]],  left_on='Application_id',  right_on='Application_Id')



df= df.drop('Loanid', 1)

df= df.drop('Application_id', 1)

df= df.drop('Application_Id', 1)

df= df.drop('Loan_id', 1)





df['max6del']= df['max6del'].replace([2], 1)

df['max6del']= df['max6del'].replace([3], 1)

df['max6del']= df['max6del'].replace([4], 1)

df['max6del']= df['max6del'].replace([5], 1)







agent_info=df.groupby(['Login_ID']).mean()

agent_info['Login_ID']=agent_info.index



agent_info.rename(columns={'Debt_to_burden_Ratio':'Agt_DTB_Ratio_mean',

                          'total_income':'Agt_DTB_Ratio_mean',

                          'TENURE':'AgtT_ENURE_mean',

                          'Sanctioned_Amount':'AgtT_Sanctioned_Amount',

                           'RPC':'Agt_RPC_mean',

                           'PTP':'Agt_PTP_mean',

                           'Login_ID':'Agt_Login_ID',

                           'max6del':'Agt_max6del',

                          }, 

                 inplace=True)







agent_info=agent_info.reset_index(drop=True)



df=pd.merge(df,agent_info , left_on='Login_ID',  right_on='Agt_Login_ID')



df=df.drop('Agt_Login_ID', 1)

df=df.drop('Login_ID', 1)

df.head()

df['max6del']= df['max6del'].replace([0], '0')

df['max6del']= df['max6del'].replace([1], '1')

df['max6del']=df['max6del'].astype(str)

df.head()
tmp=df.groupby(['max6del']).mean()

tmp
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

#df[['Debt_to_burden_Ratio','max6del']].boxplot(by='max6del')
df.plot(figsize=(1,1))

#df[['total_income','max6del']].boxplot(by='max6del')
df.plot(figsize=(1,1))

#df[['TENURE','max6del']].boxplot(by='max6del')


df.plot(figsize=(1,1))

#df[['Sanctioned_Amount','max6del']].boxplot(by='max6del')


df.plot(figsize=(1,1))

#df[['Agt_max6del','max6del']].boxplot(by='max6del')


df.plot(figsize=(1,1))

#df[['Agt_DTB_Ratio_mean','max6del']].boxplot(by='max6del')
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



#split dataset in features and target variable



#feature_cols = ['Debt_to_burden_Ratio','total_income','TENURE','Sanctioned_Amount','RPC','PTP','Agt_DTB_Ratio_mean','Agt_DTB_Ratio_mean','AgtT_ENURE_mean','AgtT_Sanctioned_Amount','Agt_max6del','Agt_RPC_mean','Agt_PTP_mean']



feature_cols = ['Debt_to_burden_Ratio','total_income','TENURE','Sanctioned_Amount','Agt_DTB_Ratio_mean','Agt_DTB_Ratio_mean','AgtT_ENURE_mean','AgtT_Sanctioned_Amount','Agt_max6del','Agt_RPC_mean','Agt_PTP_mean']

X = df[feature_cols] # Features

y = df['max6del'] # Target variable
#df['max6del']=df['max6del'].astype('int')

df['max6del'].dtypes
# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))



print("Classification Report")

print(classification_report(y_test, y_pred))
#! conda install pydotplus --yes
df_temp=df[df['Sanctioned_Amount']>df['Sanctioned_Amount'].mean()]

df_temp.reset_index(inplace = True,drop=True) 
df_tmp=df_temp.groupby(['max6del']).mean()

df_tmp