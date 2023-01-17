

import numpy as np 
import pandas as pd
import pandas as pd 
data=pd.read_csv('bbs_cust_base_scfy_20200210 (1).csv',sep=';')
#splittin the columns

data[['image',
 'newacct_no',
 'line_stat',
 'bill_cycl',
 'serv_type',
 'serv_code',
 'tenure',
 'effc_strt_date',
 'effc_end_date',
 'contract_month',
 'ce_expiry',
 'secured_revenue',
 'bandwidth',
 'term_reas_code',
 'term_reas_desc',
 'complaint_cnt',
 'with_phone_service',
 'churn',
 'current_mth_churn','extra']]=data['image,newacct_no,line_stat,bill_cycl,serv_type,serv_code,tenure,effc_strt_date,effc_end_date,contract_month,ce_expiry,secured_revenue,bandwidth,term_reas_code,term_reas_desc,complaint_cnt,with_phone_service,churn,current_mth_churn'].str.split(',',expand=True,)
#dropping the unwanted columns
data=data.drop(['image,newacct_no,line_stat,bill_cycl,serv_type,serv_code,tenure,effc_strt_date,effc_end_date,contract_month,ce_expiry,secured_revenue,bandwidth,term_reas_code,term_reas_desc,complaint_cnt,with_phone_service,churn,current_mth_churn','extra'],axis=1)
data
#as per the problem statement I have to drop these and these have no affect in the model 
data=data.drop(['line_stat','bill_cycl','serv_type','serv_code'],axis=1)
data.info()
#number of unique value in each columns
for i in data.columns:
    print(data[i].unique())
#by printing nunique() we can also get count
#finding the missing value 
#as in this dataset missing value is representd by BLANK SPACE 
a=data['term_reas_code'][0]
for i in data.columns:
    print('length of empty data is ',len(data[data[i]==a]),'of',i,'value')
#as we can see that  term_reas_code value has large data 
#which is empty so this will not be that much of usefull
#so we drop them 

data=data.drop(['term_reas_code','term_reas_desc'],axis=1)
#dropping 1927 index of effc_strt_date 
#effc_end_date ,contract_month ,ce_expiry 
#as they wil not affect that much 
data=data.drop(data[data['effc_strt_date']==a].index.to_list())
#Converting categorical into numerical

mapiings= {'0':0, '1':1, '2':2, '3':3, '5':5, '4':4, ' customer/ user pass away':8, '7':7,
       '6':6}

data['complaint_cnt']=data['complaint_cnt'].map(mapiings)


##########

map_col={'Y':1,'N':0,'0':0}

data['with_phone_service']=data['with_phone_service'].map(map_col)

data['churn']=data['churn'].map(map_col)

data['current_mth_churn']=data['current_mth_churn'].map(map_col)

#########3##
map_ing={'30M':0, '10M':0,'BELOW 10M':0, '50M':0,'100M':1,'100M (FTTO)':1,'300M (FTTO)':2, '1000M (FTTO)':2,'500M (FTTO)':2}

data['bandwidth'] = data['bandwidth'].map(map_ing)
#now changing the data type 
#as in this dataset all column has string type 

data[['contract_month','ce_expiry']]=data[['contract_month','ce_expiry']].astype('int')
data[['secured_revenue']]=data[['secured_revenue']].astype('float')

#####################
data['effc_strt_date'] = pd.to_datetime(data['effc_strt_date'],dayfirst=True)
data['effc_end_date'] = data.effc_end_date.astype('datetime64[ns]')
#changing time and date fromat into year wise 

data['start_month']=data['effc_strt_date'].dt.month
data['start_year']=data['effc_strt_date'].dt.year

data['end_month']=data['effc_end_date'].dt.month
data['end_year']=data['effc_end_date'].dt.year

data=data.drop(['effc_strt_date','effc_end_date'],axis=1)
#set unique id as index 
data.set_index(keys='newacct_no',inplace = True)
#those customers who have highest tenure
data['tenure'] = data.groupby('newacct_no').tenure.max()
#I think current mth churn has not that 
#valuable because I get to good accuracy when 
#left current_mth_churn

data['churn']=data.churn

data=data.drop(['image','current_mth_churn'],axis=1)
import matplotlib.pyplot as plt 
import seaborn as sns
#correlation between each variable 
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,cbar=True)
plt.show()
data=data.drop_duplicates()
#For now our dataset is slightly imbalance 
#but for now it not that much of imbalance 
#so I think I have go forward and skip this step 
plt.bar(x=['N','Y'],height=[len(data[data['churn']==0]),len(data[data['churn']==1])])
#splitting the dataset in two parts 
#like target and train
target=data['churn']
train=data.drop(['churn'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.25,random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

rfc.fit(X_train,y_train)
pr=rfc.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(pr,y_test))
#Total customers actually churned in last 6 months:
data[(data.ce_expiry >= -6) & (data.ce_expiry <= 0)  & (data.churn == 1)].index.nunique()
#Total customers predicted as churned by model: 
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pr)

feature_imp = pd.Series(lr.feature_importances_,index=X_train.columns).sort_values(ascending=False)
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
