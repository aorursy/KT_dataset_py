import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
print(os.listdir("../input"))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
"""
Problem Statement :
A company disbured car loans to the customers (Asset loan), with the age of loan some customers turned bad while other are performing good .
So given a dataset of good and bad customers along with loan and customer attribute you need to extract the insights from the Data  which explains the characterstics of good and bad customers

Deliverable 
1-code of data exploration
2-Presentation of your insights 

"""
data = pd.read_csv('../input/assignment.csv')
#data = pd.read_csv('assignment.csv')
ky = pd.read_csv('../input/assignment_data_dictionary.csv')
#ky = pd.read_csv('assignment_data_dictionary.csv')
data.info()
data.drop('UID',axis=1,inplace=True)
data.head(1)
data.shape
data.describe()
# Removing outliers
data = data[(data.NOOFYEARSINSAMECITY<100) & (data.RESIDENCESTABILITY<100) & (data.STABILITYINBUSINESS <100) & (data.STOCKVALUEINHAND < 10000000)
           & (data.CURRENTBUSINESSSTABILITY <100) & (data.INCOMEEMPLOYMENT<1000000)] 
data.isnull().sum()/data.shape[0]*100
# Removing columns with no values and null values greater than 15%
data = data.drop(['LANDOWNERSHIP','IRRIGATIONSOURCE','CROPSCULTIVATED','ASSETREGMONTH','ASSETLOCATION','STABILTIYCONFIRMEDTHRU',
                  'DISTFROMDEALERLOCATION','DISTFROMSCELOCATION','ASSETREGYEAR'],axis=1)
# Code to iterate over a list of columns with missing values
li = ['NETINCOMEPERMONTH','ENDUSEOFASSET','OFFICETYPE','VEHICLEMODEL','RESIDENCETYPE','BUSINESSCATEGORY',
      'RESIDENCELOCALITY']
for itm in li:
    data.drop(data[pd.isnull(data[itm])].index,inplace=True)
# Final nmber of null values
data.isnull().sum().sum()
display(data.head(2))
data['Good_Bad'] = data['Good_Bad'].apply(lambda x: 1 if x=='Good' else 0).astype(np.int32)
data['ADDRESSCONFIRMED'] = data['ADDRESSCONFIRMED'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data['RESIDENCETYPE'] = data['RESIDENCETYPE'].apply(lambda x: 1 if x=='R' else 0).astype(np.int32)
data['STABILITYCONFIRMED'] = data['STABILITYCONFIRMED'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data['ISFAMILYINVOLVED'] = data['ISFAMILYINVOLVED'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data['OFFICETYPE'] = data['OFFICETYPE'].apply(lambda x: 1 if x=='O' else 0).astype(np.int32)
data['NAMEBOARDSEEN'] = data['NAMEBOARDSEEN'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data['ISRESIDENCECOMEOFFICE'] = data['ISRESIDENCECOMEOFFICE'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data['NEIGHBOURREF'] = data['NEIGHBOURREF'].apply(lambda x: 1 if x=='POSITIVE' else 0).astype(np.int32)
data['POLITICALLINK'] = data['POLITICALLINK'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data['IMGCONFIRM'] = data['IMGCONFIRM'].apply(lambda x: 1 if x=='Y' else 0).astype(np.int32)
data.head()
data['BUSINESSMARGINNET'] = data['BUSINESSMARGINNET'].astype(np.int64)
data['NOOFEMPLOYEES'] = data['NOOFEMPLOYEES'].astype(np.int32)
data['FOIR'] = data['FOIR'].astype(np.float64)
data = data[(data.FOIR>0) & (data.FOIR<100)]
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['ASSETCOST'])
plt.scatter(data['Good_Bad'],data['ASSETCOST'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['LOANAMOUNT'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['EMI'])
sns.boxplot(data['Good_Bad'],data['EMI'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['TENURE'])
data['DESIGNATION'].value_counts()
data['DESIGNATION'] = data['DESIGNATION'].str.replace(' ','')
data['DESIGNATION'] = data['DESIGNATION'].str.replace('APPLICANT','SELF').replace('BORROWER','SELF').replace('PROPRIETOR','SELF').replace('OWNER','SELF').replace('PROP','SELF').replace('SELFSELF','SELF').replace('APPL','SELF').replace('PROPRITOR','SELF').replace('APLICANT','SELF').replace('OWNED','SELF').replace('GUARANTOR','SELF').replace('PROPERITOR','SELF').replace('PROPREITOR','SELF').replace('PROPERTOR','SELF').replace('INDIVIDUAL','SELF').replace('HIRER','SELF').replace('APPT','SELF').replace('PROPRITER','SELF').replace('PROPERTIOR','SELF').replace('self','SELF').replace('APL','SELF').replace('applicant','SELF').replace('APLLICANT','SELF').replace('APPLT','SELF').replace('PROPERATOR','SELF').replace('OWNE','SELF').replace('APPLCIANT','SELF').replace('APPLICNAT','SELF').replace('APPICANT','SELF').replace('APPRICANT','SELF').replace('OWN','SELF').replace('APPLIACANT','SELF').replace('SELFS','SELF').replace('PRORITER','SELF').replace('APPLICATN','SELF').replace('SEF','SELF').replace('PROPRETIOR','SELF').replace('SELFT','SELF').replace('APPILCANT','SELF').replace('SELIF','SELF').replace('APPLICATION','SELF').replace('PROPRETOR','SELF').replace('PROPIETOR','SELF').replace('SALF','SELF')
data['DESIGNATION'] = data['DESIGNATION'].apply(lambda x : 'OTHERS' if x!='SELF' else 'SELF')
data['DESIGNATION'].value_counts()
data['DESIGNATION'] = data['DESIGNATION'].apply(lambda x: 1 if x=='SELF' else 0).astype(np.int32)
data.groupby(['Good_Bad','DESIGNATION']).describe()
data.groupby(['Good_Bad','RESIDENCETYPE']).describe()
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['NOOFYEARSINSAMECITY'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['RESIDENCESTABILITY'])
data.groupby(['Good_Bad','RESIDENCELOCALITY']).describe()
data['BUSINESSCATEGORY'].value_counts()
data.groupby(['BUSINESSCATEGORY','Good_Bad']).describe()
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['BUSINESSMARGINGROSS'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['INCOMEEMPLOYMENT'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['STABILITYINBUSINESS'])
data.groupby(['Good_Bad','STABILITYCONFIRMED']).describe()
data.groupby(['Good_Bad','ISFAMILYINVOLVED']).describe()
data.groupby(['Good_Bad','OFFICETYPE']).describe()
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['MONTHLYTURNOVER'])
sns.boxenplot(data['Good_Bad'],data['STOCKVALUEINHAND'])
sns.boxenplot(data['Good_Bad'],data['BUSINESSMARGINNET'])
sns.boxenplot(data['Good_Bad'],data['NOOFEMPLOYEES'])
data.groupby(['Good_Bad','NAMEBOARDSEEN']).describe()
data.groupby(['Good_Bad','OFFICELOCALITY']).describe()
data.groupby(['Good_Bad','ISRESIDENCECOMEOFFICE']).describe()
data['ENDUSEOFLOAN'] = data['ENDUSEOFLOAN'].str.replace(' ','')
data['ENDUSEOFLOAN'].value_counts()
data['ENDUSEOFLOAN'] = data['ENDUSEOFLOAN'].apply(lambda x: 1 if x in ['PERSONALUSE','SELF',
                                                                               'OWNUSE','VEHICLEPURCHASE',
                                                                               'PURCHASEOFVEHICLE','PERSONALUSAGE',
                                                                               'FORPURCHASEOFVEHICLE','APPLICANTPERSONALUSE',
                                                                               'TOPURCHASETHEVEHICLE','SELFUSE',
                                                                               'ASSETTOBEUSEDBYAPPLICANT','VEHICLEPURCHASED',
                                                                               'LOANFORVEHICLEPURCHASE','FORUSEDCARPURCHASE',
                                                                               'PERSONALUSE.','PURCHASETHEVEHICLE',
                                                                               'PERSONNALUSE','FORPERSONALUSE',
                                                                               'PERSONALUSEONLY','TOPURCHASEAUSEDCAR',
                                                                               'VEHICLE','USEDCARLOAN',
                                                                               'OWNEDUSE','PURCHASEVEHICLE','BUYINGVEHICLE',
                                                                               'USEDCARPURCHASE','OWN','OWNPURPUS',
                                                                               'OWNPURPOSE','SELFOWN'] else 0)
data.groupby(['Good_Bad','ENDUSEOFLOAN']).describe()
data.groupby(['Good_Bad','NEIGHBOURREF']).describe()
data.groupby(['Good_Bad','POLITICALLINK']).describe()
data['PROFESSIONTYPE'].value_counts()
data.groupby(['Good_Bad','PROFESSIONTYPE']).describe()
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['CURRENTBUSINESSSTABILITY'])
data.drop('APPLICANTDESIGNATION',axis=1,inplace=True)
data.groupby(['Good_Bad','IMGCONFIRM']).describe()
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['TOTALEXPENSEPERMONTH'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['NETINCOMEPERMONTH'])
plt.figure(figsize=(30,30))
sns.boxenplot(data['Good_Bad'],data['FOIR'])
data.head()
data = pd.get_dummies(data,drop_first=True)
data.head()
X = data.iloc[:,1:]
y = data.iloc[:,0].values
from xgboost import XGBClassifier
from xgboost import plot_importance
mdl = XGBClassifier()
mdl.fit(X,y)
plt.figure(figsize=(30,30))
plot_importance(mdl,max_num_features=20)
plt.show()




