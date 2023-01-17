# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
telecom_churn = pd.read_csv('/kaggle/input/telecom-churn/telecom_churn_data.csv')
telecom_churn.head()
telecom_churn.shape
pd.set_option('display.max_rows',250)
telecom_churn.isnull().sum()
percentage = {}

for i in list(telecom_churn.columns):

    percentage[i] = round(telecom_churn[i].isnull().sum()/len(telecom_churn)*100, 2)

max_nullvalues = []

for i in percentage:

    if percentage[i] > 70:

        max_nullvalues.append(i)

max_nullvalues

telecom_churn[['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9']] =telecom_churn[['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9']].fillna(method = 'ffill')
telecom_churn.fillna(0, inplace = True)
telecom_churn.dtypes
columns = list(telecom_churn.columns)

date_time = []

for i in columns:

    if telecom_churn[i].dtype == 'object':

        date_time.append(i)



        
for i in date_time:

    telecom_churn[i] = pd.to_datetime(telecom_churn[i])
telecom_churn.head(10)
monthly = telecom_churn[['arpu_6', 'arpu_7', 'arpu_8', 'arpu_9']].mean()

months = ['June', 'July', 'August', 'September']

barplt = sns.barplot(x = months, y = monthly)

for x in barplt.patches:

    barplt.annotate(format(x.get_height(), '.2f'), (x.get_x() + x.get_width()/2.,x.get_height()), ha = 'center', va = 'center', xytext = (0,5), textcoords = 'offset points')

sns.pointplot(x = months, y = monthly)

plt.title('Average Revenue Per User')

plt.show()
offnet = telecom_churn[['offnet_mou_6', 'offnet_mou_7', 'offnet_mou_8', 'offnet_mou_9']].mean()

onnet = telecom_churn[['onnet_mou_6','onnet_mou_7', 'onnet_mou_8', 'onnet_mou_9' ]].mean()

months = ['June', 'July', 'August', 'September']

fig, axes = plt.subplots(1,2, figsize = (10,6))

calls = [onnet,offnet]

fig.suptitle('ONNET & OFFNET Calls')

for i, type in enumerate(calls):

    ax = sns.pointplot(months, type, ax = axes[i])

plt.show()
local_ic_call = list(telecom_churn[['loc_ic_mou_6', 'loc_ic_mou_7', 'loc_ic_mou_8', 'loc_ic_mou_9']].mean())

local_og_call = list(telecom_churn[['loc_og_mou_6', 'loc_og_mou_7', 'loc_og_mou_8', 'loc_og_mou_9']].mean())

std_ic_call = list(telecom_churn[['std_ic_mou_6', 'std_ic_mou_7', 'std_ic_mou_8', 'std_ic_mou_9']].mean())

std_og_call = list(telecom_churn[['std_og_mou_6', 'std_og_mou_7' , 'std_og_mou_8', 'std_og_mou_9']].mean())

isd_ic_call = list(telecom_churn[['isd_ic_mou_6','isd_ic_mou_7', 'isd_ic_mou_8', 'isd_ic_mou_9' ]].mean())

isd_og_call = list(telecom_churn[['isd_og_mou_6', 'isd_og_mou_7', 'isd_og_mou_8', 'isd_og_mou_9']].mean())
incomingcalls = pd.DataFrame({'months': months, 'local_call':local_ic_call, 'std_call': std_ic_call, 'isd_call': isd_ic_call})

outgoingcalls = pd.DataFrame({'months': months, 'local_call': local_og_call, 'std_call': std_og_call, 'isd_call': isd_og_call})

incomingcalls = incomingcalls.melt('months', var_name=  'cols', value_name =  'vals')

outgoingcalls = outgoingcalls.melt('months', var_name = 'cols', value_name = 'vals')
calltype = [incomingcalls, outgoingcalls]

fig , axes = plt.subplots(1,2, figsize = (10,5))

title = ['IncomingCalls', 'OutgoingCalls']

for i,values in enumerate(calltype):

    ax = sns.pointplot(x = 'months', y = 'vals', hue = 'cols', data = values, ax = axes[i])

    ax.set_title(title[i])

plt.show()
incoming_roaming = telecom_churn[['roam_ic_mou_6' , 'roam_ic_mou_7', 'roam_ic_mou_8', 'roam_ic_mou_9']].mean()

outgoing_roaming = telecom_churn[['roam_og_mou_6', 'roam_og_mou_7', 'roam_og_mou_8', 'roam_og_mou_9']].mean()

fig,axes = plt.subplots(1,2,figsize = (10,6))

fig.suptitle('Roaming Calls')

roaming_call = [incoming_roaming, outgoing_roaming]

title = ['incoming_call', 'outgoing_call']

for i,call in enumerate(roaming_call):

    ax = sns.barplot(months , y = call, ax = axes[i])

    ax.set_title(title[i])

plt.show()    
fig,axes = plt.subplots(2,2, figsize = (14,7))

sns.distplot(telecom_churn['total_rech_num_6'],color = 'blue', ax = axes[0,0] )

sns.distplot(telecom_churn['total_rech_num_7'],color = 'red', ax = axes[0,1])

sns.distplot(telecom_churn['total_rech_num_8'], color = 'green', ax = axes[1,0] )

sns.distplot(telecom_churn['total_rech_num_9'], ax = axes[1,1])

total_rech_amt = telecom_churn[['total_rech_amt_6', 'total_rech_amt_7', 'total_rech_amt_8', 'total_rech_amt_9']]

telecom_churn['avg_rech_amt'] = total_rech_amt.mean(axis = 1)

inactive_numbers = list(telecom_churn['mobile_number'][telecom_churn['avg_rech_amt']==0])

print("Numbers which haven't been recharged for past 4 months \n")

print(inactive_numbers)
below_minimum = list(telecom_churn['mobile_number'][(telecom_churn['avg_rech_amt']>0) & (telecom_churn['avg_rech_amt']<=50)])

print("Numbers which are active but have low recharge amount \n")

print(below_minimum)
medium_ranged = list(telecom_churn['mobile_number'][(telecom_churn['avg_rech_amt']>50)&(telecom_churn['avg_rech_amt']<=320)])

print('Numbers which are being actively recharged \n')

print(medium_ranged)
above_average = list(telecom_churn['mobile_number'][(telecom_churn['avg_rech_amt']>320) & (telecom_churn['avg_rech_amt']<700)])

print(above_average)
high_rech = list(telecom_churn['mobile_number'][telecom_churn['avg_rech_amt']>=700])

print('Numbers being recharged with high amounts \n')

print(high_rech)
Inactive = len(inactive_numbers)

Low = len(below_minimum)

Average = len(medium_ranged)

Above_average = len(above_average)

High = len(high_rech)

customers_rech = [Inactive, Low, Average, Above_average, High]

lab = ['Inactive','Low', 'Average', 'Above Avg' , 'High']

fig, axes = plt.subplots(1,2, figsize = (14,7))

fig.suptitle('Customers Categorized Based On There Average Recharge Amount')

axes[0].pie(customers_rech,labels = lab, autopct = '%.2f%%', pctdistance = 0.6, textprops = {'size': 12})

barp = sns.barplot(x = lab, y = customers_rech,ax = axes[1])

for x in barp.patches:

    barp.annotate(format(x.get_height()),(x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0,6), textcoords = 'offset points' )

column_names = list(telecom_churn.columns)

data_volume = []

for i in column_names:

    if 'vol' in i:

        data_volume.append(i)

telecom_churn['avg_volume'] = telecom_churn[data_volume].mean(axis = 1)

No_Data_Usage = list(telecom_churn['mobile_number'][telecom_churn['avg_volume']==0])

Lowdata_users = list(telecom_churn['mobile_number'][(telecom_churn['avg_volume']>0) & (telecom_churn['avg_volume'] <= 50)])

avgdata_users = list(telecom_churn['mobile_number'][(telecom_churn['avg_volume']> 50) & (telecom_churn['avg_volume']<=85)])

highdata_users = list(telecom_churn['mobile_number'][telecom_churn['avg_volume']> 85])
No_usage = len(No_Data_Usage)

low_usage = len(Lowdata_users)

avg_usage = len(avgdata_users)

high_usage = len(highdata_users)

Internet_usage = [No_usage, low_usage, avg_usage, high_usage]

lab = ['No_usage', 'low_usage', 'avg_usage', 'high_usage']

fig, axes = plt.subplots(1,2, figsize = (14,7))

fig.suptitle('Customers Categorization Based on there Internet Usage')

axes[0].pie(Internet_usage, labels = lab, autopct = '%.2f%%', textprops = {'size': 13})

barp = sns.barplot(x = lab, y = Internet_usage, ax = axes[1])

for x in barp.patches:

    barp.annotate(format(x.get_height()),(x.get_x() + x.get_width()/2., x.get_height()),ha = 'center', va = 'center', xytext = (0,6), textcoords = 'offset points')
avg_revenue = []

arpu2g_revenue = []

arpu3g_revenue = []

for i in columns:

    if 'arpu_2g_' in i:

        arpu2g_revenue.append(i)

    elif 'arpu_3g_' in i:

        arpu3g_revenue.append(i)

    elif 'arpu' in i:

        avg_revenue.append(i)

avg_revenue = telecom_churn[avg_revenue].mean()

arpu2g_revenue = telecom_churn[arpu2g_revenue].mean()

arpu3g_revenue = telecom_churn[arpu3g_revenue].mean()

sns.pointplot(x = months, y = avg_revenue, color = 'red')

sns.pointplot(x = months, y = arpu2g_revenue, color = 'green')

sns.barplot(x = months, y = arpu3g_revenue)

plt.legend(labels = ['avg_revenue', '2g_revenue', '3g_revenue'])

plt.title('Average Revenue Per User')

plt.show()
telecom_churn['avg_rech_amt'].describe()
telecom_churn.shape
telecom_churn.columns[telecom_churn.columns.str.contains('rech_amt')]
telecom_churn['avg_rech_6&7'] = telecom_churn[['total_rech_amt_6', 'total_rech_amt_7']].mean(axis = 1)

print('70% of average recharge amount of 6&7 month is ', np.percentile(telecom_churn['avg_rech_6&7'], 70))
telecom_churn = telecom_churn[telecom_churn['avg_rech_6&7']>=368.5]
telecom_churn
telecom_churn['churn'] = np.where(telecom_churn[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']].sum(axis =1)==0, 1, 0)
a = list(telecom_churn['churn'].value_counts())

lab = ['No Churn', 'Churn']

plt.figure(figsize = (7,5))

plt.pie(a, labels = lab, explode = (0.1, 0), autopct = "%.2f%%")

plt.title('Customer Churn Distribution')

plt.show()
telecom_churn['tenure_months'] = telecom_churn['aon']/30

print('Age of customers in the data is distributed from ', int(telecom_churn['tenure_months'].min()) , 'months to', int(telecom_churn['tenure_months'].max()), 'months')
plt.figure(figsize = (9,5))

sns.distplot(telecom_churn['tenure_months'],bins = int(155/5), hist = True)

plt.xlabel('Tenure(months)')

plt.ylabel('No of Customers')

plt.title('Distribution of customers based on tenure')

plt.show()
ten_range = [6,12,24,60,150]

lab = ['6-12 months', '1-2 years', '2-5 years', '5 years & above']

telecom_churn['tenure_range'] = pd.cut(telecom_churn['tenure_months'], ten_range, labels = lab)

plt.figure(figsize =(7,5))

count = sns.countplot(telecom_churn['tenure_range'], hue = telecom_churn['churn'])

for x in count.patches:

    count.annotate(format(x.get_height(), '.2f'),(x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center',xytext = (0,6), textcoords = 'offset points' )

plt.xlabel('Tenure')

plt.ylabel('No of customers')

plt.title('Customer Distribution')

plt.show()
a = telecom_churn[['loc_ic_mou_6', 'std_ic_mou_6', 'isd_ic_mou_6', 'roam_ic_mou_6','ic_others_6']].sum(axis = 1)

print('comparing above value (a) with total_ic_mou_6')

print(a.head())

print(telecom_churn['total_ic_mou_6'].head())
drop_col = telecom_churn.columns[telecom_churn.columns.str.contains('loc|std|isd|spl|roam|others', regex = True)].tolist()

telecom = telecom_churn.drop(drop_col, axis = 1)
to_change = telecom.columns[telecom.columns.str.contains('vbc')].tolist()

to_change
new_name = ['vbc_3g_8','vbc_3g_7','vbc_3g_6','vbc_3g_9' ]

telecom.rename(columns = {'jun_vbc_3g': 'vbc_3g_6', 'jul_vbc_3g': 'vbc_3g_7', 'aug_vbc_3g': 'vbc_3g_8','sep_vbc_3g':'vbc_3g_9'}, inplace = True)
dele_col = [i for i in max_nullvalues if 'total' not in i]

telecom.drop(dele_col, axis = 1, inplace = True)
telecom.columns
dates_col = telecom.columns[telecom.columns.str.contains('date|rch',regex = True)].tolist()

dates_col
telecom.drop(dates_col, axis = 1, inplace = True)

telecom.drop(['mobile_number', 'circle_id', 'avg_rech_amt', 'avg_volume','aon', 'avg_rech_6&7' ,'tenure_range'], axis = 1, inplace = True)
telecom
telecom.corr()
plt.figure(figsize = (16,12))

sns.heatmap(telecom.corr())
plt.figure(figsize = (17,7))

telecom.corr()['churn'].sort_values(ascending = False).plot(kind = 'bar')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
x = telecom.drop('churn' , axis = 1)

y = telecom['churn']
telecom.drop('churn', axis = 1 , inplace = True)
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(X,y ,test_size= 0.3, random_state = 42)

lr = LogisticRegression()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

accuracy_score(pred, y_test)
from imblearn.over_sampling import SMOTE

sm = SMOTE(kind = 'regular')

x_trai, y_trai = sm.fit_sample(x_train,y_train)

print(x_trai.shape)

print(y_trai.shape)
lre = LogisticRegression()

lre.fit(x_trai,y_trai)

pre = lre.predict(x_test)

from sklearn.metrics import classification_report

print(accuracy_score(pre, y_test))

print(confusion_matrix(pre, y_test))
from sklearn.feature_selection import RFE

log = LogisticRegression()

rfe = RFE(log,15)

rfe.fit(x_trai , y_trai)

rfe_features = list(telecom.columns[rfe.support_])

print('Features Identified by RFE \n', rfe_features)
x_trfe = pd.DataFrame(data = x_trai).iloc[:, rfe.support_]

y_trfe = y_trai

log = LogisticRegression()

log.fit(x_trfe, y_trfe)

x_testrfe = pd.DataFrame(data = x_test).iloc[:, rfe.support_]

predic = log.predict(x_testrfe)
print('Accuracy Score is :', accuracy_score(predic, y_test)*100)

print('confusion matrix \n', confusion_matrix(predic, y_test), '\n')

classification_report(predic, y_test)
x_train, x_test, y_train,y_test = train_test_split(X, y,test_size = 0.3,  random_state = 42)

from imblearn.over_sampling import SMOTE

smo = SMOTE()

x_tra, y_tra = smo.fit_sample(x_train, y_train)

print('x_tra shape ',x_tra.shape)

print('y_tra shape', y_tra.shape)
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(x_tra)

x_trapca = pca.fit_transform(x_tra)

print('x_trapca shape ' , x_trapca.shape)

x_testpca = pca.transform(x_test)

print('x_testpca shape', x_testpca.shape)
logi = LogisticRegression()

logi.fit(x_trapca, y_tra)

pdc = logi.predict(x_testpca)

print('Accuracy Score is', accuracy_score(pdc, y_test)*100)

print('Confusion Matrix \n', confusion_matrix(pdc, y_test))
pd.DataFrame({'pc1':pca.components_[0], 'pc2':pca.components_[1], 'pc3':pca.components_[2],'Features':list(telecom.columns)}).head(10)
plt.figure(figsize = (7,7))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('No of Components')

plt.ylabel('Cummulative Explained Varience')

plt.style.context('dark_background')

plt.show()
pc = PCA(n_components = 53)

x_trapc = pc.fit_transform(x_tra)

x_tespc = pc.transform(x_test)

print('x_trapc', x_trapc.shape)

print('x_tespc', x_tespc.shape)
logis = LogisticRegression()

logis.fit(x_trapc, y_tra)

predt = logis.predict(x_tespc)

print('Accuracy Score', accuracy_score(predt, y_test)*100)

print('Confusion Matrix \n', confusion_matrix(predt, y_test))
sns.heatmap(confusion_matrix(predt, y_test), annot = True)
x_train,x_test, y_train,y_test = train_test_split(X, y , test_size = 0.3, random_state = 42)

smot = SMOTE(kind = 'regular')

x_trsmot, y_trsmot = smot.fit_sample(x_train, y_train)

Pca = PCA(n_components = 50)

x_trpc = Pca.fit_transform(x_trsmot)

x_tspc = Pca.transform(x_test)

RF = RandomForestClassifier()

RF.fit(x_trpc, y_trsmot)

x_tspred = RF.predict(x_tspc)

print('Accuracy_score ', accuracy_score(x_tspred, y_test))

print('Confusion Matrix \n', confusion_matrix(x_tspred, y_test))
