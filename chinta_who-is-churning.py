# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
    
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
telecom_churn = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#count of online services availed
telecom_churn['Count_OnlineServices'] = (telecom_churn[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport',
       'StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)
plt.figure(figsize=(12,6))
ax = sns.countplot(x='Count_OnlineServices', hue='Churn', data=telecom_churn)
ax.set_title('Number of Services Availed Vs Churn', fontsize=20)
ax.set_ylabel('Number of Customers', fontsize=15)
ax.set_xlabel('Number of Online Services', fontsize=15)

#Finding : - Customers who does not avail any internet services are churning least,  Customers who are availing just one Online Service are churning highest. As the number of online services increases beyond one service, the less is the proportion of churn
agg = telecom_churn.replace('Yes',1).replace('No', 0).groupby('Count_OnlineServices', as_index=False)[['Churn']].mean()
agg[['Churn']] = np.round(agg[['Churn']], 2) * 100
plt.figure(figsize=(12,6))
ax = sns.barplot(x='Count_OnlineServices', y='Churn', data=agg)
ax.set_xlabel('Number of Online Services Availed', fontsize=15)
ax.set_ylabel('Percentage of Churn', fontsize=15)
ax.set_title('Number of Services Availed Vs Percentage of Churn', fontsize=20)
#Finding : - Customers who does not avail any internet services are churning least,  Customers who are availing just one Online Service are churning highest. As the number of online services increases beyond one service, the less is the proportion of churn
agg = telecom_churn.replace('Yes',1).replace('No', 0).groupby('Count_OnlineServices', as_index=False)[['MonthlyCharges']].mean()
agg[['MonthlyCharges']] = np.round(agg[['MonthlyCharges']], 0)

plt.figure(figsize=(12,6))
ax = sns.barplot(y='MonthlyCharges', x='Count_OnlineServices', data=agg)
ax.set_xlabel('Number of Online Services Availed', fontsize=15)
ax.set_ylabel('Average Monthly Charges',  fontsize=15)
ax.set_title('Avg Monthly Charges vs Number of Services', fontsize=20)
#Finding :- Customers who does not avail any internet service are paying just $33, while those with one service are paying double $66. As the number of services availed increases, the Average Monthly Charges are increasing linearly
sns.set()
agg = agg.div(agg.sum())
plt.figure(figsize=(12,6))
ax = sns.boxplot(x='Churn', y='MonthlyCharges', data=telecom_churn)
ax.set_title('Monthly Charges vs Churn', fontsize=20)
ax.set_ylabel('Monthly Charges', fontsize=15)
ax.set_xlabel('Churn', fontsize=15)
#Finding : - Higher the monthly charges, more is the possibility of Churn, non churners are paying just over $60, while churners are paying nearly $80 
telecom_churn_services = telecom_churn[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies'
                                       ,'TechSupport', 'StreamingTV', 'OnlineBackup', 'Churn']]
telecom_churn_services.replace(to_replace='Yes', value=1, inplace=True)
telecom_churn_services.replace(to_replace='No', value=0, inplace=True)
telecom_churn_services = telecom_churn_services[telecom_churn_services.OnlineSecurity !='No internet service']             
agg = telecom_churn_services.groupby('Churn', as_index=False)[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport', 'StreamingTV', 'OnlineBackup']].sum()
ax = agg.set_index('Churn').T.plot(kind='bar', stacked=True, figsize=(12,6))
patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='best')
ax.set_title('Which Service Customers Churn Higher', fontsize=20)
#Finding :- Customers who are availing Streaming Movies and StreamingTV are churning in higher proportions
plt.figure(figsize=(12,6))
ax = sns.boxplot(x='Churn', y = 'tenure', data=telecom_churn)
ax.set_title('Churn vs Tenure', fontsize=20)
ax.set_ylabel('Tenure', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)
#Finding :- Shorter the tenure, higher is the possibility of Churn
agg = telecom_churn.replace('Yes', 1).replace('No', 0).groupby('tenure', as_index=False)[['Churn']].mean()
agg = agg[agg.tenure < 25]
agg['Churn'] = np.round(agg['Churn'], 2) * 100

plt.figure(figsize=(20,6))

ax = sns.barplot(x='tenure', y='Churn', data = agg)
ax.set_title('Churn Percentage Over 24 Months of Tenure', fontsize=20)
ax.set_ylabel('Percentage of Churn', fontsize = 15)
ax.set_xlabel('Tenure in Months', fontsize = 15)
#Finding :- Over 60 percent of customers who complete one month of tenure Churn. As the length of tenure increases Churn reduces to about 25 percent at 24 months. 
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="Contract", data=telecom_churn);
ax.set_title('Contract Type vs Churn', fontsize=20)
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)
#Finding :- Customers with Month-to-Month contract are churning more, while two year contract customers are churning least 
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="PaymentMethod", data=telecom_churn);
ax.set_ylabel('Number of Customers Churned', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)
ax.set_title('Churn by Payment Method', fontsize=20)

#Finding :- Customers with Electronic Check as mode of payment are churning in higher proportion 
plt.figure(figsize=(12,6))
ax = sns.countplot(x="PaymentMethod", data=telecom_churn);

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Payment Method', fontsize = 15)

ax.set_title('Customers by Payment Method', fontsize=20)
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="MultipleLines", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)

ax.set_title('Churn by Multiple Lines', fontsize=20)
#Finding :- Surprisingly, customers with MultipleLines are churing in higher proportion
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="OnlineSecurity", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)

ax.set_title('Churn by Online Security Service', fontsize=20)
#Finding :- Customers who have not availed OnlineSecurity service are churning in higher proportion
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Contract", hue="OnlineSecurity", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Contract', fontsize = 15)
ax.set_title('Online Security Service by Contract Tenure', fontsize=20)
#Finding :- Month-to-Month customers are not availing OnlineSecurity, where as 50% of customers with One year Contract, that avail internet service also avail OnlineSecurity
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="SeniorCitizen", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)

ax.set_title('Churn Among Senior Citizens vs Non Senior (Absolute Values)', fontsize=20)
agg = telecom_churn.replace('Yes', 1).replace('No', 0).groupby('SeniorCitizen', as_index=False)[['Churn']].sum()
agg.iloc[0,1] = round(agg.iloc[0,1]/telecom_churn[telecom_churn.SeniorCitizen==0].shape[0], 2) * 100
agg.iloc[1,1] = round(agg.iloc[1,1]/telecom_churn[telecom_churn.SeniorCitizen==1].shape[0], 2) * 100
plt.figure(figsize=(12,6))
ax = sns.barplot(x='SeniorCitizen', y ='Churn', data = agg)
ax.set_ylabel('Percentage of Churn', fontsize=15)
ax.set_xlabel('Senior Citizen', fontsize=15)

ax.set_title('Churn Among Senior Citizens vs Non Senior (Percentage)', fontsize=20)

#Finding :- Senior Citizens are churning in greater proportion, almost 42% of Sr.Citizens churn compared to about 25% of non Sr.Citizens
plt.figure(figsize=(12,6))
ax = sns.countplot(x="SeniorCitizen", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize=15)
ax.set_xlabel('Senior Citizen', fontsize=15)

ax.set_title('Senior Citizen or Not', fontsize=20)
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="gender", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)

ax.set_title('Churn By Gender', fontsize=20)

#Finding:- Gender does not seem to influence Churn significantly
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="InternetService", data=telecom_churn);
ax.set_ylabel('Number of Customers', fontsize = 15)
ax.set_xlabel('Churn', fontsize = 15)
ax.set_title('Churn By Internet Service Type', fontsize=20)
#Finding:- Customers with Fiber Optic internet service are churning in alarming proportions


