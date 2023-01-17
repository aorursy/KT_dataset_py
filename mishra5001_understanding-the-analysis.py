# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import seaborn as sns



# Any results you write to the current directory are saved as output.
app_data = pd.read_csv("../input/application_data.csv")

prev_data = pd.read_csv("../input/previous_application.csv")
prev_data.head(10)
#checking the % of null values so that we can look at what attributes will drop!

print(round(100*(prev_data.isnull().sum()/len(prev_data))))
new_dropped_prev_data = prev_data.dropna(axis=1)

new_dropped_prev_data.head(10)
print(round(100*(new_dropped_prev_data.isnull().sum()/len(new_dropped_prev_data))))
new_dropped_prev_data.nunique()

# We are finding the unique values for each column as we want to find which parameters are affecting the loan repayments.
new_dropped_prev_data.shape
# looking at the number of requests recived for each Contract type and how much of them have been passed/rejected/unused or cancelled.

g = sns.barplot(x = 'NAME_CONTRACT_TYPE' , y=new_dropped_prev_data.index,hue = 'NAME_CONTRACT_STATUS',data = new_dropped_prev_data)

#g.set_yscale('log')
import matplotlib.pyplot as plt

plt.figure(figsize=(12,7))

ax = sns.boxplot(x = 'NAME_CONTRACT_TYPE' , y = 'AMT_APPLICATION',hue = 'NAME_CONTRACT_STATUS',data = new_dropped_prev_data)

ax.set_yscale('log')
# Next step is to remove the outliers, but before that, let's explore more attributes.

plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

plot_amount_weekday = sns.barplot(x = 'AMT_APPLICATION' , y = 'WEEKDAY_APPR_PROCESS_START',data = new_dropped_prev_data)

plt.title('Looking at the request of Loan Amount recieved on weekdays basis!')

plt.subplot(2,2,2)

plot_amount_status = sns.barplot(x = 'AMT_APPLICATION' , y = 'NAME_CONTRACT_STATUS',data = new_dropped_prev_data)

plt.title('Status of the requests for the variating Amount!')

plt.subplot(2,2,3)

plot_amount_client_type = sns.barplot(x = 'AMT_APPLICATION' , y = 'NAME_CLIENT_TYPE',data = new_dropped_prev_data)

plt.title('To find out the Client Type applying for how much amount of LOAN!')

plt.subplot(2,2,4)

plot_amount_contract_type = sns.barplot(x = 'AMT_APPLICATION' , y = 'NAME_CONTRACT_TYPE',data = new_dropped_prev_data)

plt.title('The Type of LOAN Request recieved and what has been the similar amount range of the requested Loan!');
plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

plot_client_type_hour_process_contract_type = sns.boxenplot(x = 'NAME_CONTRACT_TYPE' , y = 'HOUR_APPR_PROCESS_START',data = new_dropped_prev_data)

plt.title('To look for which hour duration interval, we have most REQUESTS of LOANS!')

plt.subplot(2,2,2)

plot_client_type_hour_process_contract_status = sns.boxenplot(x = 'NAME_CONTRACT_STATUS' , y = 'HOUR_APPR_PROCESS_START',data = new_dropped_prev_data)

plt.title('To look for which hour duration interval, we have the respective STATUS for applied LOANS!')

plt.subplot(2,2,3)

plot_client_type_hour_process_weekday = sns.boxenplot(x = 'WEEKDAY_APPR_PROCESS_START' , y = 'HOUR_APPR_PROCESS_START',data = new_dropped_prev_data)

plt.title('Looking for WEEKDAYS requests of Loans for respective HOUR Interval!')

plt.subplot(2,2,4)

plot_client_type_hour_process_client_type = sns.boxenplot(x = 'NAME_CLIENT_TYPE' , y = 'HOUR_APPR_PROCESS_START',data = new_dropped_prev_data)

plt.title('To look for which hour duration interval, what kind/type of client is applying for LOAN! ');

# This BARPLOT in SEABORN is focussed on the MEAN VALUES for the Amount!

plt.figure(figsize=(15,9))

plot_amount_hue_contract_type = sns.barplot(x = 'NAME_CONTRACT_STATUS', y ='AMT_APPLICATION',hue = 'NAME_CONTRACT_TYPE',data = new_dropped_prev_data)

plt.title('Looking at the Status of each Contract Type for the Variating Amount! This looks WOW!');

for i in plot_amount_hue_contract_type.patches:

    plot_amount_hue_contract_type.text(i.get_x(), i.get_height()+.2,str(round(i.get_height(),2)),fontsize = 11)
plt.figure(figsize = (20,12))

plt.subplot(2,1,1)

portfolio_status_type_amount = sns.barplot(x = 'NAME_PORTFOLIO', y='AMT_APPLICATION',hue='NAME_CONTRACT_STATUS',data = new_dropped_prev_data,ax=plt.gca())

plt.title('Trying to fetch outh the mean height for each Loan Portfolio and then the respective STATUS!')

for i in portfolio_status_type_amount.patches:

    portfolio_status_type_amount.text(i.get_x(), i.get_height()+.2,str(round(i.get_height(),2)),fontsize = 11)

    

plt.subplot(2,1,2)

portfolio_client_type_amount = sns.barplot(x = 'NAME_PORTFOLIO', y='AMT_APPLICATION',hue='NAME_CLIENT_TYPE',data = new_dropped_prev_data,ax=plt.gca())

plt.title('Trying to fetch outh the mean height for each Loan Portfolio and then the respective Client type!')

for i in portfolio_client_type_amount.patches:

    portfolio_client_type_amount.text(i.get_x(), i.get_height()+.2,str(round(i.get_height(),2)),fontsize = 11)
plt.figure(figsize = (20,12))

plt.subplot(2,1,1)

channel_plot = sns.barplot(x = 'CHANNEL_TYPE' , y = 'AMT_APPLICATION' , hue = 'NAME_CONTRACT_TYPE',data = new_dropped_prev_data)

#plt.xticks(rotation=40);

plt.title('Finding the Channels for the Distribution of Requests and which channel contributes most for the LOAN requests.!');

plt.subplot(2,1,2)

channel_plot_status = sns.barplot(x = 'CHANNEL_TYPE' , y = 'AMT_APPLICATION' , hue = 'NAME_CONTRACT_STATUS',data = new_dropped_prev_data)

plt.xticks(rotation=40);

l=[]

for i in channel_plot_status.patches:

    channel_plot_status.text(i.get_x(), i.get_height()+.2,str(round(i.get_height(),2)),fontsize = 11)

    l.append(channel_plot_status.text(i.get_x(), i.get_height()+.2,str(round(i.get_height(),2)),fontsize = 11))
# Let's look at the correlation. But first we need to fetch the important attributes and slice the data frame as per the need!

we_need = ['HOUR_APPR_PROCESS_START','AMT_APPLICATION','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']

df_we_need = new_dropped_prev_data.loc[:,we_need]

df_we_need.head(10)
cor = df_we_need.corr()

sns.heatmap(cor,annot=True);
plt.figure(figsize=(20,12))

sns.heatmap(new_dropped_prev_data.corr(),cmap='YlGnBu',annot=True);
app_data.head(10)
# lets check the missing values by percentage as usual

print(round(100*(app_data.isnull().sum()/len(app_data)),2))
#so let's drop the missing data as we can see that what we have to analyse

clean_app_data = app_data.dropna(axis=0)

clean_app_data.head(10)
clean_app_data.info()
clean_app_data.nunique()
docs_df = clean_app_data.iloc[:,-20:]

docs_df.head(10)

clean_app_data['All Docs'] = docs_df.sum(axis=1)

clean_app_data.head(10)
df_eda = clean_app_data.iloc[:,:20]

df_eda['Docs Submitted'] = clean_app_data['All Docs']

df_eda['Occupation type'] = clean_app_data['OCCUPATION_TYPE']

df_eda['Organization Type'] = clean_app_data['ORGANIZATION_TYPE']

df_eda.head(10)
print(df_eda.info())

print(df_eda.shape)

print(df_eda.isnull().sum())
# Now we will try to fetch the AMT _APPLICATION from the previous application data set

data_from_prev = new_dropped_prev_data.loc[:,['SK_ID_CURR','AMT_APPLICATION']]

merge_app_prev_df = df_eda.merge(data_from_prev,on = 'SK_ID_CURR')

merge_app_prev_df.head(10)
#Let's check the shape

print(merge_app_prev_df.shape)

print(merge_app_prev_df.isnull().sum())
merge_app_prev_df.head(5)
merge_app_prev_df.describe()
# Let's look at the Unique values

merge_app_prev_df.nunique()
# Let's look for gender wise data

sns.barplot(x = 'NAME_CONTRACT_TYPE' , y='AMT_APPLICATION',hue = 'CODE_GENDER',data = merge_app_prev_df);
# let's check for outliers in our Numerical values

df_outlier = merge_app_prev_df.loc[:,['AMT_APPLICATION','AMT_CREDIT','AMT_ANNUITY','AMT_INCOME_TOTAL','AMT_GOODS_PRICE']]

df_outlier.head(10)

#Let's plot

plt.figure(figsize=(20,5))

sns.boxenplot(data = df_outlier);

# we have outliers in AMT_INCOME_TOTAL and AMT_CREDIT
# Let's see what Occupation and Organization type from we recieve requests from of how much amount

plt.figure(figsize=(20,12))

sns.barplot(x = 'Occupation type' , y='AMT_APPLICATION',data = merge_app_prev_df);

plt.xticks(rotation=90,fontsize = 12);

plt.yticks(fontsize = 12);
# Let's see the Affect of Gender type for the Occupations

plt.figure(figsize=(20,12))

sns.barplot(x = 'Occupation type' , y='AMT_APPLICATION',hue = 'CODE_GENDER',data = merge_app_prev_df);

plt.xticks(rotation=90,fontsize = 12);

plt.yticks(fontsize = 12);
plt.figure(figsize=(16,7))

corr_merge = merge_app_prev_df.corr()

sns.heatmap(corr_merge,annot=True);