#Import libraries
import numpy as np, pandas as pd, scipy.stats as sts, csv, matplotlib.pyplot as plt

#Read dataset
df = pd.read_csv("../input/new_client_data.csv", header=0)
#The Product Managers want to know about free and premium users, not unlimited or any others.
memberships_to_compare = ['free','premium']

#As well only for users that donwloaded the new_client
ctype_to_compare = ['new_client']

#Only to plot up to the lifespan=30
lifespans_to_plot = 30
#Get rid of any users that are not in the users we want to study
q1_df = df.loc[(df['client'].isin(ctype_to_compare)) & df['membership_type'].isin(memberships_to_compare)]
#Keeping the original dataframe to reuse it for further analysis
#Summarise the users dataframe by counting the users per lifespan and membership
q1_sum_df = q1_df.groupby(['lifespan','membership_type'])['user_id'].count().reset_index(level=(0,1))

q1_sum_df.columns = ['lifespan','membership_type','count_of_users']
#Pivot the prior table to graph the users on the membership_type, up to lifespan=30
pv_q1_df = q1_sum_df.pivot(index='lifespan', columns='membership_type', values='count_of_users').head(lifespans_to_plot+1)
#Build the stacked bar plot
pv_q1_df.plot.bar(stacked=True, figsize=(12,8))
#Show the plot
plt.show()
#Sort descending the summarised dataframe, then group by membership type and perform a cummulative sum. 
cum_df = q1_sum_df.sort_index(0, ascending=False)
cum_df['cum_sum'] = cum_df.groupby(['membership_type'])['count_of_users'].cumsum()

#Sort in an ascending manner
cum_df = cum_df.sort_index(0, ascending=True)
#Pivot the prior table to graph the users on the membership_type, up to lifespan=30
pv_q1_cum_df = cum_df.pivot(index='lifespan', columns='membership_type', values='cum_sum').fillna(value=0).head(lifespans_to_plot+1)
#Build the stacked bar plot
pv_q1_cum_df.plot.bar(stacked=True, figsize=(12,8))
#Show the plot
plt.show()
#Separate 2 dataframes 
fr_sum_df = q1_sum_df.loc[q1_sum_df['membership_type'].isin(['free'])]
prem_sum_df = q1_sum_df.loc[q1_sum_df['membership_type'].isin(['premium'])]
pv_fr_df = fr_sum_df.pivot(index='lifespan', columns='membership_type', values='count_of_users').fillna(value=0).head(lifespans_to_plot+1)
#Build the stacked bar plot
pv_fr_df.plot.bar(stacked=True, figsize=(12,8))
#Show the plot
plt.show()

prem_sum_df = prem_sum_df.pivot(index='lifespan', columns='membership_type', values='count_of_users').fillna(value=0).head(lifespans_to_plot+1)
#Build the stacked bar plot
prem_sum_df.plot.bar(stacked=True, figsize=(12,8))
#Show the plot
plt.show()
#Get the dataframe with all the data and split it into two with new_client and web_client
ls_newc_df = pd.DataFrame(columns=('count_all','count_1hub'))
ls_oldc_df = pd.DataFrame(columns=('count_all','count_1hub'))
#,'client'

#Get an auxiliar DF for users of the new client and users of the old client
fl_newc_df = df.loc[df['client'].isin(['new_client'])]
fl_oldc_df = df.loc[df['client'].isin(['web_client'])]

#Get the series for all the proportions per lifespan for all new client users
ls_newc_df['count_all'] = fl_newc_df.groupby('lifespan')['user_id'].count()
ls_newc_df['count_1hub'] = fl_newc_df.groupby('lifespan')['joined_at_least_1_hub'].count()
#Calculate the proportion
ls_newc_df['proportion'] = ls_newc_df['count_1hub']/ls_newc_df['count_all']

#Do the same as before for old client users
ls_oldc_df['count_all'] = fl_oldc_df.groupby('lifespan')['user_id'].count()
ls_oldc_df['count_1hub'] = fl_oldc_df.groupby('lifespan')['joined_at_least_1_hub'].count()
#Calculate the proportion
ls_oldc_df['proportion'] = ls_oldc_df['count_1hub']/ls_oldc_df['count_all']

#Convert both DFs to series for ease of use, and because we don't need the other columns
#Additionally the variable names are not amazing to say the least, that needs to improve

newc_ser = ls_newc_df['proportion']
webc_ser = ls_oldc_df['proportion']
#Count the total number per group (web_client and new_client)
total_webclient = fl_oldc_df['user_id'].count()
total_newclient = fl_newc_df['user_id'].count()

#We need to know the number of users in both groups 
print('New Client sample size: ' + str(total_newclient))
print('Web Client sample size: ' + str(total_webclient))
nwc_mean = newc_ser.mean()
wbc_mean = webc_ser.mean()

print('New Client mean: ' + str(nwc_mean))
print('Web Client mean: ' + str(wbc_mean))
nwc_var = newc_ser.var()
wbc_var = webc_ser.var()
nwc_std = newc_ser.std()
wbc_std = webc_ser.std()

print('New Client variance: ' + str(nwc_var))
print('Web Client variance: ' + str(wbc_var))
print('New Client variance: ' + str(nwc_std))
print('Web Client variance: ' + str(wbc_std))
#Perform the t-test with non-equal variances
tstat = sts.ttest_ind(newc_ser, webc_ser, equal_var=False)

print(tstat)