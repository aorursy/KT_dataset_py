# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
campaign_desc = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/campaign_desc.csv')
product = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/product.csv')
coupon_redempt = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/coupon_redempt.csv')
transaction_data = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/transaction_data.csv')
hh_demographic = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/hh_demographic.csv')
coupon = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/coupon.csv')
campaign_table = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/campaign_table.csv')
causal_data = pd.read_csv('/kaggle/input/dunnhumby-the-complete-journey/causal_data.csv')
#  Get a list of all unique household keys in transaction_data
trans_data_hh_key_list = list(transaction_data.household_key.unique())

#  Get a list of all unique household keys in hh_demoographics
hh_demo_hh_key_list = list(hh_demographic.household_key.unique())
## A function to find common items in two lists

def common_member(a, b): 
    """A function to find the common items in two lists."""
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return list((a_set & b_set))
    else: 
        print("No common elements")
#  Create a list of unique household keys that transaction data AND demographic data is avaliable for.
trans_data_and_hh_demo_hh_keys_list = common_member(trans_data_hh_key_list, hh_demo_hh_key_list)

#  Print the length of the list so we know the amount of household we have avaliable data for
print("We have transaction AND demongraphic data on " + str(len(trans_data_and_hh_demo_hh_keys_list)) + " households.")
transaction_data.head()
hh_demographic.head()
trans_data_baskets_hh_key = pd.DataFrame(transaction_data.groupby(['household_key', 'BASKET_ID']).sum()).drop(['DAY', 'PRODUCT_ID','QUANTITY', 'STORE_ID', 'RETAIL_DISC', 'TRANS_TIME', 'WEEK_NO', 'COUPON_DISC', 'COUPON_MATCH_DISC'], axis=1)
trans_data_baskets_hh_key = trans_data_baskets_hh_key.merge(transaction_data.drop(['SALES_VALUE', 'PRODUCT_ID', 'QUANTITY', 'RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC'], axis=1), on="BASKET_ID").drop_duplicates(subset=['BASKET_ID'])
trans_data_baskets_hh_key.head()
spend_demo_data = trans_data_baskets_hh_key.merge(hh_demographic, on='household_key')
spend_demo_data.head()
print(sorted(trans_data_and_hh_demo_hh_keys_list) == sorted(list(spend_demo_data.household_key.unique())))

# Print the length of the list of household keys to verify the length matches as well. 
len(list(spend_demo_data.household_key.unique()))
#  Get the unique age_groups in AGE_DSC
age_groups = sorted(list(spend_demo_data.AGE_DESC.unique()))
print("The different age_group buckets in the dataset are " + str(age_groups) + ".")

# Create empty list of to store dataframes
age_group_spending_dfs = []

# append each individual age group dataframe to the list
for group in age_groups:
    age_group_spending_dfs.append(spend_demo_data[spend_demo_data['AGE_DESC'] == group])
    
    
# Group and Name the Transaction Spending DataFrames by Age Group
age_19_24 = age_group_spending_dfs[0]
age_25_34 = age_group_spending_dfs[1]
age_35_44 = age_group_spending_dfs[2]
age_45_54 = age_group_spending_dfs[3]
age_55_64 = age_group_spending_dfs[4]
age_65 = age_group_spending_dfs[5]
#  Get the unique age_groups in INCOME_DESC
income_groups = sorted(list(spend_demo_data.INCOME_DESC.unique()))
print("The different income buckets in the dataset are " + str(income_groups) + ".")

# Create empty list of to store dataframes
income_group_spending_dfs = []

# append each individual age group dataframe to the list
for group in income_groups:
    income_group_spending_dfs.append(spend_demo_data[spend_demo_data['INCOME_DESC'] == group])
    

#  Group and Name the Transaction Spending DataFrames by Income Group
inc_100_124K = income_group_spending_dfs[0]
inc_125_149K = income_group_spending_dfs[1]
inc_15_24K = income_group_spending_dfs[2]
inc_150_174K = income_group_spending_dfs[3]
inc_175_199K = income_group_spending_dfs[4]
inc_200_249K = income_group_spending_dfs[5]
inc_25_34K = income_group_spending_dfs[6]
inc_250K = income_group_spending_dfs[7]
inc_35_49K = income_group_spending_dfs[8]
inc_50_74K = income_group_spending_dfs[9]
inc_75_99K = income_group_spending_dfs[10]
inc_Under_15K = income_group_spending_dfs[11]
sns.set_style("darkgrid")

avg_weekly_sales_19_24 = age_19_24[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Age Group 19-24", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_19_24,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_25_34 = age_25_34[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Age Group 25-34", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_25_34,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_35_44 = age_35_44[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Age Group 35-44", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_35_44,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_45_54 = age_45_54[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Age Group 45-54", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_45_54,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_55_64 = age_55_64[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Age Group 55-64", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_55_64,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_65 = age_65[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Age Group 65+", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_65,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_Under_15K = inc_Under_15K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income Under 15K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_Under_15K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_15_24K = inc_15_24K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 15-24K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_15_24K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_25_34K = inc_25_34K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 25-34K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_25_34K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_35_49K = inc_35_49K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 35-49K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_35_49K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_50_74K = inc_50_74K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 50-74K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_50_74K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_75_99K = inc_75_99K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 75-99K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_75_99K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_100_124K = inc_100_124K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 100-124K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_100_124K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_125_149K = inc_125_149K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 125-149K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_125_149K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_150_174K = inc_150_174K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 150-175K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_150_174K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_175_199K = inc_175_199K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income Under 175-199K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_175_199K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_200_249K = inc_200_249K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 200-249K", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_200_249K,
                  markers=True)
sns.set_style("darkgrid")

avg_weekly_sales_inc_250K = inc_250K[['WEEK_NO', 'SALES_VALUE', 'household_key']].groupby(['WEEK_NO', 'household_key']).mean().reset_index()
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value by Week for Income 250K+", size=26)
ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=avg_weekly_sales_inc_250K,
                  markers=True)
print(sorted(inc_50_74K[inc_50_74K['AGE_DESC'] == '25-34'].household_key.unique()) == sorted(age_25_34[age_25_34['INCOME_DESC'] == '50-74K'].household_key.unique()))

hh_keys_age_25_34_inc_50_74K = list(age_25_34[age_25_34['INCOME_DESC'] == '50-74K'].household_key.unique())

target_demo_1 = spend_demo_data[spend_demo_data['household_key'].isin(hh_keys_age_25_34_inc_50_74K)]
target_demo_1.head()
campaign_desc['START_WEEK'] = campaign_desc['START_DAY'] / 7
campaign_desc['END_WEEK'] = campaign_desc['END_DAY'] / 7
campaign_desc.head()
#  Find the campaigns that reached our target demographic
demo_campaigns = campaign_table[campaign_table['household_key'].isin(hh_keys_age_25_34_inc_50_74K)]

#  Find the top three campaigns that reached households in our list
demo_campaigns.CAMPAIGN.value_counts().head(3)
campaign_18 = campaign_desc[campaign_desc['CAMPAIGN'] == 18]
campaign_13 = campaign_desc[campaign_desc['CAMPAIGN'] == 13]
campaign_8 = campaign_desc[campaign_desc['CAMPAIGN'] == 8]
plt.figure(figsize=(20,10))
plt.title("Average Transaction Basket Sales Value for Age Group 25-34 for Income Group 50-74K During Top 5 Campaigns", size=26)

ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=target_demo_1,
                  markers=True)

campaign_18_active = np.arange(int(campaign_18['START_WEEK']), int(campaign_18['END_WEEK']))
campaign_13_active = np.arange(int(campaign_13['START_WEEK']), int(campaign_13['END_WEEK']))
campaign_8_active = np.arange(int(campaign_8['START_WEEK']), int(campaign_8['END_WEEK']))
y = target_demo_1.SALES_VALUE.max()

ax.fill_between(campaign_18_active, y, facecolor='purple', alpha=0.3, label='Ad Campaign 18')
ax.fill_between(campaign_13_active, y, facecolor='green', alpha=0.3, label='Ad Campaign 13')
ax.fill_between(campaign_8_active, y, facecolor='yellow', alpha=0.3, label='Ad Campaign 8')

ax.set_xlabel('Week', fontsize=22)
ax.set_ylabel('Sales Value ($)', fontsize=22)
ax.set_ylim(0,100)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.legend()
print(hh_keys_age_25_34_inc_50_74K[:2])
hh_key_166 = target_demo_1[target_demo_1['household_key'] == 166]
hh_key_256 = target_demo_1[target_demo_1['household_key'] == 256]
hh_key_166 = target_demo_1[target_demo_1['household_key'] == 166]
campaign_table[campaign_table['household_key'] == 166]
plt.figure(figsize=(20,10))
plt.title("Weekly Average Spending For Household 166", size=26)

ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=hh_key_166,
                  markers=True)

ax.set_xlabel('Week', fontsize=22)
ax.set_ylabel('Sales Value ($)', fontsize=22)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.legend()
hh_key_256 = target_demo_1[target_demo_1['household_key'] == 256]
campaign_table[campaign_table['household_key'] == 256]
plt.figure(figsize=(20,10))
plt.title("Weekly Average Spending For Household 256", size=26)

ax.fill_between(campaign_13_active, y, facecolor='green', alpha=0.3, label='Ad Campaign 13')
ax.fill_between(campaign_8_active, y, facecolor='yellow', alpha=0.3, label='Ad Campaign 8')
y = hh_key_166.SALES_VALUE.max()

ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=hh_key_256,
                  markers=True)

ax.set_xlabel('Week', fontsize=22)
ax.set_ylabel('Sales Value ($)', fontsize=22)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.legend()
camp_18_hh_keys = list(campaign_table[campaign_table['CAMPAIGN'] == 18].household_key.unique())
print("There were "+ str(len(camp_18_hh_keys)) + " households reached by campaign 18.")
camp_18_spend_demo_data = spend_demo_data[spend_demo_data['household_key'].isin(camp_18_hh_keys)]
plt.figure(figsize=(20,10))
plt.title("Weekly Average Spending For Households Reached by Campaign 18", size=26)

ax = sns.lineplot(x='WEEK_NO', y='SALES_VALUE',
                  data=camp_18_spend_demo_data,
                  markers=True)

campaign_18_active = np.arange(int(campaign_18['START_WEEK']), int(campaign_18['END_WEEK']))
y = 65
ax.fill_between(campaign_18_active, y, facecolor='purple', alpha=0.3, label='Ad Campaign 18')

ax.set_xlabel('Week', fontsize=22)
ax.set_ylabel('Sales Value ($)', fontsize=22)
ax.set_ylim(20,65)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.legend()
labels = camp_18_spend_demo_data.AGE_DESC.value_counts().index
sizes = camp_18_spend_demo_data.AGE_DESC.value_counts()

fig, ax = plt.subplots(figsize=(10,10))
plt.style.use('fivethirtyeight')
plt.pie(camp_18_spend_demo_data.AGE_DESC.value_counts(), labels=labels, autopct='%1.1f%%')
plt.title("Age Groups Reached by Campaign 18")
labels = camp_18_spend_demo_data.INCOME_DESC.value_counts().index
sizes = camp_18_spend_demo_data.INCOME_DESC.value_counts()

fig, ax = plt.subplots(figsize=(10,10))
plt.style.use('fivethirtyeight')
plt.pie(camp_18_spend_demo_data.INCOME_DESC.value_counts(), labels=labels, autopct='%1.1f%%')
plt.title("Income Groups Reached by Campaign 18")