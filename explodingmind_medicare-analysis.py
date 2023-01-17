import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore") #who needs those



df = pd.read_csv('../input/Medicare_Provider_Charge_Inpatient_DRGALL_FY2016.csv', thousands=',')
df.head()
df.tail()
print(df.dtypes)

df.columns = df.columns.str.replace(' ','_')



df.Average_Total_Payments = df.Average_Total_Payments.apply(lambda x: x.strip('$'))

df.Average_Total_Payments = df.Average_Total_Payments.apply(lambda x: x.replace(',', ""))

df.Average_Total_Payments = df.Average_Total_Payments.apply(pd.to_numeric, errors='coerce')



df.Average_Covered_Charges = df.Average_Covered_Charges.apply(lambda x: x.strip('$'))

df.Average_Covered_Charges = df.Average_Covered_Charges.apply(lambda x: x.replace(',', ""))

df.Average_Covered_Charges = df.Average_Covered_Charges.apply(pd.to_numeric, errors='coerce')



df.Average_Medicare_Payments = df.Average_Medicare_Payments.apply(lambda x: x.strip('$'))

df.Average_Medicare_Payments = df.Average_Medicare_Payments.apply(lambda x: x.replace(',', ""))

df.Average_Medicare_Payments = df.Average_Medicare_Payments.apply(pd.to_numeric, errors='coerce')



print("\n\n AFTER CONVERTING STRING TO NUMERIC \n\n")

print(df.dtypes)
des = df.describe()

#delete irrelevant variables

del des['Provider_Id']

del des['Provider_Zip_Code']

des
plt.rcParams['figure.figsize'] = [15, 15]



fig  = plt.figure()



sns.distplot(df['Average_Medicare_Payments'], kde=False, ax = fig.add_subplot(221))

sns.distplot(df['Average_Covered_Charges'], kde=False, ax = fig.add_subplot(222))

sns.distplot(df['Average_Total_Payments'], kde=False, ax = fig.add_subplot(223))

sns.distplot(df['Total_Discharges'], kde=False, ax = fig.add_subplot(224))

fig.suptitle("Quantitative Variable Distributions", fontsize=36)
fig2  = plt.figure()



sns.distplot(np.log(df['Average_Medicare_Payments']), kde=True, ax = fig2.add_subplot(221))

sns.distplot(np.log(df['Average_Covered_Charges']), kde=True, ax = fig2.add_subplot(222))

sns.distplot(np.log(df['Average_Total_Payments']), kde=True, ax = fig2.add_subplot(223))

sns.distplot(np.log(df['Total_Discharges']), kde=True, ax = fig2.add_subplot(224))

fig2.suptitle("Quantitative Variable Distributions after log transform", fontsize=36)
plt.rcParams['figure.figsize'] = [15, 15]

sns.countplot(df['Provider_State'], order = df['Provider_State'].value_counts().index, palette=sns.color_palette("GnBu_d"))
def sortByState(column, dataframe):

    groupby = dataframe.groupby('Provider_State', as_index=False).mean()

    sortedStates = groupby.sort_values(column)

    return sortedStates['Provider_State']
sns.barplot(x = 'Provider_State', y = 'Average_Total_Payments', data = df, order = sortByState('Average_Total_Payments', df), palette=sns.color_palette("Blues"))
sns.barplot(x = 'Provider_State', y = 'Average_Medicare_Payments', data = df, order = sortByState('Average_Medicare_Payments', df), palette=sns.color_palette("Greens"))
sns.barplot(x = 'Provider_State', y = 'Total_Discharges', data = df, order = sortByState('Total_Discharges', df))
gb = df.groupby('Provider_State')

j = 0

i = 0



plt.rcParams['figure.figsize'] = [25, 25]



for index, group in gb:

    

    if i % 8 == 0:

        ax = plt.subplot(2, 3, j % 6 + 1)

        j = j + 1

    

    sns.distplot(group['Average_Medicare_Payments'], hist=False, label = index, ax=ax)

    i = i + 1

plt.suptitle('Distribution of Average Medicare Payments', fontsize = 36) 

    

    
gb = df.groupby('Provider_State')

j = 0

i = 0



plt.rcParams['figure.figsize'] = [25, 25]



for index, group in gb:

    

    if i % 8 == 0:

        ax = plt.subplot(2, 3, j % 6 + 1)

        j = j + 1

    

    sns.distplot(np.log10(group['Average_Medicare_Payments']), hist=False, label = index, ax=ax)

    i = i + 1



plt.suptitle('Distribution of Average Medicare Payments', fontsize = 36) 
