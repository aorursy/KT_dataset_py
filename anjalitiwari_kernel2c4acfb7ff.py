#Importing Libraries needed for this project



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as mt

from __future__ import division



# Importing the dataset

cookie_cats = pd.read_csv("C:/Users/VArun/Downloads/cookie_cats.csv")
#Checking what kind of data is stored

cookie_cats[0:10]
#Checking if any null data is present

cookie_cats.notnull().count()
#To check whether any userid is duplicate

cookie_cats.userid.nunique()
cookie_cats.info()
#Let's check for outliers in a numerical variable

import matplotlib.pyplot as plt

%matplotlib inline



plt.scatter(cookie_cats.userid,cookie_cats['sum_gamerounds'], c = "blue", marker = "s")

plt.figure()
#From the above figure it can be seen there is an outlier present in the data.

#The values of outliers gives us an incorrect information if used in analysis. It's better to get rid of it.

cookie_cats=cookie_cats[cookie_cats['sum_gamerounds']<40000]

plt.scatter(cookie_cats.userid,cookie_cats['sum_gamerounds'], c = "blue", marker = "s")

plt.figure()

# Summary Statistics for the each version 

cookie_cats.groupby('version')['sum_gamerounds'].agg(['sum','count','mean','min','max']).reset_index().rename(columns={'sum':'Total_Games_Played','count':'Total_Users', 'mean':'Average', 'min':'Min_value','max':'Max_value'})
# The sample size is almost same or the total number of users are nearly same

cookie_cats['version'].value_counts().plot(kind='bar', figsize=(4,4));
# Here we group the data as per the games played and the number of users who played it.

plot_gamerounds=cookie_cats.groupby('sum_gamerounds')['userid'].count().reset_index().rename(columns={'userid':'Total_users'})

plot_gamerounds.head(30)
# Now Plotting the same table in a graph but the total rounds played is 20.

ax = plot_gamerounds.head(20).plot(x='sum_gamerounds', y='Total_users')

ax.set_xlabel("Total Game Rounds Played")

ax.set_ylabel("Total Number of Users")

ax.set_title("Games played by a User")
# Now Plotting the same table in a graph but the total rounds played is 200.

ax = plot_gamerounds.head(200).plot(x='sum_gamerounds', y='Total_users')

ax.set_xlabel("Total Game Rounds Played")

ax.set_ylabel("Total Number of Users")

ax.set_title("Games played by a User")
# This graph checks the number of true and false values for column 'retention_1' for both the Control and Experiment group

cookie_cats.groupby('version')['retention_1'].value_counts().plot(kind='pie', figsize=(6,6))
# This graph checks the number of true and false values for column 'retention_7' for both the Control and Experiment group

cookie_cats.groupby('version')['retention_7'].value_counts().plot(kind='pie', figsize=(6,6))
#Lets look at the conversion rate for the day 1 retention for gate_30 i.e. Control Group

retention_day_1=cookie_cats.groupby('version')['retention_1'].sum() 

user_table_day1=cookie_cats.groupby('version')['userid'].count()

retention_gate_30_day1=round((retention_day_1['gate_30']/user_table_day1['gate_30']),4)

perc_retention_gate_30=round(retention_gate_30_day1*100,2)

print('The proportion of retention after day 1 for Control group is %s' %retention_gate_30_day1)

print('The percentage of retention after day 1 for Control group is %s%%' %perc_retention_gate_30)
retention_day_1
#Lets look at the conversion rate for the day 1 retention for gate_40 i.e. Experiment Group

retention_gate_40_day1=round((retention_day_1['gate_40']/user_table_day1['gate_40']),4)

perc_retention_gate_40=round(retention_gate_40_day1*100,2)

#perc_retention_gate_40

print('The proportion of retention after day 1 for Experiment group is %s' %retention_gate_40_day1)

print('The percentage of retention after day 1 for Experiment group is %s%%' %perc_retention_gate_40)
#Lets look at the conversion rate for the day 7 retention for gate_30 i.e. Control Group

retention_day_7=cookie_cats.groupby('version')['retention_7'].sum() 

user_table_day7=cookie_cats.groupby('version')['userid'].count()

retention_gate_30_day7=(retention_day_7['gate_30']/user_table_day7['gate_30'])

perc_retention_gate_30_day7=round(retention_gate_30_day7*100,2)

print('The proportion of retention after day 7 for Control group is %s' %retention_gate_30_day7)

print('The percentage of retention after day 7 for Control group is %s%%' %perc_retention_gate_30_day7)
#Lets look at the conversion rate for the day 1 retention for gate_40 i.e. Experiment Group

retention_gate_40_day7=(retention_day_7['gate_40']/user_table_day7['gate_40'])

perc_retention_gate_40_day7=round(retention_gate_40_day7*100,2)

#perc_retention_gate_40

print('The proportion of retention after day 7 for Experiment group is %s ' %retention_gate_40_day7)

print('The percentage of retention after day 7 for Experiment group is %s%%' %perc_retention_gate_40_day7)
retention_day_7
p_pool_day1 = (retention_day_1['gate_30'] + retention_day_1['gate_40'])/(user_table_day1['gate_30']+user_table_day1['gate_40'])

p_pool_day1

se_pool_day1=round(mt.sqrt(p_pool_day1*(1-p_pool_day1)*(1/user_table_day1['gate_30']+ 1/user_table_day1['gate_40'])),4)

se_pool_day1

# For 95% confidence interval the value of Z is 1.96 either we can use Z score table or we can use scipy package to calculate it

alpha=0.05

z=round(norm.ppf(1-alpha/2),2)

#Marginal Error calculation

Marginal_Error=round((z*se_pool_day1),4)

Marginal_Error
#The mean difference in the samples

p_difference = round((retention_gate_40_day1-retention_gate_30_day1),4)

p_difference
print ("The confidence interval is (%s, %s)" %(p_difference-Marginal_Error,p_difference+Marginal_Error))
t_statistic=p_difference-Marginal_Error

t_statistic
if t_statistic>0.05:

    print("Experiment performed better than control. It is statistically and practically significant")

elif t_statistic>0:

    print("Experiment performed better than control.It is statistically significant but not practically")

elif t_statistic<0:

    print("Control ia better than the Experiment. Hence, No need to make changes. ")
#cohen's d

d=round((p_difference/se_pool_day1),2)

d
p_pool_day7 = round((retention_day_7['gate_30'] + retention_day_7['gate_40'])/(user_table_day7['gate_30']+user_table_day7['gate_40']),4)

p_pool_day7
se_pool_day7=round(mt.sqrt(p_pool_day7*(1-p_pool_day7)*((1/user_table_day7['gate_30']) + (1/user_table_day7['gate_40']))),4)

se_pool_day7
# For 95% confidence interval the value of Z is 1.96

Marginal_Error_day7=z*se_pool_day7

Marginal_Error_day7
p_difference = round((retention_gate_40_day7-retention_gate_30_day7),4)

p_difference
t_statistic_7=p_difference-Marginal_Error_day7

t_statistic_7
print ("The confidence interval is" ,p_difference-Marginal_Error_day7 ,"and", p_difference+Marginal_Error_day7,)
if t_statistic>0.05:

    print("Experiment performed better than control. It is statistically and practically significant")

elif t_statistic>0:

    print("Experiment performed better than control.It is statistically singnificant but not practically")

elif t_statistic<0:

    print("Control ia better than the Experiment. Hence, No need to make changes ")
#Cohen's d

d=p_difference/se_pool_day7

d