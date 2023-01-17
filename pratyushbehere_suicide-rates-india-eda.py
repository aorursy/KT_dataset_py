#!pip install pandas

import numpy as np

import pandas as pd

s_data=pd.read_csv("../input/Suicides in India 2001-2012.csv")

s_data.info() #looking at the data types

#Checking for missing data

#It looks like there are no null values in the dataset.

print("\n")

print(s_data.isnull().sum()) #
#Taking a look at the data

s_data
#Taking a look at the counts of 'Type_code'

s_data['Type_code'].value_counts()
s_data['State'].value_counts()

#The totals have been already included in the dataset, it makes sense to store it as a separate df.
#Keeping the totals in a separate df

totals=[item for item in s_data['State'] if "Total" in item]

totals=list(set(totals)) #To extract unique values in the list

s_totals=s_data[s_data['State'].isin(totals)] #storing a subsetted df in s_totals
#Checking if the overall values add up

pd.pivot_table(s_totals,index=['State','Type_code'],values='Total',aggfunc = np.sum)
#Checking if the overall values add up for the states

s_states=s_data[~s_data['State'].isin(totals)] #storing the state and UT data separately

pd.pivot_table(s_states,index=['State','Type_code'],values='Total',aggfunc=np.sum)
pd.pivot_table(s_states,index='Type_code',values='Total',aggfunc=np.sum)
#!pip install seaborn

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('dark_background')

#The data repeats itself in the df so we won't plot the entire data just one of the type_code should do.
df=s_totals[s_totals['State']=='Total (All India)'] #Storing All India Data separately

#Storing the totals for each year as a separate df

total_yearly=pd.pivot_table(df[df['Type_code']=='Education_Status'],values='Total',index='Year',aggfunc=np.sum)
#Plotting the total number of suicides per year in India

plt.figure(figsize=(11,9))

plt.plot(total_yearly)

plt.title("Total Number of Suicides in India (by year)", fontsize=20, y=1.01)

plt.xticks(total_yearly.index)

plt.tick_params(labelsize=13)

plt.xlabel("Year", fontsize=15, labelpad=15)

plt.show()
#Getting totals per year by gender

total_gender=pd.pivot_table(df[df['Type_code']=='Education_Status'],values='Total',index=['Year'],columns=['Gender'],aggfunc=np.sum)

plt.figure(figsize=(11,9))

plt.plot(total_gender['Male'], label="Male")

plt.plot(total_gender['Female'], label="Female")

plt.legend(loc='best')

plt.title("Total Number of Suicides in India (by year)", fontsize=20, y=1.01)

plt.xticks(total_gender.index)

plt.tick_params(labelsize=13)

plt.xlabel("Year", fontsize=15, labelpad=15)

plt.show()
total_edu_female_year=pd.pivot_table(df[(df['Type_code']=='Education_Status') & (df['Gender']=='Female')],values='Total',index=['Year'],columns=['Type'],aggfunc=np.sum)

total_edu_male_year=pd.pivot_table(df[(df['Type_code']=='Education_Status') & (df['Gender']=='Male')],values='Total',index=['Year'],columns=['Type'],aggfunc=np.sum)



total_edu_female=pd.pivot_table(df[(df['Type_code']=='Education_Status') & (df['Gender']=='Female')],values='Total',index=['Type'],aggfunc=np.sum)

total_edu_male=pd.pivot_table(df[(df['Type_code']=='Education_Status') & (df['Gender']=='Male')],values='Total',index=['Type'],aggfunc=np.sum)
#Sorting values in descending order

total_edu_male.sort_values(by='Total',ascending=False,inplace=True)

total_edu_female.sort_values(by='Total',ascending=False,inplace=True)

#Changing indices to make the graph look better

as_list = total_edu_male.index.tolist()

as_list[4] = 'Hr.Secondary'

as_list[3]='Secondary'

total_edu_male.index = as_list



as_list = total_edu_female.index.tolist()

as_list[4] = 'Hr.Secondary'

as_list[3]='Secondary'

total_edu_female.index = as_list

plt.figure(figsize=(13,34))



plt.subplot(2,1,1)

sns.barplot(x=total_edu_male.index,y=total_edu_male.Total)

plt.xticks(rotation='vertical')

plt.title("Total Number of Male Suicides in India (by education)", fontsize=20, y=1.01)

plt.tick_params(labelsize=13)



plt.subplot(2,1,2)

sns.barplot(x=total_edu_female.index,y=total_edu_female.Total)

plt.xticks(rotation='vertical')

plt.title("Total Number of Female Suicides in India (by education)", fontsize=20, y=1.01)

plt.tick_params(labelsize=13)



total_social_female=pd.pivot_table(df[(df['Type_code']=='Social_Status') & (df['Gender']=='Female')],values='Total',index=['Type'],aggfunc=np.sum)

total_social_male=pd.pivot_table(df[(df['Type_code']=='Social_Status') & (df['Gender']=='Male')],values='Total',index=['Type'],aggfunc=np.sum)
#Sorting values in descending order

total_social_male.sort_values(by='Total',ascending=False,inplace=True)

total_social_female.sort_values(by='Total',ascending=False,inplace=True)
plt.figure(figsize=(13,34))



plt.subplot(2,1,1)

sns.barplot(x=total_social_male.index,y=total_social_male.Total)

plt.xticks(rotation='vertical')

plt.title("Total Number of Male Suicides in India (by social status)", fontsize=20, y=1.01)

plt.tick_params(labelsize=13)



plt.subplot(2,1,2)

sns.barplot(x=total_social_female.index,y=total_social_female.Total)

plt.xticks(rotation='vertical')

plt.title("Total Number of Female Suicides in India (by social status)", fontsize=20, y=1.01)

plt.tick_params(labelsize=13)

#Let's list the different type codes we have in the data

s_states['Type_code'].value_counts()
#Let's first remove the age group=0-100 from the dataset since it will double count the data once 

#we add all the totals up

#Keeping the totals in a separate df

rm=[item for item in s_states['Age_group'] if "0-100" in item]

rm=list(set(rm)) #To extract unique values in the list

s_states=s_states[~s_states['Age_group'].isin(rm)] #storing a subsetted df in s_totals

#Taking a look at the causes. 

#It seems that a bit of cleaning is required in this case, the last two types are basically the same. 

s_states[s_states['Type_code']=="Causes"].Type.value_counts()
#Storing the causes as a pivot_table

s_states_causes=pd.pivot_table(s_states[s_states['Type_code']=="Causes"],index='Type',values='Total',aggfunc=np.sum)

s_states_causes.sort_values(by='Total',ascending=False,inplace=True)

s_states_causes #Taking a look at the leading causes for suicide
#Storing the professional_profile as a pivot_table

s_states_prof=pd.pivot_table(s_states[s_states['Type_code']=="Professional_Profile"],index='Type',values='Total',aggfunc=np.sum)

s_states_prof.sort_values(by='Total',ascending=False,inplace=True)

s_states_prof #Taking a look at the leading causes for suicide
#Plotting causes

plt.figure(figsize=(15,15))

sns.barplot(y=s_states_causes.index,x=s_states_causes.Total)

plt.xticks(rotation='vertical')

plt.title("Causes for Suicides in India", fontsize=20, y=1.01)

plt.tick_params(labelsize=14)
#Plotting professional profile

plt.figure(figsize=(10,10))

sns.barplot(y=s_states_prof.index,x=s_states_prof.Total)

plt.xticks(rotation='vertical')

plt.title("Suicides in India (by Profession)", fontsize=20, y=1.01)

plt.tick_params(labelsize=14)
#Let's take a look at the states where most suicides occur - 

s_states_totals=pd.pivot_table(s_states[s_states['Type_code']=="Causes"],index='State',values='Total',aggfunc=np.sum)

s_states_totals.sort_values(by='Total',ascending=False,inplace=True)

s_states_totals