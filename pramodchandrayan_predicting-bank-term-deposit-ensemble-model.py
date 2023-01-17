import pandas as pd

import numpy as np

from pandas import read_csv

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')

import os

dataframe = pd.read_csv("../input/bank-full.csv")

dataframe.head(45000)



#Shape Of The Data, To See Total Row & Column Count



dataframe.shape



#Let's Get The Detail Data Info To Understand Each Independent Attribute, it's data type and its meaning

dataframe.info()
dataframe.describe().T #using data tarnspose to have better view of each data. 



#dataframe.describe(include = 'all').T
# Step 1: Delete the rows in the 'poutcome' columns where values  is 'others' as it is not helping in any inferences



del_condition = dataframe.poutcome == 'others'

dataframe1 = dataframe.drop(dataframe[del_condition].index, axis = 0, inplace = False)

print("New dataframe 1 is :", dataframe1['poutcome'].value_counts())


%matplotlib inline



# Let's see how the numeric data is distributed.



#dataframe['duration'] = dataframe['duration'].apply(lambda n:n/60).round(2)



#Change the unit of 'duration' from seconds to minutes



newdf = dataframe.copy()



newdf['duration'] = newdf['duration'].apply(lambda n:n/60).round(2)



plt.style.use('seaborn-whitegrid')



newdf.hist(bins=20, figsize=(15,10), color='lightblue', edgecolor = 'black')

plt.show()





#print("Descriptive stats of age",dataframe1['age'].describe())

print("Descriptive stats of duration",newdf['duration'].describe())

print("Descriptive stats of campaign",newdf['campaign'].describe())

print("Descriptive stats of day", newdf['day'].describe())

print("Descriptive stats of no of day past the capaign was last done: ", newdf['pdays'].describe())
plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.boxplot(x= newdf.age, color='lightblue')



plt.subplot(3,3,2)

sns.boxplot(x= newdf.balance, color='lightblue')



plt.subplot(3,3,3)

sns.boxplot(x= newdf.day, color='lightblue')



plt.show()
plt.figure(figsize= (20,15))

plt.subplot(4,4,1)

sns.boxplot(x= newdf.duration, color='lightblue')



plt.subplot(4,4,2)

sns.boxplot(x= newdf.campaign, color='lightblue')



plt.subplot(4,4,3)

sns.boxplot(x= newdf.pdays, color='lightblue')



plt.subplot(4,4,4)

sns.boxplot(x= newdf.previous, color='lightblue')



plt.show()
#SKEWNESS



from scipy.stats import zscore

import scipy.stats as stats



#Let's check Skew in all numercial attributes





Skewness = pd.DataFrame({'Skewness' : [stats.skew(dataframe1.age),stats.skew(dataframe1.day),stats.skew(dataframe1.balance),stats.skew(dataframe1.duration),stats.skew(dataframe1.campaign),stats.skew(dataframe1.pdays),stats.skew(dataframe1.previous) ]},

                        index=['age','day','balance', 'duration', 'campaign', 'pdays', 'previous'])  # Measure the skeweness of the required columns

Skewness 



#Removing outliers in balance data using zscore:



from scipy.stats import zscore



newdf[['balance']].mean()

newdf[['balance']].mean()



newdf['balance_outliers'] = newdf['balance']

newdf['balance_outliers']= zscore(newdf['balance_outliers'])



condition1 = (newdf['balance_outliers']>3) | (newdf['balance_outliers']<-3 )

newdf1 = newdf.drop(newdf[condition1].index, axis = 0, inplace = False)

newdf2 = newdf1.drop('balance_outliers', axis=1)



#original one 

plt.figure(figsize= (20,15))

sns.boxplot(x= newdf.balance, color='lightblue')



#After outlier treatment using z score



plt.figure(figsize= (20,15))

sns.boxplot(x= newdf2.balance, color='lightblue')

print("We managed to get rid to some extreme outlier shown below. ")



objdf = newdf.select_dtypes(include ='object') 

objdf.head(5)





plt.figure(figsize=(35,30))



#Job category



x = newdf.job.value_counts().index    #Values for x-axis

print("job count distribution : ", newdf.job.value_counts())

y = [newdf['job'].value_counts()[i] for i in x]   # Count of each class on y-axis



plt.subplot(6,2,1)

plt.bar(x,y, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Job Type')

plt.ylabel('Count ')

plt.title('Job Type Distribution')





#Marital Status 

x1 = newdf.marital.value_counts().index    #Values for x-axis

y1 = [newdf['marital'].value_counts()[j] for j in x1]   # Count of each class on y-axis



print("\nx1 marital attribute count: ", newdf.marital.value_counts())

plt.subplot(6,2,2)

plt.bar(x1,y1, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Marital ')

plt.ylabel('Count')

plt.title('Marital Status distribution')



#education level



x2 = newdf.education.value_counts().index    #Values for x-axis

y2 = [newdf['education'].value_counts()[k] for k in x2]   # Count of each class on y-axis



print("\nx2 education level count distribution: ", newdf.education.value_counts())



plt.subplot(6,2,3)

plt.bar(x2,y2, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('education')

plt.ylabel('Count ')

plt.title("education' distribution")



#credit defaulter or not?



x3 = newdf.default.value_counts().index    #Values for x-axis

y3 = [newdf['default'].value_counts()[l] for l in x3]   # Count of each class on y-axis



print("\nx3 Credit default count distribution: ", newdf.default.value_counts())

plt.subplot(6,2,4)

plt.bar(x3,y3, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Credit Default ?')

plt.ylabel('Count ')

plt.title("Credit Default Distribution")



#housing loan availed or not 



x4 = newdf.housing.value_counts().index    #Values for x-axis

y4 = [newdf['housing'].value_counts()[m] for m in x4]   # Count of each class on y-axis



print("\nx4 housing loan count distribution: ", newdf.housing.value_counts())

plt.subplot(6,2,5)

plt.bar(x4,y4, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('housing Loan ?')

plt.ylabel('Count ')

plt.title("Housing Loan Distribution")



#Personal Loan

x5 = newdf.loan.value_counts().index    #Values for x-axis

y5 = [newdf['loan'].value_counts()[n] for n in x5]   # Count of each class on y-axis

print("\nPersonal loan count distribution: ", newdf.loan.value_counts())

plt.subplot(6,2,6)

plt.bar(x5,y5, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Personal Loan ?')

plt.ylabel('Count ')

plt.title("Personal Loan Distribution")

plt.show()


plt.figure(figsize=(20,25))



#Mode of communication with customers

x6 = newdf.contact.value_counts().index    #Values for x-axis

y6 = [newdf['contact'].value_counts()[o] for o in x6]   # Count of each class on y-axis



print("\nDistribution Of Mode Of Communication With Customers: ", newdf.contact.value_counts())



plt.subplot(4,2,1)

plt.bar(x6,y6, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Contact Type  ?')

plt.ylabel('Count ')

plt.title("Contact Type Distribution")



#communication result



x7 = newdf.poutcome.value_counts().index    #Values for x-axis

y7 = [newdf['poutcome'].value_counts()[p] for p in x7]   # Count of each class on y-axis



print("\nDistribution Of communication result: ", newdf.poutcome.value_counts())



plt.subplot(4,2,2)

plt.bar(x7,y7, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Phone Call Outcome  ?')

plt.ylabel('Count ')

plt.title("Phone Call Outcome Distribution")



#month when customer was last contacted



x8 = newdf.month.value_counts().index    #Values for x-axis

y8 = [newdf['month'].value_counts()[q] for q in x8]   # Count of each class on y-axis



print("\nDistribution Of monthly customer contact detail : ", newdf.month.value_counts())

plt.subplot(4,2,3)

plt.bar(x8,y8, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Month Contacted  ?')

plt.ylabel('Count ')

plt.title("Month Contacted Distribution")





#FD status Which is our target variable



x9 = newdf.Target.value_counts().index    #Values for x-axis

y9 = [newdf['Target'].value_counts()[r] for r in x9]   # Count of each class on y-axis



print("\nDistribution Of customer W.R.T FD : ", newdf.Target.value_counts())

plt.subplot(4,2,4)

plt.bar(x9,y9, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('FD Status ?')

plt.ylabel('Count ')

plt.title(" FD status Distribution")



plt.show()

# 1. FD Subscribers Age Distribution: 



#But First Let's Craete Age Grouping: 



lst = [newdf2]

for column in lst:

    column.loc[column["age"] < 30,  'age_group'] = 20

    column.loc[(column["age"] >= 30) & (column["age"] <= 39), 'age_group'] = 30

    column.loc[(column["age"] >= 40) & (column["age"] <= 49), 'age_group'] = 40

    column.loc[(column["age"] >= 50) & (column["age"] <= 59), 'age_group'] = 50

    column.loc[column["age"] >= 60, 'age_group'] = 60

    

    count_age_response_pct = pd.crosstab(newdf2['Target'],newdf2['age_group']).apply(lambda x: x/x.sum() * 100)

    

count_age_response_pct = count_age_response_pct.transpose() 



age = pd.DataFrame(newdf2['age_group'].value_counts())

age['% Contacted'] = age['age_group']*100/age['age_group'].sum()

age['% FD Subscription'] = count_age_response_pct['yes']

age.drop('age_group',axis = 1,inplace = True)



age['age'] = [30,40,50,20,60]

age = age.sort_values('age',ascending = True)





plt.figure(figsize=(20,10))

sns.countplot(newdf2['age_group'], hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their job profile





plt.figure(figsize=(20,10))

sns.countplot(newdf2.job, hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their Marital Status





plt.figure(figsize=(20,10))

sns.countplot(newdf2.marital, hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their education level





plt.figure(figsize=(20,10))

sns.countplot(newdf2.education, hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their Personal Loan Status





plt.figure(figsize=(20,10))

sns.countplot(newdf2.loan, hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their Housing Loan Status





plt.figure(figsize=(20,10))

sns.countplot(newdf2.housing, hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their Credit Default  Status





plt.figure(figsize=(20,10))

sns.countplot(newdf2.default, hue = "Target", data=newdf2)

plt.tight_layout()
# Let us see the count of customers who subscribed to FD based on their mode of client communication





plt.figure(figsize=(20,10))

sns.countplot(newdf2.contact, hue = "Target", data=newdf2)

plt.tight_layout()



# Let us see the count of customers who subscribed to FD based on their last month of contact





plt.figure(figsize=(20,10))

sns.countplot(newdf2.month, hue = "Target", data=newdf2)

plt.tight_layout()

# Let us see the count of customers who subscribed to FD based on frequency of customr contact during the campaign





plt.figure(figsize=(20,10))

sns.countplot(newdf2.campaign, hue = "Target", data=newdf2)

plt.tight_layout()


scatter_age_balance = newdf2.plot.scatter('age','balance',figsize = (7,5))



plt.title('The Relationship between Age and Balance ')

plt.show()



#2. Duration &  Campaign: 

plt.figure(figsize=(8,6))

sns.scatterplot(newdf2.duration, newdf2.campaign, palette= ['pink','lightblue'] )

plt.show()


#imapct of job type, balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax1 = fig.add_subplot(221)

ax2= fig.add_subplot(221)

ax1 = sns.boxplot(newdf2['education'], newdf2['balance'], data=newdf2, ax =ax1)

#ax2 = sns.boxplot(newdf2['age'], newdf2['balance'], data=newdf2, ax =ax2)



#Before We measure the impacts of predcitor variable on our target let's have quick aggregation view using groupby

print(newdf2.groupby('Target').mean())

print(newdf2.groupby('Target').median())

#Age & Target Variable

#Let's see of how age impacts the people's decision to take fd. 



#But First Let's Craete Age Grouping: 



lst = [newdf2]

for column in lst:

    column.loc[column["age"] < 30,  'age_group'] = 20

    column.loc[(column["age"] >= 30) & (column["age"] <= 39), 'age_group'] = 30

    column.loc[(column["age"] >= 40) & (column["age"] <= 49), 'age_group'] = 40

    column.loc[(column["age"] >= 50) & (column["age"] <= 59), 'age_group'] = 50

    column.loc[column["age"] >= 60, 'age_group'] = 60

    

    count_age_response_pct = pd.crosstab(newdf2['Target'],newdf2['age_group']).apply(lambda x: x/x.sum() * 100)

    

count_age_response_pct = count_age_response_pct.transpose() 

print(count_age_response_pct)



age = pd.DataFrame(newdf2['age_group'].value_counts())

age['% Contacted'] = age['age_group']*100/age['age_group'].sum()

age['% FD Subscription'] = count_age_response_pct['yes']

age.drop('age_group',axis = 1,inplace = True)



age['age'] = [30,40,50,20,60]

age = age.sort_values('age',ascending = True)



plot_age = age[['% FD Subscription','% Contacted']].plot(kind = 'bar', figsize=(8,6), color = ('green','orange'))

plt.xlabel('Age Group')

plt.ylabel('Subscription Rate')

plt.xticks(np.arange(5), ('<30', '30-39', '40-49', '50-59', '60+'),rotation = 'horizontal')

plt.title('Subscription vs. Contact Rate by Age')

plt.show()





#ax = sns.boxplot(dataframe['Target'], newdf2['age_group'], data=newdf2)

    

    

    


#Let's Seggregate The Balance  & Perform Transaformation To Have Better Insights



lst = [newdf2]

for column in lst:

    column.loc[column["balance"] <= 0,  'balance_group'] = 'no balance'

    column.loc[(column["balance"] > 0) & (column["balance"] <= 1000), 'balance_group'] = 'low balance'

    column.loc[(column["balance"] > 1000) & (column["balance"] <= 5000), 'balance_group'] = 'average balance'

    column.loc[(column["balance"] > 5000), 'balance_group'] = 'high balance'

    

    

count_balance_response_pct = pd.crosstab(newdf2['Target'],newdf2['balance_group']).apply(lambda x: x/x.sum() * 100)

count_balance_response_pct = count_balance_response_pct.transpose()



bal = pd.DataFrame(newdf2['balance_group'].value_counts())

bal['% Contacted'] = bal['balance_group']*100/bal['balance_group'].sum()

bal['% Subscription'] = count_balance_response_pct['yes']

bal.drop('balance_group',axis = 1,inplace = True)



bal['bal'] = [1,2,0,3]

bal = bal.sort_values('bal',ascending = True)



plot_balance = bal[['% Subscription','% Contacted']].plot(kind = 'bar',

                                               color = ('royalblue','green'),

                                               figsize = (8,6))



plt.title('Subscription vs Contact Rate by Balance Level')

plt.ylabel('Subscription Rate')

plt.xlabel('Balance Category')

plt.xticks(rotation = 'horizontal')



# label the bar

for rec, label in zip(plot_balance.patches, bal['% Subscription'].round(1).astype(str)):

    plot_balance.text(rec.get_x() + rec.get_width()/2, rec.get_height() + 1,  label+'%',   ha = 'center', color = 'black')



print(count_balance_response_pct)



#ax = sns.boxplot(newdf2['Target'], newdf2['balance'], data=newdf2)
#First Let's Transform The Data, So that we can plot them meaningfully



count_job_target_pct = pd.crosstab(newdf2['Target'],newdf2['job']).apply(lambda x: x/x.sum() * 100)

count_job_target_pct = count_job_target_pct.transpose()



plot_job = count_job_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Job')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Job Category')



# Label each bar

for rec, label in zip(plot_job.patches, count_job_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_job.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')

count_marital_target_pct = pd.crosstab(newdf2['Target'],newdf2['marital']).apply(lambda x: x/x.sum() * 100)

count_marital_target_pct = count_marital_target_pct.transpose()



plot_marital = count_marital_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Marital Status')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Marital Status')



# Label each bar

for rec, label in zip(plot_marital.patches, count_marital_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_marital.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')

count_education_target_pct = pd.crosstab(newdf2['Target'],newdf2['education']).apply(lambda x: x/x.sum() * 100)

count_education_target_pct= count_education_target_pct.transpose()



plot_education = count_education_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Education Level')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Education Leve;')



# Label each bar

for rec, label in zip(plot_education.patches, count_education_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_education.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')

count_loan_target_pct = pd.crosstab(newdf2['Target'],newdf2['loan']).apply(lambda x: x/x.sum() * 100)

count_loan_target_pct= count_loan_target_pct.transpose()



plot_loan = count_loan_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Personal Loan Status')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Personal Laon Status')



# Label each bar

for rec, label in zip(plot_loan.patches, count_loan_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_loan.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')

count_creditdefault_target_pct = pd.crosstab(newdf2['Target'],newdf2['default']).apply(lambda x: x/x.sum() * 100)

count_creditdefault_target_pct= count_creditdefault_target_pct.transpose()



plot_credit_default = count_creditdefault_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Credit Default Status')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Credit Default Status')



# Label each bar

for rec, label in zip(plot_credit_default.patches, count_creditdefault_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_credit_default.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')

count_housingloan_target_pct = pd.crosstab(dataframe['Target'],dataframe['housing']).apply(lambda x: x/x.sum() * 100)

count_housingloan_target_pct= count_housingloan_target_pct.transpose()



plot_housing_loan = count_housingloan_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Housing Loan Status')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Housing Loan Status')



# Label each bar

for rec, label in zip(plot_housing_loan.patches, count_housingloan_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_housing_loan.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')



    

#yhdf = newdf2[newdf2['housing']== "yes"]



#Let's See How Mode Of Communication Impacts FD 
count_modeofcomm_target_pct = pd.crosstab(dataframe['Target'],dataframe['contact']).apply(lambda x: x/x.sum() * 100)

count_modeofcomm_target_pct= count_modeofcomm_target_pct.transpose()



plot_comm_mode = count_modeofcomm_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Mode Of Contact')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Mode Of Customer Contact')



# Label each bar

for rec, label in zip(plot_comm_mode.patches, count_modeofcomm_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_comm_mode.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')



    

#yhdf = newdf2[newdf2['housing']== "yes"]





plt.figure(figsize=(30,15))

ax = sns.boxplot(newdf2['Target'], newdf2['duration'], data=dataframe)



plt.tight_layout()
count_monthofcontact_target_pct = pd.crosstab(dataframe['Target'],dataframe['month']).apply(lambda x: x/x.sum() * 100)

count_monthofcontact_target_pct= count_monthofcontact_target_pct.transpose()

print(count_monthofcontact_target_pct)



plot_mnth_contact = count_monthofcontact_target_pct['yes'].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))                                                                               

plt.title('FD Subscription Rate by Month Of Contact')

plt.xlabel('FD Subscription Rate')

plt.ylabel('Month Of Last Customer Contact')



# Label each bar

for rec, label in zip(plot_mnth_contact.patches, count_monthofcontact_target_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_mnth_contact.text(rec.get_width()+0.8, rec.get_y()+ rec.get_height()-0.5, label+'%', ha = 'center')

ax = sns.boxplot(newdf2['balance'], newdf2['marital'], data=newdf2)
ax = sns.boxplot(newdf2['balance'], newdf2['education'], data=newdf2)
ax = sns.boxplot(newdf2['loan'], newdf2['balance'], data=newdf2)
ax = sns.boxplot(newdf2['housing'], newdf2['balance'], data=newdf2)
ax = sns.boxplot(newdf2['default'], newdf2['balance'], data=newdf2)
campaign_call_duration = sns.lmplot(x='duration', y='campaign',data = newdf2,hue = 'Target',fit_reg = False, scatter_kws={'alpha':0.6}, height =7)



plt.axis([0,65,0,65])

plt.ylabel('Number of Calls')

plt.xlabel('Duration of Calls (Minutes)')

plt.title('The Relationship between the Number and Duration of Calls (with Response Result)')



# Annotation

plt.axhline(y=5, linewidth=2, color="k", linestyle='--')

plt.annotate('Higher subscription rate when calls <5',xytext = (35,13),arrowprops=dict(color = 'k', width=1),xy=(30,6))

plt.show()
# ax.set_xticklabels(df["default"].unique(), rotation=45, rotation_mode="anchor")



# plt.style.use('dark_background')



fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(221)

ax = sns.boxplot(newdf2['default'], newdf2['balance'], hue = "Target", data=newdf2)

#imapct of job type, balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax1 = fig.add_subplot(221)

ax1 = sns.boxplot(newdf2['education'], newdf2['balance'], hue = "Target", data=newdf2)

#imapct of education, balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(221)

ax = sns.boxplot(newdf2['job'], newdf2['balance'], hue = "Target", data=newdf2)



#imapct of marital status, balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(221)

ax = sns.boxplot(newdf2['marital'], newdf2['balance'], hue = "Target", data=newdf2)
#imapct of personal loan status, balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(221)

ax = sns.boxplot(newdf2['loan'], newdf2['balance'], hue = "Target", data=newdf2)
#imapct of housing loan status, balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(221)

ax = sns.boxplot(newdf2['housing'], newdf2['balance'], hue = "Target", data=newdf2)
#imapct of contact type, bank balance on fd suscription.



fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(221)

ax = sns.boxplot(newdf2['age_group'], newdf2['balance'], hue = "Target", data=newdf2)
#CORRELATION MATRIX FOR ALL THE NUMERICAL ATTRIBUTES: 



newdf2.corr()



# Let's Change 'month' from words to numbers for easier analysis

lst = [newdf2]

for column in lst:

    column.loc[column["month"] == "jan", "month_int"] = 1

    column.loc[column["month"] == "feb", "month_int"] = 2

    column.loc[column["month"] == "mar", "month_int"] = 3

    column.loc[column["month"] == "apr", "month_int"] = 4

    column.loc[column["month"] == "may", "month_int"] = 5

    column.loc[column["month"] == "jun", "month_int"] = 6

    column.loc[column["month"] == "jul", "month_int"] = 7

    column.loc[column["month"] == "aug", "month_int"] = 8

    column.loc[column["month"] == "sep", "month_int"] = 9

    column.loc[column["month"] == "oct", "month_int"] = 10

    column.loc[column["month"] == "nov", "month_int"] = 11

    column.loc[column["month"] == "dec", "month_int"] = 12

    

    





def convert(newdf2, new_column, old_column):

    newdf2[new_column] = newdf2[old_column].apply(lambda x: 0 if x == 'no' else 1)

    return newdf2[new_column].value_counts()







corr_data = newdf2[['age','balance','day','duration','campaign','pdays','month_int', 'previous','Target']]

corr = corr_data.corr()

print("Correlation Matrix")

print(corr)



cor_plot = sns.heatmap(corr,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':10})

fig=plt.gcf()

fig.set_size_inches(20,10)

plt.xticks(fontsize=10,rotation=-30)

plt.yticks(fontsize=10)

plt.title('Correlation Matrix')

plt.show()

#sns.pairplot(newdf2,diag_kind="kde")
#As we saw at the start of the project that there are no missing value as such



newdf2.describe()
#Let's See below what are the attributes which has some meaningless data which are not adding any vals for building better model

newdf2.head(45212)

# Step 1: Delete the rows in  column 'poutcome' where it contains 'other'

condition = newdf2.poutcome == 'other'

newdf2.drop(newdf2[condition].index, axis = 0, inplace = True)

newdf2.describe()

                         

                         
for col in newdf2.select_dtypes(include='object').columns:

    print(col)

    print(newdf2[col].unique())

newdf2[['job','education']] = newdf2[['job','education']].replace(['unknown'],'other')



#Let's See our dataframe & verify if it has been updated

print("\n\nAfter Treatment", newdf2['education'].count)

newdf2['contact'].value_counts() 
# Drop column "contact" which seems to be not so useful



new_df1 = newdf2.copy()

newdf2.drop('contact', axis=1, inplace = True)

new_df2 = newdf2.copy()
# Let's get rid of  customer values with 'other' in education column as it doesn't make any sense to have such values in making any useful predcition .



logic = (new_df2['education'] == 'other')

new_df2.drop(new_df2[logic].index, axis = 0, inplace = True)

new_df2.info()


# Function to replace marital values with numercial

def marital_num(df):

    mar= [df]

    for data in mar: 

        data.loc[data['marital'] == "married", "marital_int"] = 1

        data.loc[data['marital'] == "single", "marital_int"] = 2

        data.loc[data['marital'] == "divorced", "marital_int"] = 3

        

#Job



#JOB: 

def job_num(df):

    jb= [df]

    for data in jb: 

        data.loc[data['job'] == "management", "Job_int"] = 1

        data.loc[data['job'] == "technician", "Job_int"] = 2

        data.loc[data['job'] == "entrepreneur", "Job_int"] = 3

        data.loc[data['job'] == "blue-collar", "Job_int"] = 4

        data.loc[data['job'] == "retired", "Job_int"] = 5

        data.loc[data['job'] == "admin.", "Job_int"] = 6

        data.loc[data['job'] == "services", "Job_int"] = 7

        data.loc[data['job'] == "self-employed", "Job_int"] = 8

        data.loc[data['job'] == "unemployed", "Job_int"] = 9

        data.loc[data['job'] == "student", "Job_int"] = 10

        data.loc[data['job'] == "housemaid", "Job_int"] = 11

        data.loc[data['job'] == "other", "Job_int"] = 12

        

#Education:



def edu_num(df):

    edu= [df]

    for data in edu: 

        data.loc[data['education'] == "primary", "education_int"] = 1

        data.loc[data['education'] == "secondary",  "education_int"] = 2

        data.loc[data['education'] == "tertiary", "education_int"] = 3

        data.loc[data['education'] == "unknown", "education_int"] = 4

    



#    

def pout_num(df):

    pout= [df]

    for data in pout: 

        data.loc[data['poutcome'] == "failure", "poutcome_int"] = 1

        data.loc[data['poutcome'] == "success",  "poutcome_int"] = 2

        data.loc[data['poutcome'] == "unknown", "poutcome_int"] = 3  





marital_num(new_df2)

job_num(new_df2)

edu_num(new_df2)

pout_num(new_df2)

    



convert(new_df2, "housing_binary", "housing")

convert(new_df2, "default_binary", "default")

convert(new_df2, "loan_binary", "loan")

convert(new_df2, "Fd Outcome", "Target")





new_df2.drop(['age','job', 'balance_group','housing','marital', 'default', 'loan', 'housing', 'education', 'month', 'poutcome', 'Target'], axis = 1, inplace = True) 

new_df2
new_df2['Fd Outcome'].value_counts()
sns.pairplot(new_df2,diag_kind="kde")
# # # Models to Fit & Evaluate



#from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_recall_curve

from sklearn.metrics import confusion_matrix,classification_report,f1_score, precision_score, recall_score, roc_curve, auc, average_precision_score, roc_auc_score, accuracy_score, precision_recall_curve, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.model_selection import KFold

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier





#Fitting the model

 

def fit_test_model(model, X_train, y_train, X_test):

    # Train the model

    model.fit(X_train, y_train)

    # Y Hat Prediction on Test Data

    model_pred = model.predict(X_test)

    return model_pred



# Function to calculate mean absolute error

def cross_val(X_train, y_train, model):

    # Applying k-Fold Cross Validation

    from sklearn.model_selection import cross_val_score

    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)

    return accuracies.mean()







# Takes in a model, trains the model, and evaluates the model on the test set

def fit_and_evaluate(model):

    # Train the model

    model.fit(X_train, y_train)

    # Make predictions and evalute

    model_pred = model.predict(X_test)

    model_cross = cross_val(X_train, y_train, model)

    #Return the performance metric

    return model_cross



# Function to calculate Accuracy Score

def model_accuracy_score(model, X_train, y_train, X_test):

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    accu_score = accuracy_score(y_test, model_pred)

    return accu_score





# Calculate Confusion Matrix & PLot To Visualize it



def draw_confmatrix(y_test, yhat, str1, str2):

    #Make predictions and evalute

    #model_pred = fit_test_model(model,X_train, y_train, X_test)

    cm = confusion_matrix( y_test, yhat, [0,1] )

    print("Confusion Matrix Is:", cm )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = [str1, str2] , yticklabels = [str1, str2] )

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    

# Function to calculate Precision Score For Class 0

def precision_score_class_0(model,X_train, y_train, X_test):

    # Make predictions and evalute

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    # Take turns considering the positive class either 0 or 1

    precision= precision_score(y_test, model_pred, pos_label=0) 

    return precision 



# Function to calculate Precision Score For Class 1

def precision_score_class_1(model,X_train, y_train, X_test):

    # Make predictions and evalute

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    # Take turns considering the positive class either 0 or 1

    precision= precision_score(y_test, model_pred, pos_label=1) 

    return precision 



# Function to calculate Recall Score For Class 0

def recallscore_class_0(model,X_train, y_train, X_test):

    # Make predictions and evalute

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    # Take turns considering the positive class either 0 or 1

    recallscore= recall_score(y_test, model_pred, pos_label=0) 

    return recallscore  



# Function to calculate Recall Score For Class 1

def recallscore_class_1(model,X_train, y_train, X_test):

    # Make predictions and evalute

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    # Take turns considering the positive class either 0 or 1

    recallscore= recall_score(y_test, model_pred, pos_label=1) 

    return recallscore 



# Function to calculate F1 Score For Class 0

def f1score_0(model,X_train, y_train, X_test):

    # Make predictions and evalute

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    # Take turns considering the positive class either 0 or 1

    fscore= f1_score(y_test, model_pred, pos_label=0) 

    return fscore 





# Function to calculate F1 Score For Class 1

def f1score_1(model,X_train, y_train, X_test):

    # Make predictions and evalute

    model_pred = fit_test_model(model,X_train, y_train, X_test)

    # Take turns considering the positive class either 0 or 1

    fscore= f1_score(y_test, model_pred, pos_label=1) 

    return fscore 



#Print Classification Report Metrics

def classificationreport(y_test, yhat):

    # Make predictions and evalute

    #model_pred = fit_test_model(model,X_train, y_train, X_test)

    class_report= classification_report(y_test, yhat)

    return class_report 



#Function To plot ROC Curve: For Given Model

def roc_auc_curve(model, X_test,TITLE):

    # predict probabilities

    probs = model.predict_proba(X_test)[:,1]

    # Calculating roc_auc score

    rocauc = roc_auc_score(y_test, probs)

    fpr, tpr, thresholds = roc_curve(y_test, probs)

    plt.figure(figsize=(10,10))

    plt.title(TITLE)

    plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % rocauc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    

def prec_recall_curve(model,X_train, y_train, X_test, STR):   

    # predict probabilities

    probs = model.predict_proba(X_test)[:,1]

    # predict class values

    yhat = fit_test_model(model,X_train, y_train, X_test)

    #calculate precision-recall curve

    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # calculate F1 score

    f1 = f1_score(y_test, yhat)

   #calculate precision-recall AUC

    aucscore = auc(recall, precision)

    # calculate average precision score

    ap = average_precision_score(y_test, probs)

    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, aucscore, ap))

    # plot no skill

    plt.figure(figsize=(10,10))

    plt.title(STR)

    plt.plot([0, 1], [0.5, 0.5], linestyle='--')

    # plot the precision-recall curve for the model

    plt.plot(recall, precision, marker='.')

    # show the plot

    plt.show()

from sklearn.model_selection import train_test_split



# # # Split Into Training and Testing Sets



# Separate out the features and targets & Print Their Shape.

#features = new_df2.drop(columns='Fd Outcome')

#targets = pd.DataFrame(new_df2['Fd Outcome'])



array = new_df2.values



X = array[:,0:15]

y = array[:,15]



#print("Y value is : ", y)

#print("\nTarget value is : ", targets)



# Split into 70% training and 30% testing set

#X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

# Split into 70% training and 30% testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)



#print(X_train.shape)

#print(X_test.shape)

#print(y_train.shape)

#print(y_test.shape)
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#MINMAX: 

minmax= MinMaxScaler()

X_train2 = pd.DataFrame(minmax.fit_transform(X_train))

X_test2 = pd.DataFrame(minmax.transform(X_test))





MinMax_X_train = X_train2

MinMax_X_test = X_test2

#print("Mimmax scaled train data\n", MinMax_X_train)

#print("Mimmax scaled test data\n", MinMax_X_test)



#STANDARD Sclaer: 



stdsc= StandardScaler()

X_train3 = pd.DataFrame(stdsc.fit_transform(X_train))

X_test3 = pd.DataFrame(stdsc.transform(X_test))



StdSc_X_train = X_train3

StdSc_X_test = X_test3

from sklearn.preprocessing import Normalizer

from scipy import stats



#Normalization Using Normalizer(): 

norm= Normalizer()

X_train4 = pd.DataFrame(norm.fit_transform(X_train))

X_test4 = pd.DataFrame(norm.transform(X_test))



Norm_X_train = X_train4

Norm_X_test = X_test4

#precision_score, recall_score, roc_curve, auc, average_precision_score, roc_auc_score, accuracy_score



lr = LogisticRegression()

clf = SVC()

knn = KNeighborsClassifier(n_neighbors=17, metric='minkowski', p= 2)

NB = GaussianNB()

dtree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

#rfc = RandomForestClassifier(n_estimators=40)



def scorer(i,j,k,l, m):

    for every in (i,j,k,l,m):

        every.fit(StdSc_X_train,y_train)

        yhat= every.predict(StdSc_X_test)

        #every.fit(X_train,y_train)

        print("Accuracy Score Is : ", accuracy_score(y_test, yhat))

        print(every.__class__.__name__, 'F1 score =', f1_score(y_test, yhat))

        print(every.__class__.__name__, 'classification Score =','\n', classification_report(y_test,yhat))

        print("Confusion Matrix HeatMap : ",draw_confmatrix(y_test, yhat, "NO FD", "YES FD"))

        

scorer (lr,clf,knn,NB,dtree)



NavBayer = GaussianNB()

NavBayer.fit(StdSc_X_train, y_train)



DT = DecisionTreeClassifier(criterion="entropy", max_depth=4)

DT.fit(StdSc_X_train, y_train)



SVM = SVC(probability=True)

SVM.fit(StdSc_X_train, y_train)



KNN = KNeighborsClassifier(n_neighbors=17, metric='minkowski', p= 2)

KNN.fit(StdSc_X_train, y_train)



LogReg = LogisticRegression()

LogReg.fit(StdSc_X_train, y_train)

#roc_auc_curve()

print(roc_auc_curve(LogReg, StdSc_X_test, "Logistic Regression ROC"))

print(roc_auc_curve(SVM, StdSc_X_test, "SVM ROC "))

print(roc_auc_curve(KNN, StdSc_X_test,"KNN ROC"))

print(roc_auc_curve(NavBayer, StdSc_X_test,"NB ROC"))

print(roc_auc_curve(DT, StdSc_X_test,"Decision Tree ROC"))
#roc_auc_curve()

#prec_recall_curve(model,X_train, y_train, X_test, STR)

print(prec_recall_curve(LogReg,StdSc_X_train, y_train, StdSc_X_test, "Logistic Regression Precision-Recall Curve"))

print(prec_recall_curve(SVM, StdSc_X_train, y_train, StdSc_X_test, "SVM ROC Precision-Recall Curve "))

print(prec_recall_curve(KNN, StdSc_X_train, y_train, StdSc_X_test,"KNN ROC Precision-Recall Curve"))

print(prec_recall_curve(NavBayer, StdSc_X_train, y_train, StdSc_X_test,"NB ROC Precision-Recall Curve"))

print(prec_recall_curve(DT, StdSc_X_train, y_train, StdSc_X_test,"Decision Tree ROC Precision-Recall Curve"))




k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

rfc = RandomForestClassifier(criterion='gini', n_estimators = 1000)#criterion = entopy,gini

rfc.fit(StdSc_X_train, y_train)

rfcpred = rfc.predict(StdSc_X_test)



draw_confmatrix(y_test, rfcpred,"No FD", "Yes FD")

print("RFC Accuracy Score:",round(accuracy_score(y_test, rfcpred),2)*100)

print("RFC F1 Score ",f1_score(y_test, rfcpred))

print(classificationreport(y_test,rfcpred))





        

RFCCV = (cross_val_score(rfc, StdSc_X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

rfc = RandomForestClassifier(criterion='entropy', n_estimators = 1500, max_features= 3, max_depth = 100)#criterion = entopy,gini

rfc.fit(StdSc_X_train, y_train)

rfcpred = rfc.predict(StdSc_X_test)



draw_confmatrix(y_test, rfcpred,"No FD", "Yes FD")

print("RFC Accuracy Score:",round(accuracy_score(y_test, rfcpred),2)*100)

print("RFC F1 Score ",f1_score(y_test, rfcpred))

print(classificationreport(y_test,rfcpred))
print(roc_auc_curve(rfc, StdSc_X_test, "Random Forrestor ROC Curve"))



print(prec_recall_curve(rfc,StdSc_X_train, y_train, StdSc_X_test, "Random Forrestor ROC Curve Precision-Recall Curve"))
from sklearn.ensemble import AdaBoostClassifier

#abcl = AdaBoostClassifier(base_estimator=dt_model, n_estimators=50)

ada = AdaBoostClassifier( n_estimators= 1000)

ada = ada.fit(StdSc_X_train, y_train)

ada_pred = ada.predict(StdSc_X_test)



draw_confmatrix(y_test, ada_pred,"No FD", "Yes FD")

print("Adaboost Accuracy Score:",round(accuracy_score(y_test, ada_pred),2)*100)

print("Adaboost F1 Score ",f1_score(y_test, ada_pred))

print(classificationreport(y_test,ada_pred))



print(confusion_matrix(y_test, ada_pred ))

print(round(accuracy_score(y_test, ada_pred),2)*100)

print(roc_auc_curve(ada, StdSc_X_test, "Ada Boost ROC Curve"))
print(prec_recall_curve(ada, StdSc_X_train, y_train, StdSc_X_test, "AdaBoost  Precision-Recall Curve"))
from sklearn.ensemble import GradientBoostingClassifier

#abcl = AdaBoostClassifier(base_estimator=dt_model, n_estimators=50)

Gfc = GradientBoostingClassifier( n_estimators= 1000)

Gfc = Gfc.fit(StdSc_X_train, y_train)

gfc_pred = Gfc.predict(StdSc_X_test)



draw_confmatrix(y_test, gfc_pred,"No FD", "Yes FD")

print("GBFC Accuracy Score:",round(accuracy_score(y_test, gfc_pred),2)*100)

print("GBFC F1 Score ",f1_score(y_test, gfc_pred))

print(classificationreport(y_test,gfc_pred))



print(confusion_matrix(y_test, gfc_pred ))

print(round(accuracy_score(y_test, gfc_pred),2)*100)
print(roc_auc_curve(Gfc, StdSc_X_test, "Gradient Boost ROC Curve"))

print(prec_recall_curve(Gfc,StdSc_X_train, y_train, StdSc_X_test, "Gradientboost  Precision-Recall Curve"))
new_df2['Fd Outcome'].value_counts()
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)



X_SMOTE, y_SMOTE = sm.fit_sample(StdSc_X_train, y_train)

pd.Series(y_SMOTE).value_counts()



sc = StandardScaler()

sc.fit(X_SMOTE)

X_train_std = sc.transform(X_SMOTE)

X_test_std = sc.transform(StdSc_X_test)
def scorer_bal(i,j,k,l, m):

    for model in (i,j,k,l,m):

        model.fit(X_train_std,y_SMOTE)

        yhat= model.predict(X_test_std)

        print("Accuracy Score Is : ", accuracy_score(y_test, yhat))

        print(model.__class__.__name__, 'F1 score =', f1_score(y_test, yhat))

        print(model.__class__.__name__, 'classification Score =','\n', classification_report(y_test, yhat))

        draw_confmatrix(y_test, yhat, "NO FD", "YES FD")

        

scorer_bal(lr,clf,knn,NB,dtree)
def scorer_bal(i,j):

    for model in (i,j):

        model.fit(X_train_std,y_SMOTE)

        yhat= model.predict(X_test_std)

        print("Accuracy Score Is : ", accuracy_score(y_test, yhat))

        print(model.__class__.__name__, 'F1 score =', f1_score(y_test, yhat))

        print(model.__class__.__name__, 'classification Score =','\n', classification_report(y_test, yhat))

        draw_confmatrix(y_test, yhat, "NO FD", "YES FD")





        

scorer_bal(rfc,Gfc)

#rfc : randomforest

#Gfc : gradientboost classifier