import numpy as np 

import pandas as pd 

from  matplotlib import pyplot

import seaborn as sns
# Retrieve Rates

rate_fields = ['StateCode','PlanId','IndividualRate','PrimarySubscriberAndThreeOrMoreDependents','BusinessYear','Age','RatingAreaId']

rate_chunks = pd.read_csv("../input/Rate.csv",iterator=True,chunksize = 1000,usecols=rate_fields)

rates = pd.concat(chunk for chunk in rate_chunks)



# Retrieve plan attributes

attr_fields = ['PlanId','MetalLevel','BusinessYear','StateCode']



attr_chunks = pd.read_csv("../input/PlanAttributes.csv",iterator=True,chunksize = 1000,usecols=attr_fields)

attributes = pd.concat(chunk for chunk in attr_chunks)

attributes['PlanId']=attributes['PlanId'].str[0:14] # removing unneccessary post-fix



attributes_sorted=attributes.sort_values('PlanId', ascending=True)

rates_sorted=rates.sort_values('PlanId', ascending=True)
rates_filtered=rates_sorted[rates_sorted.BusinessYear==2016]

rates_filtered=rates_filtered[rates_filtered.Age=='Family Option']



attributes_filtered=attributes_sorted[attributes_sorted.BusinessYear==2016]

attributes_filtered=attributes_filtered[attributes_filtered.MetalLevel=='Low']
aggregate_rate=rates_filtered.groupby(['PlanId']).agg({'IndividualRate': np.mean,

                                        'PrimarySubscriberAndThreeOrMoreDependents': np.mean}).reset_index()

aggregate_rate['IndVSFamilyRate']=aggregate_rate['IndividualRate']/aggregate_rate['PrimarySubscriberAndThreeOrMoreDependents']

aggregate_rate=aggregate_rate[~aggregate_rate.IndVSFamilyRate.isin([np.inf,1,0,np.nan])]
merge_plan_rates = pd.merge(attributes, aggregate_rate, how='inner', on=['PlanId'])
state_agg=merge_plan_rates.groupby('StateCode').agg({'IndividualRate': np.mean,

                                        'PrimarySubscriberAndThreeOrMoreDependents': np.mean,

                                                          'IndVSFamilyRate': np.mean}).reset_index()



plt = sns.barplot(state_agg.sort(['IndVSFamilyRate'], ascending=[0]).StateCode, state_agg.IndVSFamilyRate,palette="Blues")

sns.plt.show()
g = sns.pairplot(state_agg, hue="StateCode")