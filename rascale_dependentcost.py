import numpy as np 
import pandas as pd 
import seaborn as sns
from  matplotlib import pyplot

fields = ['StateCode','IndividualRate','PrimarySubscriberAndOneDependent','BusinessYear']

csv_chunks = pd.read_csv("../input/Rate.csv",iterator=True,chunksize = 1000,usecols=fields)
rates = pd.concat(chunk for chunk in csv_chunks)
rates = rates[np.isfinite(rates['PrimarySubscriberAndOneDependent'])]
rates = rates[rates.IndividualRate <9000]
rates = rates[rates.BusinessYear == 2016]

rates.head(n=5)

print(rates.describe())
import matplotlib.pyplot as plt

##Individual histogram
plt.hist(rates.IndividualRate.values)

##Remove records with 0 as PrimarySubscriberAndOneDependent
rates = rates[rates.PrimarySubscriberAndOneDependent > 0]

##OneDependent Histogram
plt.hist(rates.PrimarySubscriberAndOneDependent.values)
## Group data by state (using Median)
rateMed = rates.groupby('StateCode', as_index=False).median()
del rateMed['BusinessYear']



## JointPlot of grouped data

plt = sns.jointplot(x="IndividualRate", y="PrimarySubscriberAndOneDependent", data=rateMed)
sns.plt.show()
## Calculate the ratio
rateMed['ratio'] = rateMed['PrimarySubscriberAndOneDependent']/rateMed['IndividualRate']
rateMed.sort(['ratio'], ascending=[0])
plt = sns.barplot(rateMed.sort(['ratio'], ascending=[0]).StateCode, rateMed.ratio,palette="Blues")
sns.plt.show()