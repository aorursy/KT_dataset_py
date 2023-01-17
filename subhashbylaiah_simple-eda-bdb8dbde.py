import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import math

from datetime import datetime



%matplotlib inline
creditCardData = pd.read_csv("../input/creditcard.csv")

creditCardData.head()
print ('# of columns: %s'%(len(creditCardData.columns)))

creditCardData.describe()
#Checking for missing data

creditCardData.isnull().any().sum()
#Plotting a heatmap to visualize the correlation between the variables

sns.heatmap(creditCardData.corr())
# As the time provided is in seconds we can use it as seconds since epoch as we won't care about years

def convert_totime(seconds):

    return datetime.fromtimestamp(seconds);



timeAnalysis = creditCardData[['Time', 'Amount', 'Class']].copy()

timeAnalysis['datetime'] = timeAnalysis.Time.apply(convert_totime)

# As the max time is 172792 seconds and 172792 / (60*60) is about 48 hrs so we only have data for 2 days so only 

# plotting data against hours make sense

timeAnalysis['hour of the day'] = timeAnalysis.datetime.dt.hour

timeAnalysisGrouped = timeAnalysis.groupby(['Class', 'hour of the day'])['Amount'].count()
plt.figure(figsize = (10, 6))

validTransactions = timeAnalysisGrouped[0].copy()

validTransactions.name = 'Number of transactions'

validTransactions.plot.bar(title = '# of legitimate credit card transactions per hour', legend = True)
## Run this section only if your distribution is somewhat off like it shows most transactions 

## happened during the night

timeDelta = datetime.utcnow() - datetime.now() 

plt.figure(figsize = (10, 6))

timeAnalysis['hour of the day'] = timeAnalysis.datetime + timeDelta

timeAnalysis['hour of the day'] = timeAnalysis['hour of the day'].dt.hour

timeAnalysisGrouped = timeAnalysis.groupby(['Class', 'hour of the day'])['Amount'].count()

validTransactions = timeAnalysisGrouped[0].copy()

validTransactions.name = 'Number of transactions'

validTransactions.plot.bar(title = '# of legitimate credit card transactions per hour', legend = True)
plt.figure(figsize = (10, 6))

fraudTransactions = timeAnalysisGrouped[1].copy()

fraudTransactions.name = 'Number of transactions'

fraudTransactions.plot.bar(title = '# of fraud credit card transactions per hour', legend = True)
# Valid Transactions

timeAnalysis[timeAnalysis.Class == 0].Amount.plot.hist(title = 'Histogram of valid transactions')
# As the value of most transaction seems to be only about 2K - 2.5K. Lets limit the data further

timeAnalysis[(timeAnalysis.Class == 0) & (timeAnalysis.Amount <= 4000)].Amount.plot.hist(title = 'Histogram of valid transactions clipped at 4K')
# Now lets look at the Fraudulent transactions

timeAnalysis[timeAnalysis.Class == 1].Amount.plot.hist(title = 'Histogram of fraudulent transactions')
population = timeAnalysis[timeAnalysis.Class == 0].Amount

sample = timeAnalysis[timeAnalysis.Class == 1].Amount

sampleMean = sample.mean()

populationStd = population.std()

populationMean = population.mean()
z_score = (sampleMean - populationMean) / (populationStd / sample.size ** 0.5)

z_score