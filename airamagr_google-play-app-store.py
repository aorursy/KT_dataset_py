# Import important libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
# Import CSV files from Kaggle.com

gData = pd.read_csv("../input/googleplaystore.csv")

gDataReviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")

gData.head()
# Free VS Paid Apps in Google Store

gData.dropna(inplace=True)

types = gData['Type'].value_counts()



# Bar Chart

sns.barplot(types.index, types.values)

plt.title('Apps Classification: Free vs Paid')

plt.ylabel('Number of Installs')

plt.xlabel('Type')

plt.show()



# Create percentage view

total = len(gData['Type'])

paid = len(gData.loc[gData['Type'] == 'Paid'])

paidRating = gData.loc[gData['Type'] == 'Paid']

free = len(gData.loc[gData['Type'] == 'Free'])

freeRating = gData.loc[gData['Type'] == 'Free']



avgFree = freeRating['Rating'].mean()

avgPaid = paidRating['Rating'].mean()



paid_perc = (paid/total)*100

free_perc = (free/total)*100



print('PERCENTAGE OF DOWNLOADS:')

print('Free: '+str(round(free_perc,2))+'%'+' - ' + str(free)+' records'+' | Average Rating '+str(avgFree))

print('Paid: '+str(round(paid_perc,2))+'%'+' - ' + str(paid)+' records'+' | Average Rating '+str(avgPaid))
# Select relevant columns from the full dataset and chech distribution.

freeData = gData.loc[gData['Type'] == 'Free']

paidData = gData.loc[gData['Type'] == 'Paid']

ratingFreeData = freeData['Rating']

ratingPaidData = paidData['Rating']



fig = plt.figure() 

ax = fig.add_subplot(111)

ax.hist([ratingFreeData, ratingPaidData], label=("Free", "Paid"), bins=25, range=[0, 5])

ax.legend()



plt.show()
# Take sample of 600 records for each category

sampleFreeData = freeData.sample(n=600, random_state= 123)

samplePaidData = paidData.sample(n=600, random_state= 123)

ratingFree_s = sampleFreeData['Rating']

ratingPaid_s = samplePaidData['Rating']



# Create df for sample data

sampleData = pd.concat([sampleFreeData, samplePaidData])

sampleAnalysis = sampleData[['Rating','Type']]
print('Descriptive Statistics for Free Sample')

ratingFree_s.describe(include='all')
print('Descriptive Statistics for Paid Sample')

ratingPaid_s.describe(include='all')
# Visualise Histograms

fig = plt.figure() 

ax = fig.add_subplot(111)

ax.hist([ratingFree_s, ratingPaid_s], label=("Free Sample", "Paid Sample"), bins=25, range=[0, 5])

ax.legend()

plt.show()



# Visualise Boxplot

box = sns.boxplot(x="Type", y="Rating", data=sampleAnalysis, showfliers=True)

plt.title('Type Distribution')

plt.show()
# TEST FOR NORMALITY (Q-Q Plots)

import scipy.stats as stats



stats.probplot(sampleFreeData['Rating'], dist="norm", plot=plt)

plt.title("Normal Q-Q plot")

plt.show()



stats.probplot(samplePaidData['Rating'], dist="norm", plot=plt)

plt.title("Normal Q-Q plot")

plt.show()
# TEST FOR NORMALITY (shapiro test)

shaphiroTest_free = stats.shapiro(sampleFreeData['Rating'])

shaphiroTest_paid = stats.shapiro(samplePaidData['Rating'])

print(shaphiroTest_free)

print(shaphiroTest_paid)
# Mann Whitney U Test

stats.mannwhitneyu(sampleFreeData['Rating'], samplePaidData['Rating'])