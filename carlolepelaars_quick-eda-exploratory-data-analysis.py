# Dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# DataFrame
DIR = '../input/'
df = pd.read_csv(DIR + 'toy_dataset.csv')
print('Shape: {} rows and {} columns.\n'.format(df.shape[0], 
                                              df.shape[1]))

# First 5 Rows
print('First 5 rows:')
display(df.head())
# Last 5 Rows
print('Last 5 rows:')
display(df.tail())
# Basic Stats
print('Basic Statistics')
df.describe()
income_stats = [df['Income'].mean(), df['Income'].mode()[0], df['Income'].median()]
age_stats = [df['Age'].mean(), df['Age'].mode()[0], df['Age'].median()]

index = ['Mean', 'Mode', 'Median']

mean_mode_median = pd.DataFrame({'Income' : income_stats, 'Age' : age_stats}, index=index)

print('Mean, Mode and Median for Income and Age')
mean_mode_median
df['Gender'].value_counts().plot(kind='barh', 
                                 rot=0, 
                                 title='Gender Distribution', 
                                 figsize=(15,3))
male_df = df[df['Gender'] == 'Male']
female_df = df[df['Gender'] == 'Female']

x = pd.Series(male_df['Income'])
y = pd.Series(female_df['Income'])

plt.figure(figsize=(16,7))

bins = np.linspace(0, 175000, 200)

plt.hist(x, bins, alpha=0.5, label='Male Income')
plt.hist(y, bins, alpha=0.5, label='Female Income')
plt.legend(loc='upper right', prop={'size' : 28})
plt.xlabel('Income')
plt.ylabel('Frequency', rotation=0)
plt.rc('axes', labelsize=10) 
plt.rc('axes', titlesize=30) 
plt.title('Income Distribution by Gender')
# Save
#plt.savefig('Income_Dist_Gender')

plt.show()
new_df = df[df['City'] == 'New York City']
los_df = df[df['City'] == 'Los Angeles']
bos_df = df[df['City'] == 'Boston']
moun_df = df[df['City'] == 'Mountain View']
wash_df = df[df['City'] == 'Washington']
aus_df = df[df['City'] == 'Austin']
san_df = df[df['City'] == 'San Diego']
dal_df = df[df['City'] == 'Dallas']

a = pd.Series(new_df['Income'])
b = pd.Series(los_df['Income'])
c = pd.Series(bos_df['Income'])
d = pd.Series(moun_df['Income'])
e = pd.Series(wash_df['Income'])
f = pd.Series(aus_df['Income'])
g = pd.Series(san_df['Income'])
h = pd.Series(dal_df['Income'])

plt.figure(figsize=(16,7))

bins = np.linspace(0, 175000, 200)

plt.hist(a, bins, alpha=0.5, label='New York City')
plt.hist(b, bins, alpha=0.5, label='Los Angeles')
plt.hist(c, bins, alpha=0.5, label='Boston', color='cyan')
plt.hist(d, bins, alpha=0.5, label='Mountain View', color='crimson')
plt.hist(e, bins, alpha=0.5, label='Washington D.C.', color='Black')
plt.hist(f, bins, alpha=0.5, label='Austin', color='Gold')
plt.hist(g, bins, alpha=0.5, label='San Diego', color='DarkBlue')
plt.hist(h, bins, alpha=0.5, label='Dallas', color='Lime')
plt.legend(loc='upper right', prop={'size' : 22})
plt.xlabel('Income')
plt.ylabel('Frequency', rotation=0)
plt.rc('axes', labelsize=10) 
plt.rc('axes', titlesize=30) 
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20) 
plt.title('Income Distribution by City')
# Save
#plt.savefig('Income_Dist_City')

plt.show()
df['City'].value_counts().plot(kind='barh', 
                               rot=0, 
                               title='City Counts', 
                               figsize=(15,5))
df['Illness'].value_counts().plot(kind='barh', 
                                  title='Illness counts', 
                                  figsize=(15,3))