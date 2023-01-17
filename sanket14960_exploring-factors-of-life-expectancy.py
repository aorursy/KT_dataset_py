import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv')
# Removing empty spaces in column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('  ', '')
df.columns = df.columns.str.replace(' ', '_')
df.columns

# Convert 'Status' to category 
df['Status'] = df['Status'].astype('category')
# First 5 rows of df
pd.set_option('display.max_columns', 22)
df.head()
# Plot a heatmap showing correlation between predicting variables 
hm = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(hm, linewidths=.5, annot=True, cmap="YlGnBu")
# Correlation of the predicting factors chosen inititally
correlation = []

for i in list(df.columns[3:9]):
     a = df['Life_expectancy'].corr(df[i])
     correlation.append(a)

zipped = zip(list(df.columns[3:9]),correlation)
zipped = list(zipped)
zipped
# Seperate life expectancy into 2 categories
label_ranges = [0, 65, np.inf]
label_names = ['Under 65', 'Over 65']

df['LE_65'] = pd.cut(df['Life_expectancy'], bins = label_ranges, labels = label_names)

_ = sns.lmplot(x = 'Total_expenditure',y = 'Life_expectancy', data = df, col='LE_65')
_.set(xlabel='% of total government expenditure on health', ylabel='Life Expectancy')

plt.show()
X = df['Adult_Mortality']
Y = df['Life_expectancy']

sns.regplot(x= 'Adult_Mortality', y='Life_expectancy', data=df) 

plt.show()
# Calculate the correlation
results = smf.ols('Life_expectancy ~ Alcohol', data=df).fit()

def corr(data, col1, col2):
    a = data[col1].corr(df[col2])
    if a > 0:
        print(col2 + ' has a positive correlation with ' + col1 +': '+ str(a))
    else:
        print(col2 + ' has a negative correlation with ' + col1 +': '+ str(a))

corr(df, 'Life_expectancy', 'Alcohol')
corr(df, 'Life_expectancy', 'BMI') 
corr(df, 'Life_expectancy', 'Schooling') 
corr(df, 'Life_expectancy', 'Population') 