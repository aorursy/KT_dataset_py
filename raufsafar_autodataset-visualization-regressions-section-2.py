import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
df = pd.read_csv('../input/autodataset-exploratory-data-analysis-section-1/Automobile_data_set')
df.head()
#we can calculate the correlation between variables of type "int64" or "float64" using the method "corr

df.corr()
df[['bore','stroke' ,'compression-ratio','horsepower']].corr()
# Engine size as potential predictor variable of price

sns.regplot(x ='engine-size', y = 'price', data =df)

plt.ylim(0,)
df[['engine-size', 'price']].corr()
# Highway mpg is a potential predictor variable of price

sns.regplot(x="highway-mpg", y="price", data=df)
#We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704

df[['highway-mpg', 'price']].corr()
sns.regplot(x = 'peak-rpm', y = 'price', data =df)
#'peak-rpm' and 'price' and see it's approximately -0.101616

df[['peak-rpm','price']].corr()
sns.boxplot(x ='body-style', y ='price', data=df)
sns.boxplot(x="engine-location", y="price", data=df)

# you will see that if engine loc is front how much price goes down.
# drive-wheels

sns.boxplot(x="drive-wheels", y="price", data=df)
df.describe()
#The default setting of "describe" skips variables of type object. We can apply the method "describe" on the variables of type 'object' as follows

df.describe(include=['object'])
df['drive-wheels'].value_counts()
#We can convert the series to a Dataframe as follows :

df['drive-wheels'].value_counts().to_frame()
#Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 

#'drive-wheels' to 'value_counts'.

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()

drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

drive_wheels_counts
drive_wheels_counts.index.name = 'drive-wheels'

drive_wheels_counts
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()

df_group_one
# You can also group with multiple variables. see following cells

df_group_multi = df[['drive-wheels','body-style','price']]

df_group_multi = df_group_multi.groupby(['drive-wheels','body-style'],as_index= False).mean()

df_group_multi
group_pivot = df_group_multi.pivot(index = 'drive-wheels',columns = 'body-style')

group_pivot
#use the grouped results

plt.pcolor(group_pivot, cmap='RdBu')

plt.colorbar()

plt.show()
#Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.

from scipy import stats
pearson_coef , p_value =stats.pearsonr(df['wheel-base'], df['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
# Length vs Price

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#Engine-size vs Price

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#Horsepower vs Price

#Let's calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'.

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 
df_group_two=df_group_one[['drive-wheels', 'price']].groupby(['drive-wheels'])

df_group_two.head()
df_group_two.get_group('4wd')['price']
# ANOVA

f_val, p_val = stats.f_oneway(df_group_two.get_group('fwd')['price'], df_group_two.get_group('rwd')['price'], df_group_two.get_group('4wd')['price'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val)  
# Separately: fwd and rwd

f_val, p_val = stats.f_oneway(df_group_two.get_group('fwd')['price'], df_group_two.get_group('rwd')['price'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val )
df.to_csv('autodataset2')