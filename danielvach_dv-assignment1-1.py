import numpy as np 

import pandas as pd 

from sklearn import preprocessing

import os

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

#



        

#1 Read in the data. It wasn't saved as utf-8 and records past 106 are NaN, so we'll remove those.



df = pd.read_csv('../input/AUTO.csv', encoding = "cp1252", nrows = 106, thousands = ',') 

df.head()

df.info()
#3 EDA 

df.describe() #Returns count, mean, std, min, max and percentiles for each numeric column.
profile = df.profile_report(title='Pandas Profiling Report') #This is a great package for automating EDA summaries

## EDA summary

# Categorial feature distributions:

#   - Drive type is mostly 'Front' (76%) vs. 'Rear' (24%)

#   - Fuel type is mostly 'Regular' (81%), 'Premium'  (18%), Regular (1%) 

# 

# 18x106 DF dimension

profile.to_widgets()
#2 Fill missing values (found in Luggage) with the median of the column, and shorten the dataframe

df = df.fillna(df.median())

sum(pd.isnull(df['Luggage (cu. ft.)'])) #double check the median impute, should equal 0.

    

#4 We now normalize the weight attribute using the z-score normalization 



Weight_Z = pd.DataFrame((df['Weight (lbs)']-df['Weight (lbs)'].mean())/df['Weight (lbs)'].std())



Weight_Z.head(5)

df.insert(9, 'Weight_Z', Weight_Z)
df.head()
df_dummy = pd.get_dummies(df, columns=['Name', 'Drive Type', 'Fuel Type']) 

df_dummy.head(100)
corr = df.corr()

corr
corr.style.background_gradient(cmap='viridis').set_precision(2)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

df_std = scaler.fit_transform(corr)

print(df_std)

pca = PCA(n_components=3)

pca.fit(df_std)

print(pca.explained_variance_ratio_)

print(np.cumsum(pca.explained_variance_ratio_)) # 3 principle components captures at least 90% of the variance


sns.scatterplot(data= df_dummy, x='Weight (lbs)', y='Mileage (mpg)')
sns.distplot(df_dummy['Luggage (cu. ft.)'], bins=6, hist=True)
sns.distplot(df_dummy['Mileage (mpg)'], bins=7, hist=True)
ct1 = df.pivot_table(index = df['Drive Type'], columns = df['Fuel Type'], values = 'Fuel Type', aggfunc=['count'] )

print(ct1)
from matplotlib.pyplot import plot

ct1.plot(kind='bar', stacked=False)

plt.ylabel('Count of Cars')

plt.title('Drive type')
Auto_sub = df[(df['Fuel Type'] == 'Regular') & (df['Mileage (mpg)'] > 21)] 

Auto_sub.head()