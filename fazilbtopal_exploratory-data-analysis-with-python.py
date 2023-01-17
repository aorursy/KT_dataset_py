import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
df = pd.read_csv('../input/auto_clean.csv')

df.head()
print(df.dtypes)
df.corr()
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()  
# Engine size as potential predictor variable of price

sns.regplot(x="engine-size", y="price", data=df)

plt.ylim(0,)
df[["engine-size", "price"]].corr()
sns.regplot(x="highway-mpg", y="price", data=df)
df[['highway-mpg', 'price']].corr()
sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm','price']].corr()
#The correlation is 0.0823, the non-diagonal elements of the table.

df[["stroke","price"]].corr() 
# There is a weak correlation between the variable 'stroke' and 'price.' 

# as such regression will not work well.  We can see this use "regplot" to demonstrate this.



sns.regplot(x="stroke", y="price", data=df)
sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
# drive-wheels

sns.boxplot(x="drive-wheels", y="price", data=df)
df.describe()
df.describe(include=['object'])
df['drive-wheels'].value_counts()
df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()

drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

drive_wheels_counts
drive_wheels_counts.index.name = 'drive-wheels'

drive_wheels_counts
# engine-location as variable

engine_loc_counts = df['engine-location'].value_counts().to_frame()

engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)

engine_loc_counts.index.name = 'engine-location'

engine_loc_counts.head(10)
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
# grouping results

df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()

df_group_one
# grouping results

df_gptest = df[['drive-wheels','body-style','price']]

grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()

grouped_test1
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')

grouped_pivot
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0

grouped_pivot
# grouping results

df_gptest2 = df[['body-style','price']]

grouped_test_bodystyle = df_gptest2.groupby(['body-style'], as_index= False).mean()

grouped_test_bodystyle
#use the grouped results

plt.pcolor(grouped_pivot, cmap='RdBu')

plt.colorbar()

plt.show()
fig, ax = plt.subplots()

im = ax.pcolor(grouped_pivot, cmap='RdBu')



#label names

row_labels = grouped_pivot.columns.levels[1]

col_labels = grouped_pivot.index



#move ticks and labels to the center

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)

ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)



#insert labels

ax.set_xticklabels(row_labels, minor=False)

ax.set_yticklabels(col_labels, minor=False)



#rotate label if too long

plt.xticks(rotation=90)



fig.colorbar(im)

plt.show()
sns.heatmap(grouped_pivot, annot=True, fmt='.2f')
df.corr()
from scipy import stats
cols = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight',

       'engine-size', 'bore', 'city-mpg', 'highway-mpg']



for col in cols: 

    pearson_coef, p_value = stats.pearsonr(df[col], df['price'])

    print("The PearsonR between {} and price is {} with a P-value of P = {}".format(

          col, pearson_coef, p_value ))

    

    if p_value < 0.001:

        print('Correlation between {} and price is statistically significant..'.format(col))

    elif p_value < 0.05:

        print('Correlation between {} and price is statistically moderate..'.format(col))

    elif p_value < 0.1:

        print('Correlation between {} and price is statistically weak..'.format(col))

    else:

        print('Correlation between {} and price is statistically not significant..'.format(col))

        

    if pearson_coef > 0 :

        if pearson_coef > 0.85:

            print('Coeff ~{} shows that the relationship is positive and very strong.\n'.format(pearson_coef))

        elif pearson_coef > 0.75 :

            print('Coeff ~{} shows that the relationship is positive and quite strong.\n'.format(pearson_coef))

        elif pearson_coef > 0.60:

            print('Coeff ~{} shows that the relationship is positive and moderately strong.\n'.format(pearson_coef))

        elif pearson_coef > 0.50 :

            print('Coeff ~{} shows that the relationship is positive and only moderate.\n'.format(pearson_coef))

        else:

            print('Coefficient ~{} shows that the relationship is positive and weak.\n'.format(pearson_coef))

    else:

        if abs(pearson_coef) > 0.85:

            print('Coeff ~{} shows that the relationship is negative and very strong.\n'.format(pearson_coef))

        elif abs(pearson_coef) > 0.75 :

            print('Coeff ~{} shows that the relationship is negative and quite strong.\n'.format(pearson_coef))

        elif abs(pearson_coef) > 0.60:

            print('Coeff ~{} shows that the relationship is negative and moderately strong.\n'.format(pearson_coef))

        elif abs(pearson_coef) > 0.50 :

            print('Coeff ~{} shows that the relationship is negative and only moderate.\n'.format(pearson_coef))

        else:

            print('Coefficient ~{} shows that the relationship is negative and weak.\n'.format(pearson_coef))
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

grouped_test2.head(2)
df_gptest
grouped_test2.get_group('4wd')['price']
# ANOVA

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], 

                              grouped_test2.get_group('rwd')['price'], 

                              grouped_test2.get_group('4wd')['price'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val)   
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val )
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  

   

print( "ANOVA results: F=", f_val, ", P =", p_val)   
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  

 

print("ANOVA results: F=", f_val, ", P =", p_val)   