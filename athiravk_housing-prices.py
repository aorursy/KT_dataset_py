import os

os.chdir('../input/housing-price/')
import numpy as np

import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt
df_train=pd.read_csv('train.csv')
df_train.head()
df_train.shape
df_train.dtypes.unique()
df_train.columns
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
##calculate the distribution of the numerical value of 'SalePrice'

df_train['SalePrice'].describe()
# Numerical features:


df_train.OverallQual.unique()
quality = df_train.pivot_table(index='OverallQual',

                  values='SalePrice', aggfunc=np.median)

quality
quality.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
## the median sales price strictly increases as Overall Quality increases.
plt.scatter(x=df_train['GrLivArea'], y=np.log(df_train.SalePrice))

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
##visualize the relationship between the Ground Living Area GrLivArea and SalePrice.

#we see that increases in living area correspond to increases in price
#same for GarageArea.



plt.scatter(x=df_train['GarageArea'], y=np.log(df_train.SalePrice))

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
# there are many homes with 0 for Garage Area, indicating that they don’t have a garage
##Outliers can affect a regression model by pulling our estimated regression line further away from the true population regression line. So remove those from our data
df_train = df_train[df_train['GarageArea'] < 1200]
plt.scatter(x=df_train['GarageArea'], y=np.log(df_train.SalePrice))

plt.xlim(-200,1600)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()

# Examining null values
nulls = pd.DataFrame(df_train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
#In the case of PoolQC, the column refers to Pool Quality. Pool quality is NaN when PoolArea is 0, or there is no pool.
df_train = df_train.drop((['LotFrontage','GarageYrBlt']), axis=1)
df_train['MasVnrArea']= df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mean())
#correlation matrix

df=df_train.drop(['Id'],axis=1).copy()

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
## We can see that 'OverallQual', 'GrLivArea', 'GarageCars',' GarageArea ',' TotalBsmtSF etc are strongly correlated with 'SalePrice'.
#correlation map

f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(df.corr().iloc[7:8,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)

# this shows that MasVnrArea is not highly corelated to any other feature
sns.kdeplot(df.MasVnrArea,Label='MasVnrArea',color='g')
# shows that most of the values (nearly 60%) of MasVnrArea have zero value so replace nan values here with ZERO

 

f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(df.corr().iloc[24:25,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
#we find that data in this column is not spread enough so we can use mean of this column to fill its Missing Values
f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(df.corr().iloc[1:2,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
# LotFrontage: Linear feet of street connected to property



# And One Important Reason that we cant drop LotFrontage because it has significant corelation with our variable SalePrice
##  Features including 'SalePrice' which has strong correlation with 'SalePrice' and display it with a heat map.
#saleprice correlation matrix

k = 10  #number of variables for heatmap

corrmat = df_train.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

f,ax=plt.subplots(figsize=(12,9))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
##GarageArea' and 'GarageCars' are indicators showing the same thing at different angles, so we can see that the correlation coefficient is close.

#Also 'HouseStyle' and 'HeatingQC' are weakly correlated with 'SalePrice'.

target = np.log(df_train.SalePrice)

print("Skew is:", target.skew())

plt.hist(target, color='magenta')

plt.show()
## we see, the target variable SalePrice is not normally distributed
#scatter plots

sns.set()

sns.pairplot(df_train[cols], size = 3)

plt.show();

# Categorical features:

#Relation to SalePrice for all categorical features

li_cat_feats = list(categorical_feats)

nr_rows = 15

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=target, data=df_train, ax = axs[r][c])

    

plt.tight_layout()    

plt.show()   
#For many of the categorical there is no strong relation to the target.

#there are many positively correlated features like 'MSZoning', 'Neighborhood', 'Condition2'

print ("Unique values are:", df.Neighborhood.unique())
##These values describe Physical locations within Ames city limits.
condition_pivot = df.pivot_table(index='Neighborhood', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='red')

plt.xlabel('Condition2')

plt.ylabel('Sale Price')

plt.xticks(rotation=50)

plt.show()
print ("Original: \n")

print (df.MSZoning.value_counts(), "\n")
#  this data indicates the general zoning classification
condition_pivot = df.pivot_table(index='MSZoning', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='green')

plt.xlabel('Condition2')

plt.ylabel('Sale Price')

plt.xticks(rotation=0)

plt.show()
#The mean SalePrice for category "C(all)" is much lower than for the other categories. And the mean SalePrice for categories "RM" and "RH" is lower than for "RL" and "FV". 

#So, there is a large probability that a House of category "C(all)" has lower SalePrice than one of category "FV".
#for Street:



def encode(x):

    return 1 if x == 'Partial' else 0

df['enc_condition'] = df.SaleCondition.apply(encode)
condition_pivot = df.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='grey')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
# selected all of the houses where SaleCondition is equal to Patrial and assign the value 1, otherwise assign 0.
