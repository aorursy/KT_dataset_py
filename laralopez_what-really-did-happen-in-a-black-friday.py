#Import useful packages



import numpy as np # linear algebra

import pandas as pd # DataFrame

import seaborn as sns ## visualizations

import matplotlib.pyplot as plt # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression #OLS



#Use the dataset with Shift+ Enter

import os

print(os.listdir("../input"))



# Here's showed what dataset is used: BlackFriday.csv
##reading the data 

df = pd.read_csv('../input/BlackFriday.csv')
# Print the 5 head of df

print(df.head(5))

print('-----------------------------------------------------------------------------------')

# Print the info of df

print(df.info())

print('-----------------------------------------------------------------------------------')

# Print the shape of df

print(df.shape)
## to check columns with null values.

df.isna().any()
print('Product_Category_2', df['Product_Category_2'].unique())

print('-----------------------------------------------------------------------------------')

print('Product_Category_3', df['Product_Category_3'].unique())
## The values are integer range, so we'll asign zzero values to the NaN's

df.fillna(value=0,inplace=True)

## Now, we change the products valeu type from float to int

df["Product_Category_2"] = df["Product_Category_2"].astype(int)

df["Product_Category_3"] = df["Product_Category_3"].astype(int)

print('Product_Category_2', df['Product_Category_2'].unique())

print('-----------------------------------------------------------------------------------')

print('Product_Category_3', df['Product_Category_3'].unique())
df.drop(columns = ["User_ID","Product_ID"],inplace=True)

## need to always remember to use inplace to make the changes in current data frame
print(df.info())
#Age

df.iloc[:,1].head(10)
#Only the cost of every purchase can be summarized

df.iloc[:,~df.columns.isin(['Gender','Age','City_Category','Stay_In_Current_City_Years'])].describe()
#Sort by age

df1 = df.sort_values(by=['Age'])

f, axes = plt.subplots(1,2, figsize=(10, 5))

sns.countplot(df['Gender'], ax=axes[0], palette = ['C6','C0']).set_title('Who buy more?')

sns.countplot(df1['Age'], ax=axes[1]).set_title('Who has the money and energy?')

#f.show()

sns.countplot(df1['Age'],hue=df1['Gender'], palette = ['C6','C0']).set_title('Which generational gender buys more?')
#here lambda is a funtion that merge in one the two columns and set the names of the interactions. There's in total 4 types of data in 

#the new column name 

df['Genderxmarital'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)

print(df['Genderxmarital'].unique())

#get from https://www.kaggle.com/shamalip/black-friday-data-exploration
df2 = df.sort_values(by=['Age'])

sns.countplot(df2['Age'],hue=df2['Genderxmarital'],  palette = ['C6','C9','C0','C3']).set_title('Who buys more: Married or Single?')

fig, axes = plt.subplots(1,2, figsize=(10, 5))

sns.countplot(df2['Product_Category_2'],hue=df2['Genderxmarital'], ax=axes[0], orient = 'v', palette = ['C6','C9','C0','C3'])

sns.countplot(df2['Product_Category_3'],hue=df2['Genderxmarital'], ax=axes[1], orient = 'v', palette = ['C6','C9','C0','C3'])

fig.suptitle('What they buy?')
df.columns
df['City_Category'].unique()
#Occupation and City Category

fig, axes = plt.subplots(1,2, figsize=(10, 5))

sns.countplot(df2['Occupation'],hue=df2['Gender'],  palette = ['C6','C0'], ax=axes[0]).set_title('What occupation makes you buy more?')

sns.countplot(df2['City_Category'],hue=df2['Gender'],  palette = ['C6','C0'], ax=axes[1]).set_title('What locality makes you buy more?')
df["Gender"].head()
#First, change values of Gender Male=1, Femlae =0

df['Gender'] = df['Gender'].replace("F", 0)

df['Gender'] = df['Gender'].replace("M", 1)

df['Gender'].head(10)
df.iloc[:,~df.columns.isin(['Age','City_Category','Stay_In_Current_City_Years'])].describe()
#Then, change values of 'City_Category' A=0, B=1, C=2

df['City_Category'] = df['City_Category'].replace("A", 0)

df['City_Category'] = df['City_Category'].replace("B", 1)

df['City_Category'] = df['City_Category'].replace("C", 2)

df['City_Category'].head(10)
# Compute the correlation matrix

corr=df.iloc[:,~df.columns.isin(['Age','Stay_In_Current_City_Years'])].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10,133,  as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#There's a problem in statsmodels.api

import statsmodels.api as sm
df.iloc[:,~df.columns.isin(['Age','City_Category','Stay_In_Current_City_Years', 'Genderxmarital'])].head(5)

Y = df["Purchase"]

X = df.iloc[:,~df.columns.isin(['Age','City_Category','Stay_In_Current_City_Years', 'Genderxmarital','Purchase'])]

lm = LinearRegression()

fitttt = lm.fit(X,Y)
X.columns
print(lm.coef_)