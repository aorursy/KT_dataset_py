# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
""" 

Longtime reader, but first time submitting a Kaggle Kernel. I'm very excited about

this growing world of machine learning as a method of data analysis. As an engineer, 

I'm amazed at being able to create systems that can learn from data and identify patterns

which can lead to making decisions without many of the emotional biases.



Since this is my first Kernal, I do welcome comments and suggestions on how I can improve

my analysis or coding skills.



I'm going to look at data from the Black Friday dataset from a marketing perspective.

I want to determine if any of the categorical data impacts the quantity of products sold

in a city. Since I do not have any product margin data, I will assume, for this analysis,

that all product margins are equal.

"""
# Importing the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Importing the dataset

sales = pd.read_csv('../input/BlackFriday.csv')
sales.info()
# Understand amount of NaN data in dataset

total = sales.isnull().sum().sort_values(ascending=False)

percent = (100*sales.isnull().sum()/sales.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent(%)'])

missing_data.head(7)
# We can use seaborn to create a simple heatmap to see where we are missing data

plt.figure(figsize=(12, 7))

sns.heatmap(sales.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Find any duplicated Data Rows

sales.duplicated().sum()
# Determine the Discount from Max Price and add this value to the dataframe using transform

sales['Discount'] = sales.groupby('Product_ID')['Purchase'].transform('max') - sales['Purchase']
# Determine Percent Discount from Max Price and add this value to the dataframe using transform

sales['Percent_Discount'] = (

        100 * (sales['Purchase'] - sales.groupby('Product_ID')['Purchase'].transform('max')) /

        sales.groupby('Product_ID')['Purchase'].transform('max'))
# Check results of additions

sales.head()
# Find the number of unique products being sold by Product Category

sales['Product_ID'].nunique()
# Check to determine if any products are listed in more than one Product Category

prods = sales.groupby('Product_Category_1')['Product_ID'].nunique()

prods.sum()
# Since all of the products are only grouped into one Product Category,

# examine the level of product prices & discounts by city using boxplots.

plt.figure(figsize=(12, 7))

sns.boxplot(x='Product_Category_1', y='Purchase', hue_order = ['A', 'B', 'C'], 

            hue = 'City_Category', data=sales)
# Now by discount

plt.figure(figsize=(12, 7))

sns.boxplot(x='Product_Category_1', y='Discount', hue_order = ['A', 'B', 'C'], 

            hue = 'City_Category', data=sales)
# Since there are 3623 products, find the top 20 selling products

top_purc = sales.groupby('Product_ID').sum().reset_index().nlargest(20, columns='Purchase')
# Plot Top Items Sales and Discounts by Product IDs

pid_1 = (        

        top_purc

        .groupby('Product_ID')

        .agg({'Purchase': 'sum', 'Discount': 'sum'})

        .sort_values(by = 'Purchase', ascending=False)

        .stack().reset_index()

        .rename(columns={'level_1': 'Variable', 0: 'Value'})

        )



plt.figure(figsize=(12, 7))

plt.xticks(rotation=75)

sns.barplot(x='Product_ID', y='Value', hue='Variable', data=pid_1)
# Find the top 20 discounted products by Product IDs

top_disc = sales.groupby('Product_ID').sum().reset_index().nlargest(20, columns='Discount')



# Plot Top Items Discounts and Sales

pid_2 = (        

        top_disc

        .groupby('Product_ID')

        .agg({'Purchase': 'sum', 'Discount': 'sum'})

        .sort_values(by = 'Purchase', ascending=False)

        .stack().reset_index()

        .rename(columns={'level_1': 'Variable', 0: 'Value'})

        )



plt.figure(figsize=(12, 7))

plt.xticks(rotation=75)

sns.barplot(x='Product_ID', y='Value', hue='Variable', data=pid_2)
# The two preceding plots clearly show that the discounts are not straight reductions

# from the sales prices.



# The remainder of this analysis will focus on the top 20 selling products

keys_purc = list(top_purc['Product_ID'].values)

purc = sales[sales['Product_ID'].isin(keys_purc)].copy()
# I want to examine if there are any data attributes that significantly affect the 

# amount of products sold. In order to perform a principle component analysis (PCA),

# I need to convert categorical features to dummy variables using pandas. Otherwise

# machine learning algorithm won't be able to directly take in those features as inputs.



# Change Product_ID, Gender, Age, and Stay_In_Current_City to dummy variables.

prod_id = pd.get_dummies(purc['Product_ID'],drop_first=True)

sex = pd.get_dummies(purc['Gender'],drop_first=True)

age = pd.get_dummies(purc['Age'],drop_first=True)

stay = pd.get_dummies(purc['Stay_In_Current_City_Years'],drop_first=True)
# Remove original columns from copy of original dataset

purc.drop(['Product_ID','Gender','Age','Stay_In_Current_City_Years'],axis=1,inplace=True)
# Due to high quantities of NaN, drop Product_Category_2 and Product_Category_3 columns

purc.drop(['Product_Category_2','Product_Category_3'],axis=1,inplace=True)
# Add new columns onto dataframe copy and view results

purc = pd.concat([purc,prod_id,sex,age,stay],axis=1)

purc.info()
# Since Occupation and Product_Category_1 are currently integers, they need to be

# converted to string values before conversion to categorical data. Since both columns

# are currently numberic, I added a character prefix to each so that I could identify

# the original data columns. 

purc['Occupation'] = purc['Occupation'].apply(lambda x: 'j' + str(x))

purc['Product_Category_1'] = purc['Product_Category_1'].apply(lambda x: 'p' + str(x))
# Change Occupation and Product_Category_1 to dummy variables.

job = pd.get_dummies(purc['Occupation'],drop_first=True)

cat_1 = pd.get_dummies(purc['Product_Category_1'],drop_first=True)
# Remove original columns from copy of original dataset

purc.drop(['Occupation','Product_Category_1'],axis=1,inplace=True)
# Add new columns onto dataframe copy. 

purc = pd.concat([purc,job,cat_1],axis=1)
# There are now close to 60 degrees of freedom in this dataframe. The dataframe is now  

# prepared for Principle Compoent Analysis. For this analysis, I'm going to use the City_Category

# as the target values. I want to understand if any of these attributes affect the sales

# per city.



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
# Define features for modelling x component values omitting the City_Category. I am also

# omitting the Discount column since it is calculated using the Percent_Discount values.

list(purc)

features = ['Marital_Status','Purchase','Percent_Discount',

            'P00010742','P00025442','P00028842','P00046742','P00052842',

            'P00057642','P00059442','P00080342','P00110742','P00110842',

            'P00110942','P00112142','P00112542','P00114942','P00145042',

            'P00148642','P00184942','P00237542','P00255842',

            'M','18-25','26-35','36-45','46-50','51-55','55+',

            '1','2','3','4+',

            'j1','j10','j11','j12','j13','j14','j15','j16','j17','j18',

            'j19','j2','j20','j3','j4','j5','j6','j7','j8','j9',

            'p10','p16','p6']
# Separating out the features for model inputs

x = purc.loc[:, features].values
# Separating out the target values

y = purc.loc[:,['City_Category']].values

y_df = pd.DataFrame(data=y,columns=['City_Category'])
# Standardizing the features

x = StandardScaler().fit_transform(x)
# Declaring a 2 dimensional model and transforming the data

pca = PCA(n_components=2)

prinComps = pca.fit_transform(x)
# Create 2D plot of PCA results

prinDf = pd.DataFrame(data = prinComps,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([prinDf, y_df], axis = 1)



plt.figure(figsize=(12,7))

ax = sns.scatterplot(x='principal component 1', y='principal component 2', hue='City_Category', 

                     hue_order = ['A', 'B', 'C'], data=finalDf)
# Understand how much of the data variance is described by the PCA model, which is less than 8%.

pca.explained_variance_ratio_
# Since I've reduced many degrees of freedom down to only 2, I wanted to see if some attributes

# are contributing more than others to describe the variations. I made a heatmap of the results.

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=features)



plt.figure(figsize=(12,7))

sns.heatmap(df_comp,cmap='plasma')
# It can be seen many of the job or Occupation categories do not contribute to the variances,

# so let's remove those features with the 'j' prefix and run the model again.

features = ['Marital_Status','Purchase','Percent_Discount',

            'P00010742','P00025442','P00028842','P00046742','P00052842',

            'P00057642','P00059442','P00080342','P00110742','P00110842',

            'P00110942','P00112142','P00112542','P00114942','P00145042',

            'P00148642','P00184942','P00237542','P00255842',

            'M','18-25','26-35','36-45','46-50','51-55','55+',

            '1','2','3','4+',

            'p10','p16','p6']



x = purc.loc[:, features].values
# Standardizing and transforming the features

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

prinComps = pca.fit_transform(x)
# Create 2D plot of PCA results

prinDf = pd.DataFrame(data = prinComps,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([prinDf, y_df], axis = 1)



plt.figure(figsize=(12,7))

ax = sns.scatterplot(x='principal component 1', y='principal component 2', hue='City_Category', 

                     hue_order = ['A', 'B', 'C'], data=finalDf)
# Understand how much of the data variance is described by the PCA model, which increased to about 12%.

pca.explained_variance_ratio_
# Rerun the heatmap

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=features)



plt.figure(figsize=(12,7))

sns.heatmap(df_comp,cmap='plasma')
# It can now be seen that the marital status, age groups and stay in the city do not contribute as much, 

# so those variables can be removed.



features = ['Purchase','Percent_Discount',

            'P00010742','P00025442','P00028842','P00046742','P00052842',

            'P00057642','P00059442','P00080342','P00110742','P00110842',

            'P00110942','P00112142','P00112542','P00114942','P00145042',

            'P00148642','P00184942','P00237542','P00255842',

            'M','p10','p16','p6']



x = purc.loc[:, features].values
# Standardizing and transforming the features

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

prinComps = pca.fit_transform(x)
# Create 2D plot of PCA results

prinDf = pd.DataFrame(data = prinComps,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([prinDf, y_df], axis = 1)



plt.figure(figsize=(12,7))

ax = sns.scatterplot(x='principal component 1', y='principal component 2', hue='City_Category', 

                     hue_order = ['A', 'B', 'C'], data=finalDf)
# Understand how much of the data variance is described by the PCA model, which increased to about 17%.

pca.explained_variance_ratio_
# Rerun the heatmap

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=features)



plt.figure(figsize=(12,7))

sns.heatmap(df_comp,cmap='plasma')
# Let's reexamine an earlier plot created that described the amount of sales by product_ID.

plt.figure(figsize=(12, 7))

plt.xticks(rotation=75)

sns.barplot(x='Product_ID', y='Value', hue='Variable', data=pid_1)
# And the plot that described the amount of discount by product_ID.

plt.figure(figsize=(12, 7))

plt.xticks(rotation=75)

sns.barplot(x='Product_ID', y='Value', hue='Variable', data=pid_2)
# When comparing same Product_Ids between the three graphs, it does not appear that the variance

# impact is directly correlated to the total value of sales or to the discounted purchase. 

# Let's remove some of the Product_IDs and view the impact on the results.



features = ['Purchase','Percent_Discount',

            'P00010742','P00028842','P00046742','P00052842',

            'P00057642','P00059442',

            'P00110942','P00112142','P00112542','P00114942','P00145042',

            'P00148642','P00255842',

            'M','p10','p16','p6']



x = purc.loc[:, features].values
# Standardizing and transforming the features

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

prinComps = pca.fit_transform(x)
# Create 2D plot of PCA results

prinDf = pd.DataFrame(data = prinComps,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([prinDf, y_df], axis = 1)



plt.figure(figsize=(12,7))

ax = sns.scatterplot(x='principal component 1', y='principal component 2', hue='City_Category', 

                     hue_order = ['A', 'B', 'C'], data=finalDf)
# Understand how much of the data variance is described by the PCA model, which increased to about 23%.

pca.explained_variance_ratio_
# Rerun the heatmap

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=features)



plt.figure(figsize=(12,7))

sns.heatmap(df_comp,cmap='plasma')
# Remove more Product IDs

features = ['Purchase','Percent_Discount',

            'P00052842','P00255842',

            'M','p10','p16','p6']



x = purc.loc[:, features].values
# Standardizing and transforming the features

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

prinComps = pca.fit_transform(x)
# Create 2D plot of PCA results

prinDf = pd.DataFrame(data = prinComps,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([prinDf, y_df], axis = 1)



plt.figure(figsize=(12,7))

ax = sns.scatterplot(x='principal component 1', y='principal component 2', hue='City_Category', 

                     hue_order = ['A', 'B', 'C'], data=finalDf)
# Understand how much of the data variance is described by the PCA model, which increased to about 53%.

pca.explained_variance_ratio_
# Rerun the heatmap

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=features)



plt.figure(figsize=(12,7))

sns.heatmap(df_comp,cmap='plasma')
# Remove marital status and Product Category 6

features = ['Purchase','Percent_Discount',

            'P00052842','P00255842',

            'p10','p16']



x = purc.loc[:, features].values
# Standardizing and transforming the features

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

prinComps = pca.fit_transform(x)
# Create 2D plot of PCA results

prinDf = pd.DataFrame(data = prinComps,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([prinDf, y_df], axis = 1)



plt.figure(figsize=(12,7))

ax = sns.scatterplot(x='principal component 1', y='principal component 2', hue='City_Category', 

                     hue_order = ['A', 'B', 'C'], data=finalDf)
# Understand how much of the data variance is described by the PCA model, which increased to about 71%.

pca.explained_variance_ratio_
# Rerun the heatmap

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=features)



plt.figure(figsize=(12,7))

sns.heatmap(df_comp,cmap='plasma')
"""

PCA analysis can be used as a tool in exploratory data analysis. From the final heatmap, it can

be seen that the largest contributor to the Principal Component 1 variation is the purchase value.

This is closely followed by the Percent Discount values. For the Principal Component 2 variation,

the strongest contributors are Product IDs and Product Categories. At this point further analysis 

could be conducted using the only primary drivers identified in this final heatmap.



From a marketing viewpoint, it would be interesting to understand what types of advertising were

used for products P00052842 and P00255842 by city in order to determine any effects. Also, if any

unique advertising was created for product groups p10 and p16.



This concludes my analysis on the Black Friday dataset. I hope that you found it useful and as 

interesting as I did. Again, please provide me comments and/or suggestions on how I can improve

my analysis or coding skills.

"""