#Import Libraries

#Pandas

import pandas as pd



#Numpy

import numpy as np



#Matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



#Seaborn

import seaborn as sns



#Plotly

import plotly.figure_factory as ff

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objects as go

import plotly.express as px
# Read the first dataset

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# Let us look at the data

train.head() 
# This includes the list of all columns

train.columns
#Exploring the categorical variables in the dataset

train.select_dtypes(include = "object").columns
#Exploring the numerical variables in the dataset

train.select_dtypes(exclude ="object").columns
#Let's look at the info for the non.null count and data type of each column

train.info()
#By default, describe shows only numerical attributes. 

#top shows the most occured instance in that column

train.describe(include = "object").transpose()
# Let us take the unique values from MSZoning

label = train["MSZoning"].unique()

sizes = train["MSZoning"].value_counts().values



# Now we could define the Pie chart

# pull is given as a fraction of the pie radius. This serves the same purpose as explode 

fig_pie1 = go.Figure(data=[go.Pie(labels=label, values=sizes, pull=[0.1, 0, 0, 0])])

# Defining the layout

fig_pie1.update_layout(title="Zone propotion",    

        font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    ))

fig_pie1.show()



# Plotting multiple violinplots, including a box and scatter diagram

fig_vio1 = px.violin(x = train['MSZoning'], y = train["SalePrice"], box=True, points="all")

# Defining the layout

fig_vio1.update_layout(

    title="MS Zone-SalePrice",

    xaxis_title="MS Zone",

    yaxis_title="SalePrice",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig_vio1.show()
# getting the first 5 largest saleprice values

train['SalePrice'].nlargest(5)
#identifying the expensive neighborhood after slicing the indexes

HP_Index = [691,1182,1169,898,803]

train['Neighborhood'].iloc[HP_Index]
# Plotting split boxplot, including a box and scatter diagram

fig_box1 = px.box(x = train['Neighborhood'], y = train["SalePrice"])



fig_box1.update_layout(

    title="Neighborhood - SalePrice",

    xaxis_title="Neighborhood",

    yaxis_title="SalePrice",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig_box1.show()
#descriptive statistics summary

train['SalePrice'].describe()
#Plot histogram

fig_hist1 = train['SalePrice'].iplot(kind='hist', opacity=0.75, color='#007959', title='SalePrice distribution', 

                                yTitle='Count', xTitle='SalePrice', bargap = 0.20)
#jointplot - TotalBsmtSF vs SalePrice



sns.jointplot(x='TotalBsmtSF',y='SalePrice',data=train, kind='reg', color= 'orange',height = 5, ratio = 2, space=0.01)

sns.set(rc={'figure.figsize':(15,12)})
#jointplot - GrLivArea vs SalePrice



sns.jointplot(x='GrLivArea',y='SalePrice',data=train, kind='reg', color= 'green',height = 5, ratio = 2, space=0.01)

sns.set(rc={'figure.figsize':(15,12)})
# boxplot - YearBuilt - SalePrice

fig_box2 = px.box(x = train['YearBuilt'], y = train["SalePrice"])



fig_box2.update_layout(

    title="YearBuilt - SalePrice",

    xaxis_title="YearBuilt",

    yaxis_title="SalePrice",

    font=dict(

        family="Courier New, monospace",

        size=18)

    )

fig_box2.show()
# swarmplot - OverallQual - SalePrice

sns.swarmplot(x="OverallQual", y="SalePrice", data=train)
# Heatmap



# Lets consider the correlation matrix

CorrMat = train.corr() 



# using seaborn to create the heatmap

fig, ax = plt.subplots(figsize=(20,15)) 

sns.heatmap(CorrMat,linecolor='white',linewidths=1, ax=ax)
# Clustermap



sns.clustermap(CorrMat,linecolor='white',linewidths=1)
# creating pairplot using seaborn

# let us first create a list of columns which are to be studied



clmn1 = ['SalePrice', 'GarageArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF',  'OverallQual', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'LotFrontage', 'OverallCond']



sns.pairplot(train[clmn1],palette='rainbow', height = 2.5)

plt.show();
# Heatmap



# Lets consider the correlation matrix

CorrMat2 = train[clmn1].corr() 



# using seaborn to create the heatmap

fig, ax = plt.subplots(figsize=(20,15)) 

sns.heatmap(CorrMat2,linecolor='white',linewidths=1, ax=ax, annot = True)
# Comparing SalePrice and FullBath

fig, ax = plt.subplots(figsize=(15,10)) 

sns.boxplot(x="FullBath", y="SalePrice", data=train, palette='deep')



plt.title("Sale Price - Full Bathrooms")
# Comparing SalePrice and HalfBath

fig, ax = plt.subplots(figsize=(10,10)) 

sns.boxplot(x="HalfBath", y="SalePrice", data=train, palette='muted')



plt.title("Sale Price - Half Bathrooms")
# Comparing Overall Condition and HalfBath



fig, ax = plt.subplots(figsize=(15,10)) 

sns.swarmplot(x=train['OverallCond'], y=train["SalePrice"], data=train)



plt.title("Sale Price - OverallCond")
fig, ax = plt.subplots(figsize=(15,10)) 

sns.swarmplot(x=train['Electrical'], y=train["SalePrice"], data=train)



plt.title("Sale Price - Electrical")
fig, ax = plt.subplots(figsize=(15,10)) 

sns.violinplot(x="GarageCars", y="SalePrice", data=train, palette='rainbow')



plt.title("Sale Price - GarageCars")
#Plot stripplot

fig, ax = plt.subplots(figsize=(12,10)) 

sns.stripplot(x="Heating", y="SalePrice", data=train,jitter=True)
# Violin plot with hue

fig, ax = plt.subplots(figsize=(12,10)) 

sns.violinplot(x="HeatingQC", y="SalePrice", data=train, hue='CentralAir',split=True,palette='Set1')
# First of all the data type (object) related to kitchen quality is mapped to an integer value

Kitqua_map = {'Po': 0, 'Fa': 1,'TA': 2, 'Gd': 3,'Ex': 4}

train['KitchenQual'] = train['KitchenQual'].map(Kitqua_map) 



train['KitchenQual']
# KitchenQuality - SalePrice

fig, ax = plt.subplots(figsize=(15,10)) 

sns.violinplot(x="KitchenQual", y="SalePrice", data=train, palette='deep')



plt.title("Sale Price - KitchenQual")
# Let us look at the total missing values in the data set

# Lokking for any missing values in the dataframe column

miss_val = train.columns[train.isnull().any()]



# printing out the columns and the total number of missing values of all the column

for column in miss_val:

    print(column, train[column].isnull().sum())
# defining two empty lists for columns and its values

nan_columns = []

nan_values = []



for column in miss_val:

    nan_columns.append(column)

    nan_values.append(train[column].isnull().sum())



# plotting the graph

fig, ax = plt.subplots(figsize=(30,12))

plt.bar(nan_columns, nan_values, color = 'orange', width = 0.5)

ax.set_xlabel("Count of missing values")

ax.set_ylabel("Column Names")

ax.set_title("Variables with missing values");
#Exploring the numerical variables in the dataset

num_data = train.select_dtypes(exclude ="object").columns



# now let us fill the missing values with the mean of the respective column

train[num_data].apply(lambda x: x.fillna(x.mean()),axis=0)