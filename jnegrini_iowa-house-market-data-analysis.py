# Code you have previously used to load data

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler

from matplotlib.ticker import MaxNLocator

import missingno as msno



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

iowa_test_file_path = '../input/test.csv'



home_data = pd.read_csv(iowa_file_path)

home_data_test = pd.read_csv(iowa_test_file_path)

# Create target object and call it y

SalePrice = home_data.iloc[:,-1]

SalePriceTest = home_data_test.iloc[:,-1]
# Create 

home_data['YearBuilt'] = home_data['YearBuilt'].astype(int)

home_data['YrSold'] = home_data['YrSold'].astype(int)



Features = home_data.copy()

Features_test = home_data_test.copy()



Features.drop(['Id','SalePrice'], axis=1, inplace=True)

Features_test.drop ('Id', axis=1, inplace=True)
def basic_EDA(df):

    size = df.shape

    sum_duplicates = df.duplicated().sum()

    sum_null = df.isnull().sum().sum()

    is_NaN = df. isnull()

    row_has_NaN = is_NaN. any(axis=1)

    rows_with_NaN = df[row_has_NaN]

    count_NaN_rows = rows_with_NaN.shape

    return print("Number of Samples: %d,\nNumber of Features: %d,\nDuplicated Entries: %d,\nNull Entries: %d,\nNumber of Rows with Null Entries: %d %.1f%%" %(size[0],size[1], sum_duplicates, sum_null,count_NaN_rows[0],(count_NaN_rows[0] / df.shape[0])*100))

    

def bar_plot(x,y,xlabel,ylabel,title):

    plt.figure(figsize=(20,5))

    sns.set(style="ticks", font_scale = 1)

    ax = sns.barplot(x=x, y = y, palette="Blues_d")

    sns.despine(top=True, right=True, left=True, bottom=False)

    plt.xticks(rotation=70,fontsize = 12)

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    plt.title(title)



    for p in ax.patches:

                 ax.annotate("%.d" % p.get_height(), (p.get_x() + p.get_width() / 2., abs(p.get_height())),

                     ha='center', va='center', color='black', xytext=(0, 10),

                     textcoords='offset points')



def Null_Analysis(df,title):

    null_columns=df.columns[df.isnull().any()]

    null_columns_plot = df[null_columns].isnull().sum().sort_values(ascending = False)

    bar_plot(null_columns_plot.index, null_columns_plot,"Features", "Number of Null Values", title)
basic_EDA(Features)
basic_EDA(Features_test)
Null_Analysis(Features, "Training Set")

Null_Analysis(Features_test, "Test Set")
msno.heatmap(home_data)
#Separate in Numerical and Categorical Variables

numeric_data = Features.select_dtypes(include=[np.number])

numeric_data_test = Features_test.select_dtypes(include=[np.number])



categorical_data = Features.select_dtypes(exclude=[np.number])

categorical_data_test = Features_test.select_dtypes(exclude=[np.number])



numeric_features = numeric_data.columns
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(6,6,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(6,6,i)

        df1[feature].plot.kde()

        df2[feature].plot.kde()

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad = 0)

        plt.tick_params(axis='y', which='major', labelsize=6, pad = 0)

        plt.ylabel("Density",fontsize=8)

    fig.tight_layout(pad=3.0)

    plt.show();

    

plot_feature_distribution(numeric_data, numeric_data_test, "Train", "Test", numeric_features)    
def plot_reg(df1, label1, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(6,6,figsize=(22,15))



    for feature in features:

        i += 1

        plt.subplot(6,6,i)

        sns.scatterplot(x = df1[feature], y = "SalePrice",data=home_data,linewidth=0,s=10)

        #df2[feature].plot.kde()

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad = 0)

        plt.tick_params(axis='y', which='major', labelsize=6, pad = 0)

        #plt.ylabel("Density",fontsize=8)

    fig.tight_layout(pad=3.0)

    plt.show();

    

plot_reg(home_data, "Train", numeric_features)
def boxplot_disc(y, df):

    fig, axes = plt.subplots(5, 3, figsize=(25, 25))

    axes = axes.flatten()



    for i, j in zip(df.columns[:-1], axes):



        sortd = df.groupby([i])[y].median().sort_values(ascending=False)

        sns.boxplot(x=i,y=y,data=df,palette='Blues_d',order=sortd.index,ax=j)

        j.tick_params(labelrotation=45)

        j.yaxis.set_major_locator(MaxNLocator(nbins=18))



        plt.tight_layout()
DiscreteNumeric = home_data.loc[:,['MSSubClass','OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','GarageCars','MoSold','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','YrSold','YearBuilt','SalePrice']]

boxplot_disc('SalePrice', DiscreteNumeric)
cm = sns.diverging_palette(220, 20, sep=5, as_cmap=True)

corr = numeric_data.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(20,15))

    ax = sns.heatmap(corr, mask=mask, vmin = -0.5, vmax=0.8, linewidths=.5,annot=True,fmt=".2f",cmap = cm,annot_kws={"size": 10},cbar=False)
def boxplot(y, df):

    fig, axes = plt.subplots(14, 3, figsize=(25, 80))

    axes = axes.flatten()



    for i, j in zip(df.select_dtypes(include=['object']).columns, axes):



        sortd = df.groupby([i])[y].median().sort_values(ascending=False)

        sns.boxplot(x=i,y=y,data=df,palette='Blues_d',order=sortd.index,ax=j)

        j.tick_params(labelrotation=45)

        j.yaxis.set_major_locator(MaxNLocator(nbins=18))



        plt.tight_layout()
boxplot('SalePrice', home_data)