import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import missingno

import warnings

import datetime

warnings.filterwarnings("ignore")

%matplotlib inline    

print("Dependencies Loaded.")
def time_series_plot(df):

    """Given dataframe, generate times series plot of numeric data by daily, monthly and yearly frequency"""

    print("\nTo check time series of numeric data  by daily, monthly and yearly frequency")

    if len(df.select_dtypes(include='datetime64').columns)>0:

        for col in df.select_dtypes(include='datetime64').columns:

            for p in ['D', 'M', 'Y']:

                if p=='D':

                    print("Plotting daily data")

                elif p=='M':

                    print("Plotting monthly data")

                else:

                    print("Plotting yearly data")

                for col_num in df.select_dtypes(include=np.number).columns:

                    __ = df.copy()

                    __ = __.set_index(col)

                    __T = __.resample(p).sum()

                    ax = __T[[col_num]].plot()

                    ax.set_ylim(bottom=0)

                    ax.get_yaxis().set_major_formatter(

                    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

                    plt.show()



                    

def numeric_eda(df, hue=None):

    """Given dataframe, generate EDA of numeric data"""

    print("\nTo check: \nDistribution of numeric data")

    display(df.describe().T)

    columns = df.select_dtypes(include=np.number).columns

    figure = plt.figure(figsize=(20, 10))

    figure.add_subplot(1, len(columns), 1)

    for index, col in enumerate(columns):

        if index > 0:

            figure.add_subplot(1, len(columns), index + 1)

        sns.boxplot(y=col, data=df, boxprops={'facecolor': 'None'})

    figure.tight_layout()

    plt.show()

    

    if len(df.select_dtypes(include='category').columns) > 0:

        for col_num in df.select_dtypes(include=np.number).columns:

            for col in df.select_dtypes(include='category').columns:

                fig = sns.catplot(x=col, y=col_num, kind='violin', data=df, height=5, aspect=2)

                fig.set_xticklabels(rotation=90)

                plt.show()

    

    # Plot the pairwise joint distributions

    print("\nTo check pairwise joint distribution of numeric data")

    if hue==None:

        sns.pairplot(df.select_dtypes(include=np.number))

    else:

        sns.pairplot(df.select_dtypes(include=np.number).join(df[[hue]]), hue=hue)

    plt.show()





def top5(df):

    """Given dataframe, generate top 5 unique values for non-numeric data"""

    columns = df.select_dtypes(include=['object', 'category']).columns

    for col in columns:

        print("Top 5 unique values of " + col)

        print(df[col].value_counts().reset_index().rename(columns={"index": col, col: "Count"})[

              :min(5, len(df[col].value_counts()))])

        print(" ")

    

    

def categorical_eda(df, hue=None):

    """Given dataframe, generate EDA of categorical data"""

    print("\nTo check: \nUnique count of non-numeric data\n")

    print(df.select_dtypes(include=['object', 'category']).nunique())

    top5(df)

    # Plot count distribution of categorical data

    for col in df.select_dtypes(include='category').columns:

        fig = sns.catplot(x=col, kind="count", data=df, hue=hue)

        fig.set_xticklabels(rotation=90)

        plt.show()

    



def eda(df):

    """Given dataframe, generate exploratory data analysis"""

    # check that input is pandas dataframe

    if type(df) != pd.core.frame.DataFrame:

        raise TypeError("Only pandas dataframe is allowed as input")

        

    # replace field that's entirely space (or empty) with NaN

    df = df.replace(r'^\s*$', np.nan, regex=True)



    print("Preview of data:")

    display(df.head(3))



    print("\nTo check: \n (1) Total number of entries \n (2) Column types \n (3) Any null values\n")

    print(df.info())



    # generate preview of entries with null values

    if len(df[df.isnull().any(axis=1)] != 0):

        print("\nPreview of data with null values:")

        display(df[df.isnull().any(axis=1)].head(3))

        missingno.matrix(df)

        plt.show()



    # generate count statistics of duplicate entries

    if len(df[df.duplicated()]) > 0:

        print("\n***Number of duplicated entries: ", len(df[df.duplicated()]))

        display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head())

    else:

        print("\nNo duplicated entries found")



    # EDA of categorical data

    categorical_eda(df)

    

    # EDA of numeric data

    numeric_eda(df)

        

    # Plot time series plot of numeric data

    time_series_plot(df)

    

print('Template loaded.')    
# Loading in the data

ecommerce_data_path = "../input/ecommerce-bookings-data/ecommerce_data.csv"

ecom_data = pd.read_csv(ecommerce_data_path)



# Working with a copy to avoid modifying the original

ed = ecom_data.copy()



# Get a quick preview of the data

print(ed.head(),ed.info())
# Correcting datatypes

ed['date'] = ed['date'].astype('datetime64')

ed['product_id'] = ed['product_id'].astype('category')

ed['city_id'] = ed['city_id'].astype('category')

print('dtypes updated')
ed.info()
eda(ed)
from scipy import stats



# Generates the Z score for each entry in the 'orders' column

z=np.abs(stats.zscore(ed.orders))



# Print the first 20 Z scores

print(z[:20])
# Group together all outliers

outls = [i for i in z  if i>3 or i<-3]

print(outls[:20])



# Find index of outliers

outls_loc = np.where((z>3) |(z<-3))[0]

print(outls_loc[:20])
len(outls_loc)
ed = ed.drop(outls_loc)

ed.info()
ed.orders.describe()
fig,ax=plt.subplots(figsize=(18,7))

sns.scatterplot(y=ed.orders,x=ed.date,size=1)

ax.set_xlim([datetime.date(2018,7,10),datetime.date(2019,12,16)])
plt.figure(figsize=(18,12))

heatdf = pd.DataFrame({'Orders':ed.orders,'City_ID':ed.city_id,'Date':ed.date})

result = heatdf.pivot_table(index='Date',columns='City_ID',values='Orders')

sns.heatmap(data=result)