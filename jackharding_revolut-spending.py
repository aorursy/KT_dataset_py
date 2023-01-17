import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd 

import datetime as dt
revolut = pd.read_csv('/kaggle/input/revolut.csv', delimiter=';', parse_dates=[0], index_col=[0], thousands=',')



# Rename columns

revolut = revolut.rename(columns={"Completed Date ": "CompletedDate", " Description ": "Description", " Paid Out (EUR) ": "PaidOut",

       " Paid In (EUR) ": "PaidIn", " Exchange Out": "ExchangeOut", " Exchange In": "ExchangeIn", " Balance (EUR)": "Balance",

       " Category": "Category", " Notes": "Notes"}) 



del revolut['ExchangeIn']

del revolut['ExchangeOut']

del revolut["Notes"]



# Strip blanks, convert to floats and replace NaN with zeros

revolut.PaidOut = pd.to_numeric(revolut.PaidOut.str.strip(), errors='coerce').replace(np.nan, 0)

revolut.PaidIn = pd.to_numeric(revolut.PaidIn.str.strip(), errors='coerce').replace(np.nan, 0)



# Convert columns to string

def to_string(series):

    return series.str.strip().astype('str')



revolut.Description = to_string(revolut.Description)

revolut.Category = to_string(revolut.Category)



# revolut.to_csv('/kaggle/working/revolut.csv', sep=',')

revolut.head(5)
# revolut = pd.read_csv('/kaggle/working/revolut.csv')

# revolut.head(5)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('€')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

plotPerColumnDistribution(revolut, 10, 5)
categories = list(revolut.Category.unique())

categories
category_spent = list(revolut.groupby('Category').PaidOut.mean())

category_spent
explode = [0] * len(categories) # Make list of zeros

explode[categories.index("Restaurants")] = 0.2 # makes restaurant category bigger



fig1, ax1 = plt.subplots()

ax1.pie(category_spent, explode=explode, labels=categories, autopct='%1.1f%%',

        shadow=True, startangle=90, pctdistance=0.85)# Equal aspect ratio ensures that pie is drawn as a circle



#draw circle

centre_circle = plt.Circle((0,0),0.4,fc='black')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()
revolut['DayOfWeek'] = revolut.index.dayofweek + 1

revolut.head(3)
days_spent = revolut.groupby('DayOfWeek').PaidOut.mean()

days_spent.plot.bar()
category_spent = revolut.groupby('Category').PaidOut.mean()

category_spent.plot.bar()