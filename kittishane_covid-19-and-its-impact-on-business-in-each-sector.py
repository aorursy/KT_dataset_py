import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

stock_df = pd.DataFrame()

symbols = []

import os

# for dirname, _, filenames in os.walk('/kaggle/input/stock-market-dataset/stocks'):

#     for filename in filenames:

#         symbol = filename.split('.')[0]

#         symbols.append(symbol)

#         new_df = pd.read_csv(os.path.join(dirname, filename))

#         new_df['Symbol'] = symbol

#         stock_df = pd.concat([stock_df,new_df], ignore_index = True)



for dirname, _, filenames in os.walk('/kaggle/input/stock-market-dataset/stocks'):

    for i in range(1,40):

        symbol = filenames[i].split('.')[0]

        symbols.append(symbol)

        new_df = pd.read_csv(os.path.join(dirname, filenames[i]))

        new_df['Symbol'] = symbol

        stock_df = pd.concat([stock_df,new_df], ignore_index = True)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

stock_df = stock_df.loc[stock_df.Date > '2019-06-01'].copy()
mean_df = stock_df.groupby(['Date']).mean()

plt.title('Mean price of the NASDAQ index')

plt.xlabel('Date')

plt.ylabel('Mean Price')

plt.xticks(rotation = 'vertical')

plt.plot(mean_df.index,mean_df['High'])
plt.figure(figsize=(15,10))

plt.title('Prices of all the stocks on NASDAQ')

plt.ylabel('stock price')

plt.xlabel('date')



locs, labels = plt.xticks()

print(labels)

print(locs)



for symbol in symbols:

    symbol_df = stock_df.loc[stock_df['Symbol'] == symbol]

    plt.plot(symbol_df.Date,symbol_df.High, label=symbol)

    plt.legend()
sector_df = pd.read_excel('/kaggle/input/company-symbol-and-industry-sector/industry.xlsx')



sector_df = sector_df.rename(columns={'GICS\xa0Sector': 'Sector','GICS Sub Industry' : 'Sub Industry'})

print(sector_df['Sector'].value_counts())

sector_df.columns

def import_industry(industry):

    industry_list = sector_df.loc[sector_df['Sector'] == industry, 'Symbol'].to_list()

    df = pd.DataFrame()

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            symbol = filename.split('.')[0]

            if symbol in industry_list:

                new_df = pd.read_csv(os.path.join(dirname, filename))

#                 print(filename)

                new_df['Symbol'] = symbol

                df = pd.concat([df,new_df], ignore_index = True)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.loc[df['Date'] > '2019']



    return df

    

healthcare_df = import_industry('Health Care')

energy_df = import_industry('Energy')

industrial_df = import_industry('Industrials')

it_df = import_industry('Information Technology')

financials_df = import_industry('Financials')

cd_df = import_industry('Consumer Discretionary')

cs_df = import_industry('Consumer Staples')

realestate_df = import_industry('Real Estate')

utilities_df = import_industry('Utilities')

materials_df = import_industry('Materials')

communiation_df = import_industry('Communication Services')
            

def industry_plot(industry_df, industry, ax):    

#     plt.figure(figsize=(15,10))

    for symbol in industry_df['Symbol'].unique():

        symbol_df = industry_df.loc[industry_df['Symbol'] == symbol]

        ax.plot(symbol_df.Date,symbol_df.High, label = symbol)

        ax.set_title(industry)

#         ax.legend(loc='upper left', fontsize='small')

    mean = industry_df.groupby('Date').mean()

    ax.plot(mean.index,mean.High,'go',linestyle='dashed')
fig, ax  = plt.subplots( nrows = 6, ncols = 2,figsize=(25,35))



industry_plot(healthcare_df, 'Health Care', ax[0][0]) 

industry_plot(it_df, 'Information Technology',ax[0][1])

industry_plot(financials_df, 'Finalcials',ax[1][0])    

industry_plot(cd_df, 'Consumer Discretionary',ax[1][1]) 

industry_plot(cs_df, 'Consumer Staples',ax[2][0]) 

industry_plot(realestate_df, 'Real Estate',ax[2][1]) 

industry_plot(utilities_df, 'Utilities',ax[3][0]) 

industry_plot(materials_df, 'Materials',ax[3][1]) 

industry_plot(communiation_df , 'Communication Services',ax[4][0]) 

industry_plot(industrial_df , 'Industrials',ax[4][1]) 

industry_plot(energy_df , 'Energy',ax[5][0]) 
healthcare_sub_sector = sector_df.loc[sector_df['Sector'] == 'Health Care', 'Sub Industry'].to_list()

healthcare_sub_sector = list(set(healthcare_sub_sector))

print(healthcare_sub_sector)

# pharmaceuticals_sub = sector_df.loc[sector_df['Sub Industry'] == 'Pharmaceuticals', 'Symbol'].to_list()

# print(f'Pharmaceuticals: {pharmaceuticals_sub}')
sub_sector_df = healthcare_df.merge(sector_df, on = 'Symbol')[['Date','High','Symbol','Sector','Sub Industry']]

sub_sector_df.head()



sub_sector_list = sub_sector_df['Sub Industry'].unique()



sub_sector_firms = {}

for sub in sub_sector_list:

    firms = sub_sector_df.loc[sub_sector_df['Sub Industry'] == sub, 'Symbol'].unique()

    dict = {sub: firms}

    sub_sector_firms.update(dict)

# print(sub_sector_firms)    



# _________________________________________________________________________________________________________________________



y_dimension = 4

x_dimension = 3





fig, axes = plt.subplots(y_dimension, x_dimension,figsize=(20,20))



plt.subplots_adjust(hspace = 0.4 )

x_counter = 0

y_counter = 0

    

for key in sub_sector_firms:

#     print(y_counter,x_counter)

    axes[y_counter,x_counter].title.set_text(key)

    

    industry_df = healthcare_df.loc[healthcare_df['Symbol'].isin(sub_sector_firms[key])]

    

#     df.loc[df['state_code'].isin(['12','09'])].any():

    if len(sub_sector_firms[key]) >1:

        industry_mean = industry_df.groupby('Date').mean()

        axes[y_counter,x_counter].plot(industry_mean.index,industry_mean.High,'go',linestyle='dashed')



    for symbol in sub_sector_firms[key]:

        new_df = healthcare_df.loc[healthcare_df['Symbol'] == symbol]

        axes[y_counter,x_counter].tick_params(labelrotation=45)

        axes[y_counter,x_counter].tick_params(axis='x', which='major', labelsize=10)

        axes[y_counter,x_counter].plot(new_df['Date'],new_df['High'])

    x_counter = (x_counter+1) if (x_counter < x_dimension-1 ) else 0

    if x_counter == 0:

        y_counter += 1



plt.figure(figsize=(15,10))

plt.title('Biotechnology Sector')

plt.ylabel('Stock price')

plt.xlabel('Date')



for symbol in sub_sector_firms['Biotechnology']:

        new_df = healthcare_df.loc[healthcare_df['Symbol'] == symbol]

        plt.plot(new_df['Date'],new_df['High'], label = symbol)

        plt.legend()

    
