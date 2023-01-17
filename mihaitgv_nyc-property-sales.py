import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

sns.set(rc={'figure.figsize':(11,8)})

import warnings

warnings.filterwarnings('ignore')

#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)

#pd.set_option('display.max_colwidth', -1) 

df = pd.read_csv("../input/nyc-property-sales/nyc-rolling-sales.csv")

plt.figure(figsize=(11, 6))

df.head(3)
#Remove the column 'EASE-MENT' and 'Unnamed: 0'. There is no practical use of this data in our analysis. 

df.drop(columns='EASE-MENT', inplace=True)

df.drop(columns='Unnamed: 0', inplace=True)
# Display numbers in float format.

pd.options.display.float_format = '{:.2f}'.format



#Replace the ' -  ' with value of 0 across the dataframe. 

df.replace({' -  ':0}, inplace = True)



#Replace BOROUGH code with the name.

df['BOROUGH'][df['BOROUGH'] == 1] = 'Manhattan'

df['BOROUGH'][df['BOROUGH'] == 2] = 'Bronx'

df['BOROUGH'][df['BOROUGH'] == 3] = 'Brooklyn'

df['BOROUGH'][df['BOROUGH'] == 4] = 'Queens'

df['BOROUGH'][df['BOROUGH'] == 5] = 'Staten Island'



#Change data type of the columns for better data manipulation.

df = df.astype({'SALE PRICE':'int',

                'LAND SQUARE FEET':'int',

                'GROSS SQUARE FEET':'int',

                'SALE DATE': 'datetime64',

               })
#Create the column 'MONTH-YEAR' for further analysis

df['YEAR-MONTH'] = pd.to_datetime(df['SALE DATE']).dt.to_period('M')



#Create column 'PRICE/GROSS SQUARE FEET' for further analysis. 

df['PRICE/GROSS SQUARE FEET'] = df['SALE PRICE']/df['GROSS SQUARE FEET']

#For all propreties that has 'SALE PRICE' or 'GROSS SQUARE FEET' equal with 0 we will replace the result in 'PRICE/GROSS SQUARE FEET' with the value 0.

df.replace(np.nan,0, inplace = True)

df.replace(np.inf,0, inplace = True)





#Change data type of the columns for better memory utilization.

df['BOROUGH'] = df['BOROUGH'].astype('category')

df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')

df['YEAR-MONTH'] = df['YEAR-MONTH'].astype('category')
#Number of properties sold in each month



df.sort_values(by='YEAR-MONTH', inplace= True)

sns.countplot(x = 'YEAR-MONTH',

             data = df,

             )

plt.ylabel('No. of propreties sold')

plt.title('No. of properties sold in each yearly quarter')

plt.show()
#Percentage of properties sold in each borough.



#Create the dataframe for the pie chart purpose. 

df_sales_per_borough = df[['BOROUGH', 'SALE PRICE']].groupby(by='BOROUGH').count().reset_index()

df_sales_per_borough.rename(columns={'SALE PRICE': 'Count sales no.'}, inplace=True)





#Generate vizualization plot

plt.pie(df_sales_per_borough['Count sales no.'], labels=df_sales_per_borough['BOROUGH'],  autopct='%1.1f%%', labeldistance=1.2)

plt.title('Percentage of properties sold in each borough')

plt.show()
# Price/gross square feet on each borough. 



#Select propreties with price over 1000 and GSF bigger than 0.

df_price_gsqf = df.loc[(df['SALE PRICE'] > 1000) & (df['GROSS SQUARE FEET'] > 0)] 



#Create the dataframe for the bar chart.

df_price_gsqf_plt=df_price_gsqf[['BOROUGH', 'PRICE/GROSS SQUARE FEET']].groupby(by='BOROUGH').mean().sort_values(by='PRICE/GROSS SQUARE FEET', ascending = False).reset_index()



#Generate vizualization plot

sns.barplot(y = 'BOROUGH', x = 'PRICE/GROSS SQUARE FEET', data = df_price_gsqf_plt, orient = 'h' )

plt.title('Price per gross square feet on each borough')

plt.show()

# Filter the properties that are having 0 or very low values(data qaulity issues) in order to generate a more reliable and revelant average information.

new_df = df.loc[(df['SALE PRICE'] > 1000) ]



#Filter the properties from a specific borough and generate the average price sale for each month.

Manhattan = new_df[new_df['BOROUGH'] == 'Manhattan'][['YEAR-MONTH', 'SALE PRICE']].groupby(by='YEAR-MONTH').mean().reset_index()



#Generate vizualozation plot 

sns.barplot(x = 'YEAR-MONTH', y = 'SALE PRICE', data = Manhattan, orient = 'v' )

plt.show()
Bronx = new_df[new_df['BOROUGH'] == 'Bronx'][['YEAR-MONTH', 'SALE PRICE']].groupby(by='YEAR-MONTH').mean().reset_index()



sns.barplot(x = 'YEAR-MONTH', y = 'SALE PRICE', data = Brooklyn, orient = 'v' )

plt.show()
Brooklyn = new_df[new_df['BOROUGH'] == 'Brooklyn'][['YEAR-MONTH', 'SALE PRICE']].groupby(by='YEAR-MONTH').mean().reset_index()



sns.barplot(x = 'YEAR-MONTH', y = 'SALE PRICE', data = Bronx, orient = 'v' )

plt.show()
Queens = new_df[new_df['BOROUGH'] == 'Queens'][['YEAR-MONTH', 'SALE PRICE']].groupby(by='YEAR-MONTH').mean().reset_index()



sns.barplot(x = 'YEAR-MONTH', y = 'SALE PRICE', data = Queens, orient = 'v' )

plt.show()
Staten_Island = new_df[new_df['BOROUGH'] == 'Staten Island'][['YEAR-MONTH', 'SALE PRICE']].groupby(by='YEAR-MONTH').mean().reset_index()



sns.barplot(x = 'YEAR-MONTH', y = 'SALE PRICE', data = Staten_Island, orient = 'v' )

plt.show()