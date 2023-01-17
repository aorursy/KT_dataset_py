# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
map_file = pd.read_csv("/kaggle/input/collections-of-top-grossing-movies/Mapping_file.csv")

inflation = pd.read_csv("/kaggle/input/collections-of-top-grossing-movies/Inflation_Data_v2.csv",na_values="no data",encoding='latin-1')

currency = pd.read_csv("/kaggle/input/collections-of-top-grossing-movies/currency_data.csv")

tkt_price = pd.read_csv("/kaggle/input/collections-of-top-grossing-movies/movie_ticket_prices.csv")
tkt_price.head()
tkt_price.DATE.value_counts()
map_file.head(10)
rcParams["figure.figsize"] = 30,10

g = sns.countplot(x="Market",data=map_file,order=map_file.Market.value_counts().iloc[:10].index)

plt.xticks(rotation=45)
##Cleaning tkt_price dataset

tkt_price.dtypes
#Converting AMOUNT variable into 'float' type

tkt_price["AMOUNT"] = tkt_price["AMOUNT"].apply(lambda x:float(x.replace("$","")))

tkt_price.AMOUNT.describe()
##Cleaning currency data file

currency.head()
currency.isna().sum()
currency_imputed = currency.bfill(axis="columns")
currency_imputed.dtypes
#We need to convert all the variables into float datatype

for i in list(currency_imputed.columns)[1:]: currency_imputed[i] = currency_imputed[i].apply(float)

currency_imputed.dtypes
currency_imputed.head()
#We can see that some exchange rates are equal to 0. This is a data discrepancy as such values are not possible. 

#Let's replace 0 with 0.1 to avoid further problems

currency_imputed = currency_imputed.replace(to_replace = 0,value=0.1)
currency_imputed.head()
##Cleaning inflation data.

inflation.head()
#Imputing all missing values with 0 (0 inflation rate means there is no change in price from previous year)

inflation = inflation.fillna(0)
#Merging tkt_price dataset with map_file to get currency codes

tkt_price_v2 = pd.merge(tkt_price,map_file[["Ticket_Price_Name","Currency_Code","Market"]],left_on='COUNTRY',right_on="Ticket_Price_Name",how="left",copy=False).drop_duplicates()

#Merging with currency dataset to get exchange rates

tkt_price_v3 = pd.merge(tkt_price_v2,currency_imputed[["Currency_Code","2014"]],on="Currency_Code",how="left",copy=False)

#Calculating ticket prices of every market

tkt_price_v3["2014"] = tkt_price_v3["2014"].apply(float)

tkt_price_grpby = tkt_price_v3[["Market","AMOUNT","2014"]].groupby("Market").mean().reset_index()

tkt_price_grpby["tkt_price_org"] = tkt_price_grpby["2014"]*tkt_price_grpby["AMOUNT"]
tkt_price_grpby.head()
#Merging inflation with map_file to get market

inflation_v2 = pd.merge(inflation,map_file[["Market","Inflation_Country_Name"]],left_on="Inflation rate",right_on="Inflation_Country_Name",how="right",copy=False)

#Calculating CPI

inflation_index = pd.DataFrame()

inflation_index["Market"] = inflation_v2.Market.copy()

#Taking reference as 1979

inflation_index["1979"] = 100

for i in range(1980,2021):

    inflation_index[str(i)] = inflation_index[str(i-1)]*(1 + inflation_v2[str(i)]/100)
inflation_index.head(10)
#From CPI at every year, we can calculate prices using direct proportion using 2014 data

inflation_price = pd.DataFrame()

inflation_price["Market"] = tkt_price_grpby["Market"].copy()

#Fixing 2014 data

inflation_price["2014"] = tkt_price_grpby.tkt_price_org.copy()

inflation_index_v2 = pd.merge(inflation_index,tkt_price_grpby[["Market","tkt_price_org"]],on="Market",how="right").drop_duplicates("Market").sort_values(["Market"]).reset_index()

#Calculating prices prior to 2014

for i in range(2013,1979,-1): inflation_price[str(i)] = inflation_index_v2[str(i)]*inflation_price[str(i+1)]/inflation_index_v2[str(i+1)]

#Calculating prices after 2014

for i in range(2015,2020): inflation_price[str(i)] = inflation_index_v2[str(i)]*inflation_price[str(i-1)]/inflation_index_v2[str(i-1)]
inflation_price.head()
#Getting market from map_file

currency_v2 = pd.merge(currency_imputed,map_file[["Market","Currency_Code"]],on="Currency_Code").drop_duplicates()

#Aggregating currencies to get exchange rates on market level

currency_v3 = currency_v2.groupby("Market").mean().reset_index().sort_values(["Market"])
currency_v3.head()
#Converting currencies into USD

inflation_price_usd = pd.DataFrame()

inflation_price_usd["Market"] = inflation_price.Market.copy()



for i in range(1980,2020): inflation_price_usd[str(i)] = inflation_price[str(i)]/currency_v3[str(i)]
inflation_price_usd.head()
#Transposing columns into rows

inflation_price_melt = inflation_price_usd.melt(id_vars="Market",var_name="Year")

inflation_price_melt.head()
top_markets_list = ["Domestic","China","United Kingdom","Germany","Russia","South Korea","Japan","India","Spain","Brazil"]

tkt_prices_top_markets = inflation_price_melt[inflation_price_melt.Market.isin(top_markets_list)]

tkt_prices_top_markets.Market.value_counts()
top_markets = pd.pivot_table(tkt_prices_top_markets,values="value",index="Year",columns="Market")

top_markets
sns.lineplot(data=top_markets,dashes=False).set_title("Ticket Prices in USD")
#Read collections data

bo_coll = pd.read_csv("/kaggle/input/collections-of-top-grossing-movies/bo_collections_data.csv")

bo_coll.head()
#Checking missing values

bo_coll.isna().sum()
#Imputing release date with previous row

bo_coll_fillna = bo_coll.bfill(axis="rows")
bo_coll.dtypes 
#Converting release date into datetime variable and creating release year variable

bo_coll_fillna["Release Date"] = bo_coll_fillna["Release Date"].apply(lambda x:datetime.datetime.strptime(x,"%b %d, %Y"))

bo_coll_fillna["Release_Year"] = bo_coll_fillna["Release Date"].apply(lambda x:x.year)
#Converting opening and gross to integer variables

def convert_to_int(coll):

    if coll == "–":

        return np.NaN

    else:

        return int(coll.replace("$","").replace(",",""))

    

bo_coll_fillna["Opening"] = bo_coll_fillna["Opening"].apply(convert_to_int)

bo_coll_fillna["Gross"] = bo_coll_fillna["Gross"].apply(convert_to_int)
bo_coll_fillna.head()
bo_coll_fillna.dtypes
inflation_price_melt.dtypes
#Converting year to 'int' variable

inflation_price_melt.Year = inflation_price_melt.Year.apply(int)
#Getting ticket cost at different years in different countries

coll_adj = pd.merge(bo_coll_fillna,inflation_price_melt,how="left",

                    left_on=["Market","Release_Year"],right_on=["Market","Year"],

                    copy=False)

coll_adj.head()
#Also getting the 2019 ticket price to calculate adjusted collections in 2019

tkt_adj_19 = inflation_price_melt[inflation_price_melt.Year==2019]

coll_adj_v1 = pd.merge(coll_adj,tkt_adj_19,how="left",

                    left_on=["Market"],right_on=["Market"],

                    copy=False)

coll_adj_v1 = coll_adj_v1.drop(["Year_x","Year_y"],axis=1)

coll_adj_v1.columns = ["Release_ID","Market","Release_Date","Opening","Gross","Release_Year","tkt_release","tkt_2019"]

coll_adj_v1.head()
#Creating Tickets sold and adjusted gross variable

coll_adj_v1["tickets_sold"] = (coll_adj_v1.Gross/coll_adj_v1.tkt_release)

coll_adj_v1["Adjusted_Gross"] = coll_adj_v1.Gross*coll_adj_v1.tkt_2019/coll_adj_v1.tkt_release

coll_adj_v1.head()
coll_uglytruth = coll_adj_v1[coll_adj_v1.Release_ID == "gr1940607493"]
#Let's plot the top 20 countries by tickets sold

coll_uglytruth_top20 = coll_uglytruth.sort_values(["tickets_sold"],ascending=False).head(20)

coll_uglytruth_top20
fig, g1 = plt.subplots(figsize=(20,6))

g1.set_title('Tickets sold and gross collections in top 20 markets', fontsize=16)

g1.set_xlabel('Market', fontsize=16)

g1.set_ylabel('Tickets Sold', fontsize=16)

g1 = sns.barplot(data=coll_uglytruth_top20,x="Market",y="tickets_sold")

ylabels = ['{:,.2f}'.format(x) + 'K' for x in g1.get_yticks()/1000]

g1.set_yticklabels(ylabels)

#specify we want to share the same x-axis

g2 = g1.twinx()

color = 'tab:red'

#line plot creation

g2.set_ylabel('Gross collections', fontsize=16)

g2 = sns.lineplot(x='Market', y='Gross', data = coll_uglytruth_top20, sort=False, color="tab:red")

g2.tick_params(axis='y', color="tab:red")

ylabels2 = ['{:,.2f}'.format(x) + 'M' for x in g2.get_yticks()/1000000]

g2.set_yticklabels(ylabels2)

#show plot

plt.show()