%matplotlib inline 

import pandas as pd 

import numpy as np 

from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.arima_model import ARIMA, ARMA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,silhouette_score

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

import seaborn as sns 

import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = (25,10)

plt.rcParams['font.size'] = 15

from typing import Dict, List

from datetime import datetime, timedelta

plt.style.use("ggplot")

from matplotlib.colors import ListedColormap

# Aliases

from pandas.core.series import Series

np.random.seed(10000) # Inital seeding

from pandas import Timestamp

from pandas.core.frame import DataFrame
def to_isoformat(row:Series) -> Series:

    """

    Note: Original date format MM/DD/YYYY, Concat ['Date'] + ['Time']

    This funciton converts the given rows into isoformat 

    :row: given a row vector that contains Date and Time 

    :return: a Series of datetime vectors 

    """

    

    date: Series = row[0].split("/")

    time: Series = row[1]



    date[0] = date[0] if eval(date[0]) >=10 else f"0{date[0]}"

    date[1] = date[1] if eval(date[1]) >=10 else f"0{date[1]}"

    seconds: int = int( np.random.randint(10,60,size=1) ) 

    return datetime.fromisoformat( f"{date[-1]}-{date[0] }-{date[1]}T{time}:{seconds}")

    
def get_col_desc(col_one:List, col_two: List[str]) -> Dict:

    """

    This is a helper function, that return the map description of the given 2 columns 

    :col_one: given a list of column data 

    :col_two: given a list of column names 

    :return: a dictionary that contains column attributes

    """

    if len(col_one) != len(col_two):

        raise ValueError("Column one must be unique.")

    return dict(zip(col_two, col_one ))
sales = pd.read_csv("../input/supermarket-sales/supermarket_sales - Sheet1.csv")

# Join columns Date and Time Columns to add precision 

sales['Date'] = pd.to_datetime( sales[['Date','Time']].apply(to_isoformat,axis=1) ) # Concat Date and Time together to see the sale trend

sales.set_index('Date', inplace=True)

sales.drop('Time',inplace=True, axis=1)
sales.head(25)

# Factorize catogries attributes

sales['Map_Payment'] = pd.factorize(sales['Payment'])[0] # PAYMENT 

sales['Map_Gender'] = pd.factorize(sales['Gender'])[0]

sales['Map_City'] = pd.factorize(sales['City'])[0]

sales['Map_Branch'] = pd.factorize(sales['Branch'])[0]

sales['Map_Product_Line'] = pd.factorize(sales['Product line'])[0]

sales['Map_Customer_Type'] = pd.factorize(sales['Customer type'])[0]

col_names: List[str] = sales.columns.tolist() # store the colum names 
sales.to_csv("./supermarket_sales.csv")
### Variables details 

sales[col_names[6:10] + col_names[11:] ].describe() 
sales.corr() 
sales.cov() 
sns.heatmap(sales.corr() )
pd.plotting.scatter_matrix(sales[col_names[6:10] + col_names[11:] ],figsize=(50,50))

map_gender = pd.factorize(sales.Gender)

gender: Dict = get_col_desc( col_one=sorted(set(map_gender[0])), col_two=map_gender[-1].tolist() )



map_city = pd.factorize(sales.City)

city: Dict = get_col_desc(sorted(set(map_city[0].tolist())) , map_city[-1].tolist()  )
sales['Total'].mean() 
sns.distplot(sales['Total'])
# Do men buy more items than women ?

# Branch A, 

sns.factorplot(col='Map_Branch', x='Map_Gender', y='Quantity', data=sales, kind='box') 

plt.xlabel(f"Gender: {list(gender.keys())} ={list(gender.values())}")
# Sold items by city 

sns.factorplot(col='Map_City', x='Map_Gender', y='Quantity', data=sales, kind='box') 

plt.xlabel(f"Gender: {list(gender.keys())} ={list(gender.values())}")
sales.groupby('Branch')['Total'].agg(['max','count'])
# Which branch genereate more $$$ 

sales.groupby('Branch')['Total'].agg(['max','count']).plot(kind='bar')

plt.yticks(range(0,1200,50))

plt.title("Branch that generate $$$.")

plt.grid(True)
# Which Branch city generate $$$ ? 

# {'Yangon': 0, 'Naypyitaw': 1, 'Mandalay': 2}, City = Naypitaw , Branch = C == $$$

sns.factorplot(col='Map_City', x='Map_Branch', y='Total', data=sales, kind='box')
sales.groupby('City')['gross income'].agg(['max', 'min', 'mean','median'])
# Are the buyers Poor or Rich by city,

sns.boxenplot(x='Map_City', y='gross income', data=sales)
# Are the clients satisfied with the services ? 6.972

sns.distplot(sales.Rating)

plt.title("Customer Service Distributions")


# Store rating based on customer service

sns.factorplot(col='Map_City', x='Map_Branch', y='Rating', data=sales, kind='box')

# Payment methods 
sales['2019-01']['Product line'].value_counts().plot(kind='barh')

plt.title("Popular items sold in January 2019.")
sales['2019-02']['Product line'].value_counts().plot(kind='barh')

plt.title("Popular items sold in February 2019.")
sales['2019-03']['Product line'].value_counts().plot(kind='barh')

plt.title("Popular items sold in March 2019.")
sns.distplot(sales['2019-01'].Total,norm_hist=True,label="Total Sales in January 2019")

plt.legend() 
# Diff 

sales['2019-01'].Total.diff(periods=5).plot() 
# Graph the sales in January, do people spend more money on new year ? 

# In January there is a spike on the 12th and drop on the 13th



sales.Total['2019-01'].resample('D').mean().plot() 

plt.title("Sales Trend in Month of January 2019")
# Note: Sales spikes on the 11 February 2019 and drop on 13 February 2019 

sales.Total['2019-02'].resample('D').mean().plot() 

plt.title("Sales Trend in Month of February 2019")

plt.xlabel("By Dates")
# Note: Sales spikes on the 29 March 2019 and drop on 07 and 26 February 2019 

sales.Total['2019-03'].resample('1H').mean().dropna().plot() 

plt.title("Sales Trend in Month of March 2019")

plt.xlabel("By Dates")

plt.xlim([sales['2019-03'].index[0], sales['2019-03'].index[3]])
sales.Total['2019-03-29'].resample('60T').mean().dropna().plot()

plt.title("Peak Sales hourly on 29 March 2019")

plt.xlabel("Hourly Sales")
# Resample by Day

sales.Total.resample('D').mean().plot() 

plt.title('Rolling Averages: Sales Trends from January - March 2019')
# Graph Total Rolling Window

sales['Total'].rolling(window=20).mean().plot() 

plt.title("Sales Total with Rolling window = 20 ")
# Lags from 0 to 1000 

autocorr_socres: List[float] = [ sales['Total'].autocorr(lag=lag) for lag in range(1001)] 
pd.DataFrame(np.array(autocorr_socres) , index=range(0,1001), columns=['Autocorrelation Score']).plot() 

plt.title("Autocorrelation Total Price")

plt.xlabel("Number of Lags")

plt.xticks(range(0,1100,100))
# Note: it hould decrease to zero as lag increases 

autocorrelation_plot(sales.Total)

plt.grid(True)

plt.ylim([-0.1, 0.1])

sales['Quantity'].autocorr(lag=500)
autocorrelation_plot(sales.Quantity)

#plt.yticks(range(-0.1, 0.1, 0.05))

plt.ylim([-0.1, 0.1])
# Drop when lag = 5 

plot_acf(sales.Total, lags=10)
# Little spike at lags = 11 and 25

plot_acf(sales.Total, lags=30)
sales['Total'].diff(1).autocorr(lag=1) # Lag=1 

# ARMA coeff : 0.0308
model_arma = ARMA(sales.Total, (1,0)).fit() 

model_arma.summary() 
model_arma.resid['2019-01'].plot() 

plt.title("Residual in January 2019")
model_arma.resid.plot() 

plt.title("ARMA Model Residuals")

# Sales are equaly distributed from January 2019 to April 2019 
# Residual 

plot_acf(model_arma.resid, lags=30)

plt.title('ARMA Model Residual Autocorrelation')
model_arma.predict(1,100).plot() 
fig, ax = plt.subplots() 

ax = sales['2019-01']['Total'].plot(ax=ax)

model_arma.plot_predict(1,60, ax=ax)

plt.title("Forecasting Sales Quantity in March 2019")

plt.xlim([min( sales['2019-03'].index.tolist() ), max( sales['2019-03'].index.tolist() )])
# model_arma = ARMA(sales.Quantity, (2,0)).fit() 

# model_arma.summary() 
# Clusters sales based on the following columns

col_names: List[str] = sales.columns.tolist()

sales[[col_names[7], col_names[9]]] = sales[[col_names[7], col_names[9]]].astype(float)

clusters: List[str] = [col_names[7], col_names[9]] + col_names[13:]
pd.plotting.scatter_matrix( sales[clusters] , figsize=(150,150))
km_sales = KMeans(n_clusters=np.random.randint(5,10), random_state=np.random.randint(100,150))

km_sales.fit(sales[clusters])
# Number of cluster equals random number from 5 - 10  

sales['sales_cluster'] = km_sales.labels_
# Sillhouete scores 

silhouette_score(sales[clusters], km_sales.labels_ )
cluster_colors: List[str] = np.array( ['red', 'blue', 'gold', 'darkviolet', 'lime', 'yellow', 'green', 'blueviolet', 'lime', 'tomato', 'orangered','aqua','cyan', 'turquoise','steelblue'] )
sales.groupby('sales_cluster').mean() 
cluster_centers = sales.groupby('sales_cluster').mean() 

plt.scatter(sales['Map_City'], sales['gross income'], c=cluster_colors[sales['sales_cluster']], s=100 )

plt.scatter(cluster_centers['Map_City'], cluster_centers['gross income'], linewidths=1, marker="X",  c='black',s=300 )

plt.xlabel("City")

plt.ylabel("Gross Income")
table_colors = pd.DataFrame.from_dict( cluster_colors )

table_colors.columns = ['Table Colors']

table_colors 
cluster_centers = sales.groupby('sales_cluster').mean() 

plt.scatter(sales['Rating'], sales['Total'], c=cluster_colors[sales['sales_cluster']], s=25 )

plt.scatter(cluster_centers['Rating'], cluster_centers['Total'], linewidths=1, marker="X",  c='black',s=100 )

plt.xlabel("Rating")

plt.ylabel("Total")
sales['Total'].max() 
plt.scatter(sales['gross income'], sales['Rating'], c=cluster_colors[sales['sales_cluster']], s=25 )

plt.scatter(cluster_centers['gross income'], cluster_centers['Rating'], linewidths=1, marker="X",  c='black',s=100 )

plt.xlabel("Gross Income")

plt.ylabel("Rating")
pd.plotting.scatter_matrix(sales[clusters], c=cluster_colors[sales['sales_cluster']], figsize=(50,50), s=75)
cluster_ranges = range(5,100)

inertias: List[float] = []

silhouette_scores: List[float] = [] 

for n_cluster in cluster_ranges:

    kmeans = KMeans(n_clusters=n_cluster)

    kmeans.fit(sales[clusters])

    inertias.append(kmeans.inertia_)

    silhouette_scores.append(silhouette_score(sales[clusters], kmeans.labels_))
# Datframe Silhouette Scores and Inertias 

cluster_scores = pd.DataFrame(np.array([inertias, silhouette_scores]).transpose() , columns=['Inertias', 'Silhouete Scores'], index=range(5,100))
cluster_scores[ cluster_scores['Silhouete Scores'] == cluster_scores['Silhouete Scores'].max() ] # The best n_cluster = 7
cluster_scores['Inertias'].plot() 

plt.title("Clusters Inertias")

plt.grid(True)
cluster_scores['Silhouete Scores'].plot() 

plt.xticks(range(5,101,5))

plt.xlabel("Number of clusters")

plt.ylabel("Silhouette Coefficients")

scaled_models: List[float] = [] 

for n_cluster in range(5,101):

    kmeans = KMeans(n_clusters=n_cluster, random_state=np.random.randint(100,150))

    kmeans.fit( StandardScaler().fit_transform(sales[clusters]) )

    scaled_models.append( silhouette_score(StandardScaler().fit_transform(sales[clusters]), kmeans.labels_) )
cluster_scores['scaled_silhouette'] = scaled_models[0:95]
plt.plot(scaled_models)

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Coefficients")

plt.title("Scalled Models")

plt.xticks(range(5,101,5))
cluster_scores[ cluster_scores['scaled_silhouette']  == cluster_scores['scaled_silhouette'].max() ]
kmeans = KMeans(n_clusters=99, random_state=123)

kmeans.fit(StandardScaler().fit_transform(sales[clusters]))
# fig, ax = plt.subplots(1, len(business_dates), sharey=True)

business_dates: List[Timestamp] = pd.DataFrame(sales.Total.resample('1D').max()).sort_values('Total', ascending=False).index.tolist() 

for index,business_date in enumerate(pd.DataFrame(sales.Total.resample('1D').max()).sort_values('Total', ascending=False).index):

    busy_day: str = str( business_date).split(' ')[0]

    #print(f"[\033[92m+\033[0m] Business Date: {busy_day}")

    # Todo: Get store branch and location 

    sales[busy_day]['Total'].plot()

    plt.title(f"Sales on {busy_day}")

    plt.xlabel("Sales by hours")

    plt.ylabel("Total sales")

    plt.savefig(f"busy_store_{busy_day}.jpg")

    plt.show() 
sales[ str(list(filter(lambda operation_date: str(operation_date).split(' ')[0] == "2019-01-01", business_dates) )[0] ).split(" ")[0] ].groupby('Branch')['Total'].agg(['max', 'mean','min']).plot(kind='bar')

plt.title("Sales on new year day")

plt.grid(True)
round( sales.describe()['Total'].to_dict().get('mean'), 4 )
sales['Customer type'].map({'Member': 'Member', 'Normal': 'Non-Member'})

sales.groupby('Customer type').agg(['max'])['Total'].plot(kind='bar',colormap=ListedColormap(["#0580FB"]), rot=0)

plt.yticks(range(0,1100, 50))

plt.title("Total Sales between Members vs Non Members")

plt.ylabel("Total Sales")

plt.grid(True)
pd.DataFrame( sales.groupby('Branch')['Total'].resample('M').mean() )
pd.DataFrame( sales.groupby('Branch')['Total'].resample('M').mean() ).plot(kind='bar') # .plot(kind='bar')  #.hist(by='Total',sharey=True) 

sales.groupby('Branch').agg(['min','count'])['Total']
sales.groupby('Branch').agg(['min','count'])['Total'].plot(kind='bar')

plt.grid(True)

plt.title("Branch with the most customer but lower profit ")
for index,month in enumerate(range(1,4)):

    sales[f'2019-0{month}'].groupby('Product line')['Total'].agg(['min','mean', 'max']).plot(kind='bar')

    map_months: List[str] = list( map(lambda month: 'January' if month == 1  else 'February' if month == 2 else 'March', range(1,4)) )

    plt.title(f"Product Line Sales in Month of {map_months[index]}")

    plt.ylabel("Total Sales")