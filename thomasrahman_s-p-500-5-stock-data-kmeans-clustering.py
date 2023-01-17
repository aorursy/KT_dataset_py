import os #for interacting with the operating system to import csv's path

import pandas as pd # data processing

import numpy as np # linear algebra

import matplotlib.pyplot as plt # Plotting library



pd.set_option('display.max_columns', 505) # Configuration of the jupyter notebook output (now it's configurated to display all the columns)
path = '../input/sandp500/individual_stocks_5yr/individual_stocks_5yr'

csvs = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]



df = pd.DataFrame() #Creation of a DataFrame where we are going to upload the csvs

    

for file in csvs:

    stock_df = pd.read_csv(file) #Read each csv

    stock_df.index = pd.DatetimeIndex(stock_df.date) #Definition of "Date" as index

    name = stock_df['Name'].iloc[0]

    df[name] = stock_df['close'] #Upload only the close value.



df= df.sort_index(axis=1)

df.describe() #Metrics of our new DataFrame
df.info() #More information about our DataFrame
df.shape #A bit more information ...
df.columns.unique() #Checking that ours columns are unique and we haven't uploaded duplicate csvs/stock names.
df.isnull().sum().sum() #Amount of missing values (IMPORTANT to check always this!!)
df.columns[df.isnull().any()] #Cjecking which stocks have columns with any missing value
df['ALLE'] #Example of a Stock with null values (NaN)
df=df.fillna(method ='bfill') #To fill the nulls values, we are going to use the next not null value
df['ALLE'] #Checking if this magic filling function worked ... yes! ;)
df.isnull().sum().sum() #Checking if also worked for all the 16755 null values ... double yes!!
df_month = df.resample('M').last() #Selection of the last value of each month.

df_month.head()
df_month_random100 = df_month[df_month.columns.to_series().sample(100)] #Selection of 100 random stocks for easier visualization.
start = df_month_random100.iloc[0] #Selection of the initial value for each stock selected.

returns = (df_month_random100 - start) / start #Calculation of % variation between each date and the initial value.

returns
returns = returns.iloc[1:,:] #Elimination of the first row, since the variation between the initial value ant itself is always 0.
def plot_stock(name, returns=returns): #Definition of a function function to plot the monthly variation of the stocks.

    returns[name].plot(label=name, alpha=0.9);
returns_sample = returns[returns.columns.to_series().sample(5)] #Random selection of 5 stocks

returns_sample_name = returns_sample.columns #Array with the name of the Stocks that were random selected



n=0

while n < 5: 

    plot_stock(returns_sample_name[n])

    n=n+1



plt.legend()

plt.title('Random 5 Stock Variaton')
from sklearn.cluster import KMeans # KMeans clustering library



Sum_of_squared_distances = []

K = range(1,20)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(returns.T) #Don't forget to "Transpose" the DataFrame!!

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum of squared distances')

plt.title('Elbow Method For Optimal k')

plt.show()
#The optimal "K" that is on the "Elbow" of the plot is 4. The 4 will be our clusters!

kmeans = KMeans(n_clusters=4, random_state=50).fit(returns.T);
#The Magic of K-means has been done, now we have to visualize the differents clusters.

clusters = {}

for l in np.unique(kmeans.labels_):

    clusters[l] = []



for i,l in enumerate(kmeans.predict(returns.T)):

    clusters[l].append(returns.columns[i])

for c in sorted(clusters):

    print('Cluster ' + str(c) + ': ', end='')

    for symbol in clusters[c]:

        print(symbol, end=' , ')

    print()

    print()
fig, axs = plt.subplots(2, 2,  sharey=True, figsize=(10, 10))

fig.suptitle('Stock Clusters based on their % variation since the initial value')

#plt.ylabel('Stock Price Variation')

axs[0, 0].plot(returns[clusters[0]])

axs[0, 0].set_title('Cluster 0')

axs[1, 0].plot(returns[clusters[1]])

axs[1, 0].set_title('Cluster 1')

axs[0, 1].plot(returns[clusters[2]])

axs[0, 1].set_title('Cluster 2')

axs[1, 1].plot(returns[clusters[3]])

axs[1, 1].set_title('Cluster 3')



for ax in axs.flat:

    ax.set(ylabel='% price variation')



plt.show()