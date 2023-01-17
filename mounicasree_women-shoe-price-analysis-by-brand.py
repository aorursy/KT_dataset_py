# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt



shoe_df = pd.read_csv("/kaggle/input/womens-shoes-prices/7210_1.csv",low_memory=False)

shoe_df.head(10)

shoe_df.columns
shoe_normal_dist = shoe_df[shoe_df['prices.amountMin'] != shoe_df['prices.amountMax']]

nike_distr = shoe_normal_dist[shoe_normal_dist['brand']=='Nike']

nike_distr['avgPrice'] = (nike_distr['prices.amountMin'] + nike_distr['prices.amountMax'])/2

x_min = nike_distr['avgPrice'].min()

x_max = nike_distr['avgPrice'].max()

np_arr =  nike_distr['avgPrice'].values

mean = np.mean(np_arr)

std = np.std(np_arr)

mean

std



x = np.linspace(x_min, x_max, 100)



y = scipy.stats.norm.pdf(x,mean,std)



plt.plot(x,y, color='coral')



plt.grid()



plt.xlim(x_min,x_max)

plt.ylim(0,0.25)



plt.title('Test for normal distribution graph for Nike',fontsize=10)



plt.xlabel('x')

plt.ylabel('Normal Distribution')



plt.savefig("nike_price_distribution.png")

plt.show()
shoe_df.rename(columns={"prices.amountMin":"minPrice","prices.amountMax":"maxPrice"},inplace=True)

brand_shoe_df = shoe_df[["brand","minPrice","maxPrice"]]

brand_shoe_df['brand'] = brand_shoe_df['brand'].str.lower()

brand_shoe_df.head()
#Check for missing valuesand drop the rows which contains missing values in any one of the columns, as they don't contribute to analysis.

brand_shoe_df = brand_shoe_df.dropna(axis=0)

brand_shoe_df[brand_shoe_df['brand'].isnull()==True]

brand_shoe_df[brand_shoe_df['minPrice'].isnull()==True]

brand_shoe_df[brand_shoe_df['maxPrice'].isnull()==True]

#No missing values noticed
price_range_df = brand_shoe_df['maxPrice']-brand_shoe_df['minPrice']

brand_shoe_df.insert(3,'priceRange',price_range_df,True)

brand_shoe_df[brand_shoe_df['maxPrice']-brand_shoe_df['minPrice']== price_range_df.max()]
#Calculate Average Price for each and add it to the data frame

avg_price=(brand_shoe_df["minPrice"]+brand_shoe_df["maxPrice"])/2

brand_shoe_df.insert(4,"avgPrice",avg_price,True)
brand_shoe_df.head()
#drop minPrice and maxPrice columns as these are further analysis is based on average price and brand

brand_shoe_df.drop(columns=["minPrice","maxPrice"],inplace=True)
brand_shoe_df.head()
#Group the column of brand_shoe_df by brand

brand_avg = brand_shoe_df.groupby(["brand"],as_index=False).mean()

brand_avg[brand_avg['brand']=='novica']
#Checking the max scale of avg price

brand_avg[brand_avg['avgPrice']==brand_avg['avgPrice'].max()]
#Checking the min scale of avg price

brand_avg[brand_avg['avgPrice']==brand_avg['avgPrice'].min()]
#The above min and max scales conclude that the graph range should be set to the range of (0-3400)

#Dividing the btand_avg dataframe into bins for better visibility on the graph

bins = np.linspace(min(brand_avg['avgPrice']),max(brand_avg['avgPrice']),13)

bin_names=['300_range','600_range','900_range','1200_range','1500_range','1800_range','2100_range','2400_range','2700_range','3000_range','3300_range','3400_range']

brand_avg['avgPrice_bin']=pd.cut(brand_avg['avgPrice'],bins,labels=bin_names,include_lowest=True)
brand_avg
plt.scatter(brand_avg['avgPrice_bin'], brand_avg['avgPrice'])

plt.title('Scatter plot pythonspot.com')

plt.ylabel('Number of brands')

plt.xlabel('Price ranges')

plt.show()

#Grouping the brand_avg by avgPrice_bin and dropping avgPrice as further analysis is based on avg_prce bin and number of brands within that range

brand_price_group = brand_avg.groupby(['avgPrice_bin'],as_index=False).count()

brand_price_group.drop(['avgPrice'],inplace=True,axis=1)

brand_price_group
#Draw pie chart to show the number of brands that fall in a particular range

values=brand_price_group['brand']

bins = brand_price_group['avgPrice_bin']

colors=['#DAF7A6','#F8F482','#F8C282','#82F8F0','#F4B8E1','#FFC300','#525EC5','#FF5733','#DAF7A6','#FFC300','#D8A2D4','#A2D8D8']

patches, texts = plt.pie(values, startangle=90, radius=1.2,colors=colors)

percent = 100.*values/values.sum()

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(bins, percent)]

plt.legend(patches,labels , loc='best', bbox_to_anchor=(-0.1, 1.),

           fontsize=8)

np.count_nonzero(brand_avg.brand.unique())
#Get the average price of all the brands

brand_avg.head()