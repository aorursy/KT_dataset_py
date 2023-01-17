import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.corr()
data.head(10)
data.columns
data.number_of_reviews.plot(kind= 'line', color= 'g', label= 'number_of_reviews', linewidth= 2, alpha= 0.5, grid= True, linestyle= ':')

data.price.plot(color= 'r', label= 'price', linewidth= 2, alpha= 0.5, grid= True, linestyle= ':')

plt.legend(loc= 'upper right')          #legend= puts label into plot

plt.xlabel('x axis')                    #label= name of label

plt.ylabel('y axis')                    #label= name of label

plt.title('Line Plot')                  #title= title of plot

plt.show()
data.plot(kind = 'scatter', x = 'number_of_reviews', y= 'price', alpha = 0.5, color = 'red')

plt.xlabel('number_of_reviews')

plt.ylabel('price')

plt.title('Number of reviews - price Scatter Plot')

plt.show()
data.price.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.price.plot(kind = 'hist',bins = 50)

plt.clf()
series = data['number_of_reviews']        

print(type(series))     

data_frame = data[['number_of_reviews']]  

print(type(data_frame))
x = data['number_of_reviews']>500     # Defans değeri 200'den büyük olan verileri x değişkenine atıyoruz.

data[x]
data[(data['number_of_reviews']>500) & (data['price']>50 )]

#	