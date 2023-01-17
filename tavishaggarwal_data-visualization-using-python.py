import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
housePropertyDataset = pd.read_csv("../input/house-prices-dataset/train.csv")
sns.set()



plt.plot(housePropertyDataset['SalePrice'])



plt.legend(loc='upper right')

plt.show()
sns.set()



# We can use the styles for the plot

# print(plt.style.available) used to print all available styles

plt.style.use('ggplot')



plt.plot(housePropertyDataset['SalePrice'])



plt.xlabel('Property ID')

plt.ylabel('Sales Price')



plt.xlim(1100,1200)

plt.ylim(100000,800000)



plt.title('Price of various property located')



plt.show()

plt.axes([200, 100000, 1, 1])

plt.plot(housePropertyDataset['SalePrice'])



plt.xlabel('Property ID')

plt.ylabel('Sales Price')



plt.title('Price of various property located')



plt.show()

plt.style.use('ggplot')



plt.subplot(1,2,1)

plt.plot(housePropertyDataset['SalePrice'], color='red')

plt.title('Property Sale Price variation')

plt.xlabel('Property ID')

plt.ylabel('Sales Price')



plt.subplot(1,2,2)

plt.scatter(housePropertyDataset['GarageArea'],housePropertyDataset['SalePrice'], color='green')

plt.title('Top 10 Female Population')

plt.xlabel('Property Sales Price')

plt.ylabel('Garbage Area')





# Improve the spacing between subplots and display them

plt.tight_layout()

plt.show()
x = housePropertyDataset['GarageArea']

y = housePropertyDataset['SalePrice']

plt.hist2d(x,y, bins=(10,20), range=((0, 1500), (0, 700000)), cmap='viridis')



plt.colorbar()



plt.xlabel('Garbage Area')

plt.ylabel('Property Sales Price')



plt.tight_layout()

plt.show()
plt.scatter(housePropertyDataset['GarageArea'],housePropertyDataset['SalePrice'], label='data', 

            color='red', marker='o')



sns.regplot(x='GarageArea', y='SalePrice', data=housePropertyDataset

            , scatter=None, color='blue', label='order 1', order=1)



sns.regplot(x='GarageArea', y='SalePrice', data=housePropertyDataset

            , scatter=None, color='green', label='order 2', order=2)

plt.legend(loc='lower right')

plt.show()

x = housePropertyDataset['GarageArea']

y = housePropertyDataset['SalePrice']

plt.hexbin(x,y, gridsize=(15,10),extent=(0, 1500, 0, 700000), cmap='winter')



plt.colorbar()



plt.xlabel('Garbage Area')

plt.ylabel('Sales Price')

plt.show()



# Looking at output of the graph, can we say males are more educated from female?
# Plot a linear regression between 'GarageArea' and 'SalePrice'

sns.lmplot(x='GarageArea', y='SalePrice', data=housePropertyDataset, 

           col='Street') # We can also use 'hue' parameter instead of col parameter to plt on the same graph



# Display the plot

plt.show()

sns.residplot(x='GarageArea', y='SalePrice', data=housePropertyDataset, color='blue')



# Display the plot

plt.show()
u = list(range(1, 10))

v = list(range(11, 20))

X,Y = np.meshgrid(u,v)

Z  = 3*np.sqrt(X**2 + Y**2)



plt.subplot(2,1,1)

plt.contour(X, Y, Z)



plt.subplot(2,1,2)

plt.contour(X, Y, Z, 20) # 20 contour





plt.show()
u = list(range(1, 10))

v = list(range(11, 20))

X,Y = np.meshgrid(u,v)

Z  = 3*np.sqrt(X**2 + Y**2)



plt.subplot(2,1,1)

plt.contourf(X, Y, Z)



plt.subplot(2,1,2)

plt.contourf(X, Y, Z, 20, cmap='winter') # 20 contour will be mapped





plt.show()
plt.subplot(1,2,1)

sns.stripplot(x='Street', y='SalePrice', data=housePropertyDataset, jitter=True, size=3)

plt.xticks(rotation=90)



plt.subplot(1,2,2)

sns.stripplot(x='Neighborhood', y='SalePrice', data=housePropertyDataset, jitter=True, size=3)

plt.xticks(rotation=90)



plt.subplots_adjust(right=3)

plt.show()

plt.subplot(1,2,1)

sns.swarmplot(x='Street', y='SalePrice', data=housePropertyDataset, hue='SaleCondition')

plt.xticks(rotation=90)



plt.subplot(2,2,2)

sns.stripplot(x='Neighborhood', y='SalePrice', data=housePropertyDataset, hue='SaleCondition')

plt.xticks(rotation=90)



plt.subplots_adjust(right=3)

plt.show()

plt.subplot(2,1,1)

sns.violinplot(x='SaleType', y='SalePrice', data=housePropertyDataset)



plt.subplot(2,1,2)

sns.violinplot(x='SaleType', y='SalePrice', data=housePropertyDataset, color='lightgray', inner=None)



sns.stripplot(x='SaleType', y='SalePrice', data=housePropertyDataset, jitter=True, size=1.5)



plt.show()
fig = plt.figure(figsize = (16,89))

sns.jointplot(x='GarageArea', y='SalePrice', data=housePropertyDataset)

plt.xticks(rotation=90)

plt.show()
data = housePropertyDataset[['SaleCondition', 'SalePrice','OverallQual',

                             'TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']]

sns.pairplot(data)

plt.show()
numeric_features = housePropertyDataset.select_dtypes(include=[np.number])

sns.heatmap(numeric_features.corr())

plt.title('Correlation heatmap')

plt.show()
sns.boxplot(x='Street', y='SalePrice', data=housePropertyDataset)

plt.title('Sale price by Street')

plt.show()
def ecdf(data):

    # Number of data points: n

    n = len(data)

    # x-data for the ECDF: x

    x = np.sort(data)

    # y-data for the ECDF: y

    y = np.arange(1, n+1) / n

    return x, y
SalePrice = housePropertyDataset['SalePrice']

x_vers, y_vers = ecdf(SalePrice)



plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')



plt.ylabel('ECDF')



plt.show()