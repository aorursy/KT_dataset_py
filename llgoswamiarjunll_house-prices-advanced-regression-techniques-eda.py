import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
#%matplotlib notebook
%matplotlib inline
import seaborn as sns
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
pd.set_option('max_columns', 500)
data.head()
data.columns
#Looking at the fields in the data description, we select to visualize the following fields by intuition:
#Aesthetics of the house - LotArea, BldgType, HouseStyle, LandSlope, OverallQual, OverallCond, YearBuilt, YearRemodAdd, HeatingQC, CentralAir, KitchenQual, GarageType
#Locality - Neighborhood, Condition1
fig = plt.figure()
fig.set_size_inches(40, 40, forward=True)
#plt.rcParams['figure.figsize'] = (20,20)
#plt.rcParams['figure.figsize'] = [10, 5]
p1 = fig.add_subplot(4,2,1)
p1.hist(bins=100,x=data.SalePrice,range=(0,400000),histtype='bar')
plt.xlabel('SalePrice')
plt.ylabel('No. of houses sold')
plt.title('Freq Distribution of sale by price')

p2 = fig.add_subplot(4,2,2)
p2.scatter(y=data.SalePrice[data.LotArea<100000], x=data.LotArea[data.LotArea<100000],s=10)
#p2.plot(y=data.SalePrice, x=data.LotArea,secondary_y = True)
plt.xlim = 100000
plt.ylabel('SalePrice')
plt.xlabel('LotArea')
plt.title('Identifying relation between LotArea and SalePrice')

p3 = fig.add_subplot(4,2,3)
p3.scatter(y=data.SalePrice, x=data.OverallQual,s=5)
plt.ylabel('SalePrice')
plt.xlabel('OverallQual')
plt.title('Identifying relation between Overall Quality and SalePrice')

p4 = fig.add_subplot(4,2,4)
p4.scatter(y=data.SalePrice, x=data.YearBuilt,s=10)
plt.ylabel('SalePrice')
plt.xlabel('YearBuilt')
plt.title('Identifying relation between Overall Quality and SalePrice')

p5 = fig.add_subplot(4,2,5)
p5.bar(left=data.Neighborhood, height=data.SalePrice)
plt.xticks(rotation = 90)
plt.xlabel('Neighborhood')
plt.ylabel('SalePrice')
plt.title('Identifying relation between Neighborhood and SalePrice')

p6 = fig.add_subplot(4,2,6)
p6.hist(data.Neighborhood, bins=50)
plt.xticks(rotation = 90)
plt.xlabel('Neighborhood')
plt.ylabel('No. of houses sold')
plt.title('Identifying relation between Neighborhood no. of houses sold')

p7 = fig.add_subplot(4,2,7)
p7 = sns.violinplot(y=data['SalePrice'],x=data['Neighborhood'])
plt.xticks(rotation = 90)
plt.xlabel('Neighborhood')
plt.ylabel('SalePrice')
plt.title('Distribution of SalePrice by Neigborhood')

p7 = fig.add_subplot(4,2,8)
p7 = sns.violinplot(y=data['SalePrice'],x=data['Heating'])
plt.xticks(rotation = 90)
plt.xlabel('Heating')
plt.ylabel('SalePrice')
plt.title('Distribution of SalePrice by Heating')

plt.show()
data_1 = data[['Neighborhood','LotArea','SalePrice','Heating','YearBuilt']]
sns.pairplot(data_1)