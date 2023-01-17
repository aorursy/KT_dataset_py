import os

import pandas as pd
os.listdir("../input")
data = pd.read_csv(os.path.join("../input","train.csv"))
data.shape       #shape of the training set
data.head(5)
data.columns
data = data.drop(columns = ['Id'])
data.info()
data.isnull().sum()
num_features = data._get_numeric_data().columns  #numeric features may or may not be categorical

cat_features = list(set(data.columns) - set(num_features))
len(num_features)
import matplotlib.pyplot as plt

import seaborn as sns
sns.kdeplot(data['SalePrice'])
plt.figure(figsize = (50,40))

for i in range(1,len(num_features)):

    plt.subplot(8,5,i)

    plt.title(num_features[i])

    sns.kdeplot(data[num_features[i]])

data.describe()
plt.figure(figsize = (80,30))

for i in range(1,len(num_features)):

    plt.subplot(5,8,i)

    plt.title(num_features[i])

    sns.boxplot(data[num_features[i]], orient = "v")
data[['YearBuilt','YearRemodAdd','YrSold','SalePrice']].sample(10).sort_values(by=['YearBuilt'])
x = data.loc[data['YearBuilt']>=2000,['YearBuilt']].sort_index(ascending = False)

x.count()   #number of houses built between 2000 and 2010 out of 1460
data['YearBuilt'].unique()
data['YearBuilt'].value_counts().sort_index().plot()
data.loc[data['YearBuilt']>=2000,['YearBuilt','YrSold','SalePrice']].sample(10)
d = data[['YrSold','SalePrice']].sort_values(by=['YrSold'])

d.sample(5)
d['YrSold'].unique()
d1 = d.groupby(['YrSold'])

d1.describe()
plt.plot(d1.mean())

plt.title('Mean Sale Price vs Year Sold')

plt.xlabel('YrSold')

plt.ylabel('MeanSalePrice')
m = data[['MoSold','SalePrice']].sort_values(by=['MoSold']).groupby(['MoSold'])

m.describe()
plt.plot(m.mean())

plt.title('Mean Sale Price vs Month Sold')

plt.xlabel('Month Sold')

plt.ylabel('Mean Sale Price')
b = data[['BedroomAbvGr','1stFlrSF','2ndFlrSF','SalePrice']].sort_values(by=['BedroomAbvGr'])

b['SF_per_bed'] = (b['1stFlrSF']+b['2ndFlrSF'])/b['BedroomAbvGr']

b.groupby(['BedroomAbvGr']).describe()
x = b.groupby(['BedroomAbvGr'])

x = x.mean()[x.count()>50]     #excluding some of the outliers

x = x.loc[x['SalePrice']>0]

x
plt.plot(x['SalePrice'])

plt.xlabel('BedroomAbvGr')

plt.ylabel('MeanSalePrice')
b1 = x.sort_values(by=['SF_per_bed'])

b1
data[['ScreenPorch','SalePrice']].sample(12)
data['ScreenPorch'].unique()
data['ScreenPorch'].value_counts().head()
data_life = data.loc[:,['YearRemodAdd','SalePrice']]    

    #num_of_years is the age of the house at the time of selling

data_life['num_of_years'] = data['YrSold'] - data['YearBuilt']

data_life.sample(7)
dl = data_life[['num_of_years','SalePrice']].sort_values(by=['num_of_years']).groupby(['num_of_years'])

dl.describe().sample(5)
x = dl.mean()[dl.count()>5]

x = x.loc[x['SalePrice']>0]

plt.plot(x)

plt.title('Mean Sale Price vs Age of House')

plt.xlabel('AgeofHouse')

plt.ylabel('MeanSalePrice')
corr = data.corr()

corr
plt.figure(figsize = (32,30))

sns.heatmap(corr,cmap = 'nipy_spectral', annot = True)
x = data.loc[:,['OverallQual','SalePrice']].groupby(['OverallQual'])

x.describe()
x = x.mean()[x.count()>5]

x = x.loc[x['SalePrice']>0]
plt.plot(x)

plt.title('MeanSalePrice vs OverallQual')

plt.xlabel('Overall Quality')

plt.ylabel('MeanSalePrice')
plt.figure(figsize = (50,40))

for i in range(0,len(cat_features)):

    plt.subplot(8,6,i+1)

    #plt.title(cat_features[i])

    sns.countplot(data[cat_features[i]])



#double click images to zoom
data['Utilities'].describe()
data['RoofMatl'].describe()
data['Heating'].describe()
data_above_mean_sp = data.loc[data['SalePrice']>=181000]

data_below_mean_sp = data.loc[data['SalePrice']<181000]
data_above_mean_sp.shape
data_below_mean_sp.shape
int_col = ['Neighborhood','Foundation','Exterior2nd','HouseStyle','KitchenQual']
plt.figure(figsize = (25,50))

i = 0

print('\tSale Price above 181000\t\t\t\t\t\tSale Price below 181000')

for col in int_col:

    i = i + 1

    plt.subplot(5,2,i)

    label_am = data_above_mean_sp[col].unique()

    size_am = data_above_mean_sp[col].value_counts()

    #print(size_am, label_am)

    plt.pie(size_am,labels = label_am, autopct = '%1.1f%%',shadow = True,startangle = 0)

    plt.legend()

    plt.title(col)

    i = i + 1

    plt.subplot(5,2,i)

    label_bm = data_below_mean_sp[col].unique()

    size_bm = data_below_mean_sp[col].value_counts()

    #print(size_bm)

    plt.pie(size_bm, labels = label_bm,autopct = '%1.1f%%', shadow = True, startangle = 0)

    plt.legend()

    plt.title(col)
x = data[['KitchenQual','KitchenAbvGr','SalePrice']].sort_values(['SalePrice'])

x.sample(8)
data['KitchenQual'].value_counts()
data['KitchenAbvGr'].value_counts()
x = x.drop(['KitchenAbvGr'], axis = 1)
x = x.groupby(['KitchenQual'])

x.describe()
plt.plot(x.mean()['SalePrice'])
data['MSZoning'].value_counts()
x = data.loc[data['GrLivArea']<3000,['GrLivArea','SalePrice']].sort_values('GrLivArea')

x.sample(6)
sns.scatterplot('GrLivArea','SalePrice',data = x)
data[['OverallCond','SalePrice']].sample(5)