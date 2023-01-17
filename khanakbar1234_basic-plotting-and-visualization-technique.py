import pandas as pd
import numpy as np
import json
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/house-price-prediction/house_prices.csv")
df.info()
df.describe().T
mask = df.isnull()
total = mask.sum()
percent = 100*mask.mean()
missing_data = pd.concat([total, percent], axis=1,join='outer',
               keys=['count_missing', 'perc_missing'])
missing_data.sort_values(by='perc_missing', ascending=False, inplace=True)
missing_data
nullable_columns = df.columns[mask.any()].tolist()
msno.matrix(df[nullable_columns].sample(500))
plt.show()
msno.heatmap(df[nullable_columns], figsize=(18,18))
plt.show()
#Delete the columns having more than 80% of values missing. Use the .loc operator on the DataFrame we created in 
#Step 2 to select only those columns that had fewer than 80% of their values missing:
data = df.loc[:,missing_data[missing_data.perc_missing < 80].index]
#Replace null values in the FireplaceQu column with NA values. Use the .fillna() method to replace null 
#values with the NA string:
data['FireplaceQu'] = data['FireplaceQu'].fillna('NA') 
data['FireplaceQu']

#import pyplot as plt
plt.figure(figsize=(8,6))
plt.hist(data.SalePrice, bins=range(0,800000,50000))
plt.ylabel('Number of Houses')
plt.xlabel('Sale Price')
plt.show()
object_variables = data.select_dtypes(include=[np.object])
object_variables.nunique().sort_values()
counts = data.HouseStyle.value_counts(dropna=False)
counts.reset_index().sort_values(by='index')
fig, ax = plt.subplots(figsize=(10,10))
slices = ax.pie(counts, 
                labels = counts.index, 
                colors = ['white'], 
                wedgeprops = {'edgecolor': 'black'} )
patches = slices[0]
hatches =  ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '\.', '*']
colors = ['white', 'white', 'lightgrey', 'white', 
          'lightgrey', 'white', 'lightgrey', 'white']
for patch in range(len(patches)):
    patches[patch].set_hatch(hatches[patch])
    patches[patch].set_facecolor(colors[patch])
plt.title('Pie chart showing counts for\nvarious house styles')
plt.show()
#Plot a histogram using seaborn for the LotArea variable. Use seaborn's .distplot() function as the primary 
#plotting function, to which the LotArea series in the DataFrame needs to be passed (without any null values, 
#use .dropna() on the series to remove them). To improve the plot view, also set the bins parameter and specify the 
#X-axis limits using plt.xlim():
plt.figure(figsize=(10,7)) 
sns.distplot(data.LotArea.dropna(), bins=range(0,100000,1000)) 
plt.xlim(0,100000) 
plt.show() 
#Calculate the skew and kurtosis values for the values in each column:
data.skew().sort_values()
data.kurt()
plt.figure(figsize = (12,10))
sns.heatmap(data.corr(), square=True, cmap="RdBu", vmin=-1, vmax=1)
plt.show()
#Plot a more compact heatmap having annotations for correlation values using the following subset of features:
feature_subset = [
    'GarageArea', 'GarageCars','GarageCond','GarageFinish','GarageQual','GarageType',
    'GarageYrBlt','GrLivArea','LotArea','MasVnrArea','SalePrice'
]
#Now do the same as in the previous step, this time selecting only the above columns in the dataset 
#and adding a parameter, annot, with a True value to the primary plotting function, with everything else remaining the same:
plt.figure(figsize = (12,10))
sns.heatmap(data[feature_subset].corr(), square=True, annot=True, cmap="RdBu", vmin=-1, vmax=1)
plt.show()
sns.pairplot(data[feature_subset].dropna(), kind ='scatter', diag_kind='kde')
plt.show()
plt.figure(figsize=(10, 10))
sns.boxplot(x='GarageCars', y="SalePrice", data=data)
plt.show()
plt.figure(figsize=(10,7))
sns.lineplot(x=data.YearBuilt, y=data.SalePrice, ci=None)
plt.show()



