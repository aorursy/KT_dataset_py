import pandas as pd
import numpy as np
from sklearn import preprocessing
file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter03/bank-full.csv'

# Reading the banking data
bankData = pd.read_csv(file_url,sep=";")
bankData.head()
# Normaliz0ing data
# Get values from column Balance
x = bankData[['balance']].values.astype(float)

# Transform the column by normalizing it with minmaxScaler:
minmaxScaler = preprocessing.MinMaxScaler()
bankData['balanceTran'] = minmaxScaler.fit_transform(x)

bankData.head()
# Adding a small numerical constant to eliminate 0 values
bankData['balanceTran'] = bankData['balanceTran'] + 0.00001
# Let us transform values for loan data
bankData['loanTran'] = 1

# Giving a weight of 5 if there is no loan
bankData.loc[bankData['loan'] == 'no', 'loanTran'] = 5

bankData.head()
# Let us transform values for Housing data
bankData['houseTran'] = 5

# Giving a weight of 1 if there is no house
bankData.loc[bankData['housing'] == 'no', 'houseTran'] = 1

bankData.head()
# Let us now create the new variable which is a product of all these
bankData['assetIndex'] = bankData['balanceTran'] * bankData['loanTran'] * bankData['houseTran']

bankData.head()
# Finding the quantile
q25, q50, q75 = np.quantile(bankData['assetIndex'],[0.25,0.5,0.75])

print('quantile values:', q25, ',', q50, ',', q75)
bankData['assetClass'] = 'Quant1'
bankData.loc[(bankData['assetIndex'] > 0.38) & (bankData['assetIndex'] < 0.57), 'assetClass'] = 'Quant2'
bankData.loc[(bankData['assetIndex'] > 0.57) & (bankData['assetIndex'] < 1.9), 'assetClass'] = 'Quant3'
bankData.loc[bankData['assetIndex'] > 1.9, 'assetClass'] = 'Quant4'

bankData.head()
# Calculating total of each asset class
assetTot = bankData.groupby('assetClass')['y'].agg(assetTot='count').reset_index()

assetTot.head()
# Calculating the category wise counts
assetProp = bankData.groupby(['assetClass', 'y'])['y'].agg(assetCat='count').reset_index()

assetProp.head(10)
# Merging both the data frames
assetComb = pd.merge(assetProp, assetTot, on = ['assetClass'])
assetComb['catProp'] = (assetComb.assetCat / assetComb.assetTot)*100

assetComb