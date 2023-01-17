import numpy as np

import pandas as pd

import math

import scipy.spatial

import os

import seaborn as sns

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt;
pd.DataFrame({

    'Customers': ['A', 'B', 'C'],

    'Bottles': [1, 1, 1],

    'Nails': [1, 1, 1]

})
pd.DataFrame({

    'Customers': ['A', 'B', 'C'],

    'Bottles': [4, 1, 5],

    'Nails': [1, 2, 1]

})
def showVector(vector):

    plt.figure(figsize = (100,100))

    plt.imshow(vector, aspect = 100)



def sigmoid(x):

  return 1 / (1 + math.exp(-x))

sigmoid = np.vectorize(sigmoid)
data = pd.read_csv(

    '/kaggle/input/ecommerce-data/data.csv',

    encoding = "ISO-8859-1",

    dtype = {'CustomerID': str, 'InvoiceID': str}

)



data
customers = data.CustomerID.unique()

products = data.StockCode.unique()

numberOfCustomers = len(customers)

numberOfProducts = len(products)

pd.DataFrame([{

    'Customers': numberOfCustomers,

    'Products': numberOfProducts

}], index=['quantity'])
productTransactionCounts = data.StockCode.value_counts()

productPurchaseFrequency = pd.DataFrame(data = {

    'Product': productTransactionCounts.index.values,

    '#Transactions': productTransactionCounts.values

})

productPurchaseFrequency.head()
productQuantityTotals = data.groupby('StockCode')['Quantity'].sum()

productQuantityTotals = pd.DataFrame(data = {

    'Product': productQuantityTotals.index.values,

    'QuantitySum': productQuantityTotals.values

})

productQuantityTotals.sort_values(by='QuantitySum', ascending=False).head()
customerQuantityTotals = data.groupby('CustomerID')['Quantity'].sum()

customerQuantityTotals = pd.DataFrame(data = {

    'Customer': customerQuantityTotals.index.values,

    'QuantitySum': customerQuantityTotals.values

})

customerQuantityTotals.sort_values(by='QuantitySum', ascending=False)
numberOfCancellations = len(data[data.Quantity < 0])

numberOfKeptTransactions = len(data) - numberOfCancellations

percentageOfCancellations = round(numberOfCancellations / len(data) * 100, 2)



fig, ax = plt.subplots()

ax.pie(

    [numberOfKeptTransactions, numberOfCancellations],

    labels=(

        f'Retained Transactions: #{numberOfKeptTransactions}',

        f'Cancelled Transactions {percentageOfCancellations}%: #{numberOfCancellations}'

    ),

    startangle=90

)

plt.show()
# Remove all cancellations

data = data[data.Quantity > 0];
customerProductMatrix = data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0, aggfunc='sum')

customerProductMatrix
numberOfCustomerProductRelations = customerProductMatrix.size

numberOfCustomerProductRelationsFillfilled = np.count_nonzero(customerProductMatrix)

sparsity = 1 - (numberOfCustomerProductRelationsFillfilled / numberOfCustomerProductRelations)

density = 1 - sparsity

pd.DataFrame([{

    '#CustomerProductRelations': numberOfCustomerProductRelations,

    '#CustomerProductRelationsFullfilled': numberOfCustomerProductRelationsFillfilled,

    'Sparsity': sparsity,

    'Density': density

}], index=['quantity'])
matrix = sigmoid(customerProductMatrix)

plt.figure(figsize = (100,50))

plt.imshow(matrix);
sampleCustomerID = '18177'

sampleCustomer = np.asarray(customerProductMatrix.loc[sampleCustomerID])

sampleCustomer = np.reshape(sampleCustomer, [1, sampleCustomer.shape[0]])

showVector(sampleCustomer)
sampleCustomerIndex = customerProductMatrix.index.get_loc(sampleCustomerID)

distances = scipy.spatial.distance.cdist(customerProductMatrix, sampleCustomer, metric='euclidean')

distances[sampleCustomerIndex][0] = distances.mean()

bestMatchingCustomer = customerProductMatrix[distances == distances.min()]

bestMatchIndex = distances.argmin()



showVector(np.reshape(distances, [1, distances.shape[0]]))
fig = plt.figure(figsize=(100,20))

plt.plot(distances)

plt.axvline(bestMatchIndex, color='k', linestyle='dashed', linewidth=2);
showVector(sampleCustomer)
showVector(bestMatchingCustomer)
diff = np.subtract(bestMatchingCustomer, sampleCustomer)

diff[diff < 0] = 0

showVector(diff)
customerProductMatrix['Distances'] = distances

customerProductMatrix = customerProductMatrix.sort_values('Distances')

customerProductMatrix.head()
del customerProductMatrix['Distances']

bestMatchingCustomers = customerProductMatrix.head()

bestMatchingCustomerBinaries = np.where(bestMatchingCustomers > 0, 1, 0)

consensus = np.expand_dims(np.sum(bestMatchingCustomerBinaries, axis=0), 0)

consensus[sampleCustomer > 0] = 0

showVector(consensus)
customerProductMatrix.loc['Rank'] = np.squeeze(consensus, 0)

customerProductMatrix = customerProductMatrix.T.sort_values('Rank', ascending=False)

recommendedProducts = customerProductMatrix[customerProductMatrix['Rank'] > 0]

recommendedProducts
recommendedProducts = [ recommendedProducts['Rank'] ]

recommendedProducts = pd.DataFrame(recommendedProducts).T

recommendedProductsTop = recommendedProducts.head(20)

recommendedProductsTop
data.loc[data['StockCode'].isin(recommendedProductsTop.index.tolist())]['Description'].unique()
data.loc[data['CustomerID'] == sampleCustomerID]['Description']