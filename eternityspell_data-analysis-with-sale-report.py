# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
saleReport = pd.read_csv('../input/SalesReport.csv')

saleReport = saleReport[saleReport['Product Code'] != 'SHIP']

saleReport = saleReport[saleReport['Product Code'] != 'DISCOUNT']
def getProductFromReport(saleReportinput):

    productCodeList = saleReportinput['Product Code'].unique()

    productSellQuantityList = []

    for productCode in productCodeList:

        total = 0

        productSaleQuantity = saleReportinput[saleReportinput['Product Code'] == productCode]['Order Qty']

        for productQuantity in productSaleQuantity:

            total += int(productQuantity.split('.')[0].replace(',', ''))

        productSellQuantityList.append(total)

    productSellQuantityForThisPeriod = pd.DataFrame(columns=['Product Code', 'Sell Quantity'])

    productSellQuantityForThisPeriod['Product Code'], productSellQuantityForThisPeriod['Sell Quantity'] = productCodeList, productSellQuantityList

    

    return productSellQuantityForThisPeriod
productSellQuantityForAYear = getProductFromReport(saleReport)

productSellQuantityForAYear = productSellQuantityForAYear.sort_values(by='Sell Quantity', ascending=False)

top100bestsellProduct = productSellQuantityForAYear.head(100)

sellLessThan5 = productSellQuantityForAYear[productSellQuantityForAYear['Sell Quantity'] <= 5]

top100bestsellProduct.to_csv('top100bestsellProduct.csv')

sellLessThan5.to_csv('sellLessThan5.csv')
firstMonthSaleReport = saleReport[saleReport['Order Date'] <= 20180418.0]

firstMonthSaleReport = firstMonthSaleReport[firstMonthSaleReport['Order Date'] >= 20180318.0]

secondMonthSaleReport = saleReport[saleReport['Order Date'] <= 20180518.0]

secondMonthSaleReport = secondMonthSaleReport[secondMonthSaleReport['Order Date'] >= 20180418.0]

thirdMonthSaleReport = saleReport[saleReport['Order Date'] <= 20180618.0]

thirdMonthSaleReport = thirdMonthSaleReport[thirdMonthSaleReport['Order Date'] >= 20180518.0]

forthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20180718.0]

forthMonthSaleReport = forthMonthSaleReport[forthMonthSaleReport['Order Date'] >= 20180618.0]

fifthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20180818.0]

fifthMonthSaleReport = fifthMonthSaleReport[fifthMonthSaleReport['Order Date'] >= 20180718.0]

sixthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20180918.0]

sixthMonthSaleReport = sixthMonthSaleReport[sixthMonthSaleReport['Order Date'] >= 20180818.0]

seventhMonthSaleReport = saleReport[saleReport['Order Date'] <= 20181018.0]

seventhMonthSaleReport = seventhMonthSaleReport[seventhMonthSaleReport['Order Date'] >= 20180918.0]

eighthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20181118.0]

eighthMonthSaleReport = eighthMonthSaleReport[eighthMonthSaleReport['Order Date'] >= 20181018.0]

ninthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20181218.0]

ninthMonthSaleReport = ninthMonthSaleReport[ninthMonthSaleReport['Order Date'] >= 20181118.0]

tenthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20190118.0]

tenthMonthSaleReport = tenthMonthSaleReport[tenthMonthSaleReport['Order Date'] >= 20181218.0]

eleventhMonthSaleReport = saleReport[saleReport['Order Date'] <= 20190218.0]

eleventhMonthSaleReport = eleventhMonthSaleReport[eleventhMonthSaleReport['Order Date'] >= 20190118.0]

twelfthMonthSaleReport = saleReport[saleReport['Order Date'] <= 20190318.0]

twelfthMonthSaleReport = twelfthMonthSaleReport[twelfthMonthSaleReport['Order Date'] >= 20190218.0]



firstMonthSellDistribution = getProductFromReport(firstMonthSaleReport)

secondMonthSellDistribution = getProductFromReport(secondMonthSaleReport)

thirdMonthSellDistribution = getProductFromReport(thirdMonthSaleReport)

forthMonthSellDistribution = getProductFromReport(forthMonthSaleReport)

fifthMonthSellDistribution = getProductFromReport(fifthMonthSaleReport)

sixthMonthSellDistribution = getProductFromReport(sixthMonthSaleReport)

seventhMonthSellDistribution = getProductFromReport(seventhMonthSaleReport)

eighthMonthSellDistribution = getProductFromReport(eighthMonthSaleReport)

ninthMonthSellDistribution = getProductFromReport(ninthMonthSaleReport)

tenthMonthSellDistribution = getProductFromReport(tenthMonthSaleReport)

eleventhMonthSellDistribution = getProductFromReport(eleventhMonthSaleReport)

twelfthMonthSellDistribution = getProductFromReport(twelfthMonthSaleReport)
from functools import reduce

productSellDistributionLastYear = reduce(lambda x, y: pd.merge(x, y, on='Product Code', how='outer'), [firstMonthSellDistribution, secondMonthSellDistribution, thirdMonthSellDistribution, forthMonthSellDistribution, fifthMonthSellDistribution, sixthMonthSellDistribution, seventhMonthSellDistribution, eighthMonthSellDistribution, ninthMonthSellDistribution, tenthMonthSellDistribution, eleventhMonthSellDistribution, twelfthMonthSellDistribution])

productSellDistributionLastYear.fillna(0, inplace=True)

productSellDistributionLastYear.columns = ['Product Code', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th']



kwfrf10bk = productSellDistributionLastYear[productSellDistributionLastYear['Product Code'] == 'KWFRF10BK']

productSellDistributionEachThreeMonthLastYear = pd.DataFrame(columns=['Product Code', '1st 3 month', '2nd 3 month', '3rd 3 month', '4th 3 month'])

productSellDistributionEachThreeMonthLastYear['Product Code'] = productSellDistributionLastYear['Product Code'] 

productSellDistributionEachThreeMonthLastYear['1st 3 month'] = productSellDistributionLastYear['1st'] + productSellDistributionLastYear['2nd'] + productSellDistributionLastYear['3rd']

productSellDistributionEachThreeMonthLastYear['2nd 3 month'] = productSellDistributionLastYear['4th'] + productSellDistributionLastYear['5th'] + productSellDistributionLastYear['6th']

productSellDistributionEachThreeMonthLastYear['3rd 3 month'] = productSellDistributionLastYear['7th'] + productSellDistributionLastYear['8th'] + productSellDistributionLastYear['9th']

productSellDistributionEachThreeMonthLastYear['4th 3 month'] = productSellDistributionLastYear['10th'] + productSellDistributionLastYear['11th'] + productSellDistributionLastYear['12th']



productSellDistributionhalfYearLastYear =  pd.DataFrame(columns=['Product Code', 'first half year', 'next half year'])

productSellDistributionhalfYearLastYear['Product Code'] = productSellDistributionLastYear['Product Code'] 

productSellDistributionhalfYearLastYear['next half year'] = productSellDistributionLastYear['4th'] + productSellDistributionLastYear['5th'] + productSellDistributionLastYear['6th'] + productSellDistributionLastYear['7th'] + productSellDistributionLastYear['8th'] + productSellDistributionLastYear['9th'] 

productSellDistributionhalfYearLastYear['first half year'] = productSellDistributionLastYear['10th'] + productSellDistributionLastYear['11th'] + productSellDistributionLastYear['12th'] + productSellDistributionLastYear['1st'] + productSellDistributionLastYear['2nd'] + productSellDistributionLastYear['3rd']



productSellDistributionLastYear.to_csv('productSellDistributionLastYear.csv')

productSellDistributionEachThreeMonthLastYear.to_csv('productSellDistributionEachThreeMonthLastYear.csv')

productSellDistributionhalfYearLastYear.to_csv('productSellDistributionhalfYearLastYear.csv')

products = []

variances = []

for productCode in productSellDistributionLastYear['Product Code']:

    row = productSellDistributionLastYear[productSellDistributionLastYear['Product Code'] == productCode]

    variance = np.var([int(row['1st']), int(row['2nd']), int(row['3rd']), int(row['4th']), int(row['5th']), int(row['6th']), int(row['7th']), int(row['8th']), int(row['9th']), int(row['10th']), int(row['11th']), int(row['12th'])])

    products.append(productCode)

    variances.append(variance)

productVarianceDistributionLastYear = pd.DataFrame(columns=['Product Code', 'Variances'])

productVarianceDistributionLastYear['Product Code'], productVarianceDistributionLastYear['Variances'] = products, variances



productVarianceDistributionLastYear.sort_values(by='Variances', inplace=True, ascending=False)
productsInventory = pd.read_csv('../input/inventory.csv')
productsInventory
condition1Passed = productsInventory[productsInventory['Quantity in Stock'] == '0.00']

productSellDistributionLastYear
productVarianceDistributionLastYear
bigVarianceProduct = []

for productCode in productSellDistributionLastYear['Product Code']:

    row = productSellDistributionLastYear[productSellDistributionLastYear['Product Code'] == productCode]

    sellList = list([int(row['1st']), int(row['2nd']), int(row['3rd']), int(row['4th']), int(row['5th']), int(row['6th']), int(row['7th']), int(row['8th']), int(row['9th']), int(row['10th']), int(row['11th']), int(row['12th'])])

    if int(productVarianceDistributionLastYear[productVarianceDistributionLastYear['Product Code'] == productCode]['Variances']) ** 2 > 1.8:

        bigVarianceProduct.append(productCode)
np.var([2, 0, 0, 0, 2, 1, 1, 3, 0, 1, 1, 0])




inventoryMayShortageList = productSellDistributionLastYear.loc[productSellDistributionLastYear['Product Code'].isin(bigVarianceProduct)]
bigVarianceProduct = []

productCodesList = []

monthRequiredList = []

for productCode in inventoryMayShortageList['Product Code']:

    row = inventoryMayShortageList[inventoryMayShortageList['Product Code'] == productCode]

    sellList = list([int(row['1st']), int(row['2nd']), int(row['3rd']), int(row['4th']), int(row['5th']), int(row['6th']), int(row['7th']), int(row['8th']), int(row['9th']), int(row['10th']), int(row['11th']), int(row['12th'])])

    monthRequired = 12

    for i in range(1, len(sellList)):

        if sellList[i] == 0:

            j = i - 1

            while sellList[j] == 0:

                j -= 1

            if sellList[j] > 5:

                monthRequired -= 1

    

    productCodesList.append(productCode)

    monthRequiredList.append(monthRequired)

    

shortageProductwithMonthRequired = pd.DataFrame(columns=['Product Code', 'Month Required'])

shortageProductwithMonthRequired['Product Code'] = productCodesList

shortageProductwithMonthRequired['Month Required'] = monthRequiredList

                
shortageProductwithMonthRequired = shortageProductwithMonthRequired[shortageProductwithMonthRequired['Month Required'] != 12]
shortageProductwithMonthRequired
averageSales = []

for productCode in productSellDistributionLastYear['Product Code']:

    row = productSellDistributionLastYear[productSellDistributionLastYear['Product Code'] == productCode]

    saleQuantityforAyear = sum([int(row['1st']), int(row['2nd']), int(row['3rd']), int(row['4th']), int(row['5th']), int(row['6th']), int(row['7th']), int(row['8th']), int(row['9th']), int(row['10th']), int(row['11th']), int(row['12th'])])

    averageSale = 0

    if productCode in shortageProductwithMonthRequired['Product Code']:

        averageSale = float(saleQuantityforAyear) / shortageProductwithMonthRequired[shortageProductwithMonthRequired['Product Code'] == productCode] / 4

        averageSales.append(averageSale)

    else:

        averageSale = float(saleQuantityforAyear) / 12 / 4

        averageSales.append(averageSale)



        



productSellPerWeek = pd.DataFrame(columns=['Product Code', 'Sell per Week'])

productSellPerWeek['Product Code'] = productSellDistributionLastYear['Product Code']

productSellPerWeek['Sell per Week'] = averageSales

    
productSellPerWeek
productSellPerWeek = productSellPerWeek[productSellPerWeek['Sell per Week'] != 0.000000]
productsInventory
renamedProducts = []

for product in productSellPerWeek['Product Code']:

    quantity = productsInventory[productsInventory['Product Code'] == product]['Quantity in Stock']

    if quantity.empty:

        renamedProducts.append(product)

    
remainingWeeks = []

productsCodeList = []

productSellPerWeekList = []

productStockList = []

for product in productSellPerWeek['Product Code']:

    if product not in renamedProducts:

        remainingWeeks.append(int(float(productsInventory[productsInventory['Product Code'] == product]['Quantity in Stock'].values[0].replace(',', '')) / productSellPerWeek[productSellPerWeek['Product Code'] == product]['Sell per Week'].values[0]))

        productsCodeList.append(product)

        productSellPerWeekList.append(productSellPerWeek[productSellPerWeek['Product Code'] == product]['Sell per Week'].values[0])

        productStockList.append(float(productsInventory[productsInventory['Product Code'] == product]['Quantity in Stock'].values[0].replace(',', '')))

productsStockSellingRateDistribution = pd.DataFrame(columns=['Product Code','Quantity in Stock', 'Sell per Week', 'Remaining Weeks'])

productsStockSellingRateDistribution['Product Code'] = productsCodeList

productsStockSellingRateDistribution['Sell per Week'] = productSellPerWeekList

productsStockSellingRateDistribution['Quantity in Stock'] = productStockList

productsStockSellingRateDistribution['Remaining Weeks'] = remainingWeeks
productsStockSellingRateDistribution = productsStockSellingRateDistribution.sort_values(by='Remaining Weeks')
productsStockSellingRateDistribution.to_csv('productsStockSellingRateDistribution.csv')