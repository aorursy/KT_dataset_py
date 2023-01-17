import numpy as np

import pandas as pd
# Load data

sku_category_filepath = '../input/sku-category/sku_category.csv'

sku_category = pd.read_csv(sku_category_filepath,sep=None,decimal=',',engine='python')



# These columns are not relevant according to the problem description

sku_category.drop('compare', axis=1,inplace=True)

sku_category.drop('sector', axis=1,inplace=True)



# This datafram will contain all of the data

sku = sku_category.copy()



for i in range(1,32):

    if i<10:

        date = '2017-03-0{}'.format(i)

    else:

        date = '2017-03-{}'.format(i)

    

    sku_price_filepath = '../input/sku-price/{}.csv'.format(date)

    sku_price = pd.read_csv(sku_price_filepath,sep=None,decimal=',',engine='python')

    sku_price.columns = ['sku','price-{}'.format(date)]

    

    sku_sold_filepath = '../input/sku-unitssold/{}.csv'.format(date)

    sku_sold = pd.read_csv(sku_sold_filepath,sep=None,decimal=',',engine='python')

    sku_sold.columns = ['sku','sold-{}'.format(date)]

    

    # Manual data cleaning

    sku_price['sku'] = sku_price['sku'].astype(str)

    sku_price.drop(sku_price[sku_price['sku'] == 'S080501_500_30_EUR'].index.values, axis=0, inplace = True)

    sku_price.drop(sku_price[sku_price['sku'] == 'S080501_1500_30_EUR'].index.values, axis=0, inplace = True)

    sku_price['sku'] = pd.to_numeric(sku_price['sku'])

    

    sku_sold['sku'] = sku_sold['sku'].astype(str)

    sku_sold.drop(sku_sold[sku_sold['sku'] == 'S080501_500_30_EUR'].index.values, axis=0, inplace = True)

    sku_sold.drop(sku_sold[sku_sold['sku'] == 'S080501_1500_30_EUR'].index.values, axis=0, inplace = True)

    sku_sold['sku'] = pd.to_numeric(sku_sold['sku'])



    sku = pd.merge(sku, sku_price, on='sku', how='left')

    sku = pd.merge(sku, sku_sold, on='sku', how='left')

    

sku.head()
sku.sort_values(by='sold-2017-03-01').head(20)
print('We have {} rows (products) in our table'.format(sku.shape[0]))

sumProductsSold=0

for i in range(1,32):

    if i<10:

        date = '2017-03-0{}'.format(i)

    else:

        date = '2017-03-{}'.format(i)

        

    if sku['price-{}'.format(date)].count() != sku['sold-{}'.format(date)].count():

        print('The price and sold columns do not have the same number of entries for {}'.format(date))

        

    print('{}: {} products sold'.format(date,sku['price-{}'.format(date)].count()))

    sumProductsSold+=sku['price-{}'.format(date)].count()

    

print('At maximum, out of the {} products, {} were sold in the given data'.format(sku.shape[0], sumProductsSold))
sku_numberSold = sku.copy()

columnsToDrop = [x for x in sku.columns if x != 'sku']

sku_numberSold.drop(columnsToDrop, axis=1, inplace=True)

sku_numberSold.insert(1,'avgNumberSold',0.)

for i in range(1,32):

    if i<10:

        date = '2017-03-0{}'.format(i)

    else:

        date = '2017-03-{}'.format(i)

        

    for index in sku.index:

        number_sold = sku.at[index, 'sold-{}'.format(date)]

        if (number_sold > 0):

            sku_numberSold.at[index, 'avgNumberSold'] += number_sold



sku_numberSold['avgNumberSold'] = sku_numberSold['avgNumberSold']/31

sku_numberSold.sort_values('avgNumberSold',ascending=False).head(20)
sku_probability = sku_numberSold.copy()   

sku_probability.rename(columns={'avgNumberSold':'probability'}, inplace=True)

for index in sku_probability.index:

    probability = np.round(1-np.exp(-sku_probability.at[index, 'probability']),5)

    sku_probability.at[index, 'probability'] = probability

        

sku_probability.sort_values(by='probability', ascending=False).head(20)
output = sku_probability

output.to_csv('OptimusPrice_Problem1.csv',index=False)