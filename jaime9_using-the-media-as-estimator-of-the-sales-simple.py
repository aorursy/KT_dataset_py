# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load data (use only data from 2017)

train = pd.read_csv('../input/train.csv',skiprows=range(1, 101688780))



#Remove information about returned items

ind = train['unit_sales'] >= 0

train = train[ind]

train.reset_index(inplace=True) #reset the indexes to account for the removed rows

train['date'] = pd.to_datetime(train['date']) # make the date column a valid datetime object



stores = pd.read_csv('../input/stores.csv') # load stores information

items = pd.read_csv('../input/items.csv') #load information about items for sale







grouped = train.groupby(['item_nbr', 'store_nbr']) #group data by items and stores

grouped = grouped.groups # get the groups

groupedKeys = list(grouped.keys()) # save the indexes that forma the groups in a list





dates = pd.date_range(train['date'].iloc[0], train['date'].iloc[-1]) # generate all dates from january 01 of 2017

itemSales = np.zeros((len(dates),len(items),len(stores))) # matrix to save the item sales



#here we go throgh all the groups and organize the data in the three-dimensional matrix (Future kernels will use this)

for k in groupedKeys:

    indItem = np.where(items['item_nbr']==k[0]) # map the item_nbr to the index in the items.csv data

    indStore = k[1] # index for the store

    

    index = grouped[k] #indexes that form te group 'k'

    ind = dates.searchsorted(np.array(train['date'].iloc[index])) #every index is associated with a date, here we map the dates to our new format in the 3D matrix

    

    itemSales[ind,indItem,indStore-1] = train['unit_sales'].iloc[index] #assign the value of the items.

    
meanSales = np.expm1(np.mean(np.log1p(itemSales),axis=0))
test = pd.read_csv('../input/test.csv') #load test data



grouped_test = test.groupby(['item_nbr','store_nbr']) # group the data by items and stores

grouped_test = grouped_test.groups

testkeys = list(grouped_test.keys())



predicted = np.zeros((len(test),1)) # initialize the predicted output vector

for k in testkeys:

    # for cada group look in the meanSales the prediction. note that meanSales is 4100X54 (items,stores)

    indItem = np.where(items['item_nbr']==k[0])

    indStore = k[1]

    

    index = grouped_test[k]

    predicted[index,0] = meanSales[indItem,indStore-1]

    

    

# now save the data



submit = pd.DataFrame(np.random.randn(len(test),2), columns=['id', 'unit_sales'])    

submit['id'] = test['id']

submit['unit_sales'] = predicted # undo the log transform

 

submit.to_csv('prediction_01.csv', index = False)
