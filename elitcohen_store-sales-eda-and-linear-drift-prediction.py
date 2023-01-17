import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')
submission.info()
submission.head()
test.info()
test.head()
train.info()
train.head()
train.describe()
# calculate monthly sales per store

dates = train['date'].apply(lambda x: x[:-3]).unique() # chop off the "day" part of the date

all_storesales = {}

for id in range(1,11):

    print('calculating store',id)

    all_storesales[id] = []

    for date in dates:

        # extract the sales data for that store, for that date

        storedata = train[(train['store'] == id) & (train['date'].apply(lambda x: x[:-3]) == date)]

        storesales = storedata['sales'].sum()

        all_storesales[id].append(storesales)
# plot the results for each store

for id in range(1,11):

    plt.figure(figsize=(20,10))

    plt.plot(dates, all_storesales[id])

    plt.title('Store '+str(id)+' Sales', fontsize=30)

    plt.xticks(rotation=90)

    plt.show()
# collect total sales per item, per store

items = range(1,51)

all_itemsales = {}

for id in range(1,11):

    print('calculating store',id)

    all_itemsales[id] = []

    for item in items:

        itemdata = train[(train['store'] == id) & (train['item'] == item)]

        itemsales = itemdata['sales'].sum()

        all_itemsales[id].append(itemsales)
# calculate the mean sales per product, across all 10 stores

mean_item_sales = np.array(all_itemsales[1])

for id in range(2,11):

    mean_item_sales += np.array(all_itemsales[id])

mean_item_sales = np.divide(mean_item_sales, 10)



# plot them

for id in range(1,11):

    plt.figure(figsize=(20,10))

    plt.bar(items, all_itemsales[id])

    plt.plot(items, mean_item_sales, color='red')

    plt.title('Store '+str(id)+' Sales By Item', fontsize=30)

    plt.xticks(items)

    plt.show()
dates_projected = [ '2018-'+str(mo) if mo >= 10 else '2018-0'+str(mo) for mo in range(1,13) ] # some effort needed to account for leading '0' in single-digit months e.g. '2018-07'



all_storesales_projected = {}

for id in range(1,11):

    print('calculating store',id)

    

    all_storesales_projected[id] = []

    # iterate over months, and THEN years, to collect the trend of a single month's sales over multiple years, and then repeat for all months

    for month in range(1,13):

        month_pts = []

        for year in range(2013,2018):

            # get num sales for same month from past years

            date = str(year)+'-'+str(month) if month >= 10 else str(year)+'-0'+str(month)

            storedata = train[(train['store'] == id) & (train['date'].apply(lambda x: x[:-3]) == date)]

            storesales = storedata['sales'].sum()

            month_pts.append(storesales)

            

        # predict next value by taking the average of the diffs between consecutive years, and append it to projected sales

        # this could be improved with a true linear regression, as long as there aren't any extreme outliers

        total_diff = 0

        for idx,mp in enumerate(month_pts[1:]):

            total_diff += mp - month_pts[idx-1]

        mean_diff = total_diff/(len(month_pts)-1)

        next_pt = month_pts[-1] + mean_diff

        all_storesales_projected[id].append(next_pt)
# plot the predicted monthly sales

for id in range(1,11):

    plt.figure(figsize=(20,10))

    plt.plot(dates, all_storesales[id])

    plt.plot(dates_projected, all_storesales_projected[id], color='red')

    plt.title('Store '+str(id)+' Sales', fontsize=30)

    plt.xticks(rotation=90)

    plt.show()
days_in_month = { 1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31 }



predicted_sales = []

for idx,row in test.iterrows():

    month = int(row['date'].split('-')[1])

    id = row['store']

    item = row['item']

    

    # get the predicted sales for the month

    total_month_sales_projected = all_storesales_projected[id][month-1]

    

    # get product's fraction of sales

    item_sales_fraction = float(all_itemsales[id][item-1]) / sum(all_itemsales[id])

    

    # get predicted monthly sales for that product, and divide by # days in that month to get daily sales of that product

    item_sales_projected = total_month_sales_projected*item_sales_fraction / days_in_month[month]

    predicted_sales.append(item_sales_projected)



# add predictions to submission file

submission['sales'] = predicted_sales

submission.to_csv('submission_basic.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
train_rf = train.copy()



# separate month/day/year into separate columns, as they are independent variables for RF

train_rf['year'] = train_rf['date'].apply(lambda x: int(x.split('-')[0]))

train_rf['month'] = train_rf['date'].apply(lambda x: int(x.split('-')[1]))

train_rf['day'] = train_rf['date'].apply(lambda x: int(x.split('-')[2]))



train_rf = train_rf.drop('date', axis=1)
# train the model

model = RandomForestRegressor(n_estimators=100)

model.fit(train_rf.drop('sales',axis=1), train_rf['sales'])
test_rf = test.copy()
# apply same data transformation to the test set

test_rf['year'] = test_rf['date'].apply(lambda x: int(x.split('-')[0]))

test_rf['month'] = test_rf['date'].apply(lambda x: int(x.split('-')[1]))

test_rf['day'] = test_rf['date'].apply(lambda x: int(x.split('-')[2]))



test_rf = test_rf.drop(['date','id'], axis=1)
# make predictions

pred = model.predict(test_rf)

print(pred)



# set on test dataframes

test_rf['sales'] = pred

test['sales'] = pred
test.info()
test.head()
# plot monthly sales per store

dates = test['date'].apply(lambda x: x[:-3]).unique()

all_storesales_projected = {}

for id in range(1,11):

    print('calculating store',id)

    all_storesales_projected[id] = []

    for date in dates:

        storedata = test[(test['store'] == id) & (test['date'].apply(lambda x: x[:-3]) == date)]

        storesales = storedata['sales'].sum()

        all_storesales_projected[id].append(storesales)

        

prev_dates = train['date'].apply(lambda x: x[:-3]).unique()



for id in range(1,11):

    plt.figure(figsize=(20,10))

    plt.plot(prev_dates, all_storesales[id])

    plt.plot(dates, all_storesales_projected[id], color='red')

    plt.title('Store '+str(id)+' Sales (RF)', fontsize=30)

    plt.xticks(rotation=90)

    plt.show()