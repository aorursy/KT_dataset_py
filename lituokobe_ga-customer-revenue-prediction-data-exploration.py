# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Any results you write to the current directory are saved as output.
# Baby Nian will get started now.
def load_df(path, nrows=None):
    json_columns = ['device','geoNetwork','totals','trafficSource']
    
    df = pd.read_csv(path,
                     #make sure that the json in csv will be converted as dict, otherwise, it will be string
                     converters = {json_column: json.loads for json_column in json_columns},

                     #make sure 'fullVisitorId' is string
                     dtype = {'fullVisitorId':'str'},
                     nrows=nrows)

    for json_column in json_columns:
        #conver the dict as a dataframe
        converted_df = json_normalize(df[json_column])

        #format the name, f'{json_column}.{subcolumn}' = '{}.{}'.format(json_column,subcolumn)
        converted_df.columns = [f'{json_column}.{subcolumn}' for subcolumn in converted_df.columns]

        #remove the origin columns that are dict, and add the dataframe made of the new columns, keep both the ids
        df = df.drop(json_column, axis = 1).merge(converted_df, right_index=True, left_index=True)
        
    print(f'file:{path},shape:{df.shape}')
    #output the loaded and processed df
    return df
%%time
train_df = load_df('../input/train.csv')
train_df.head()
%%time
test_df = load_df('../input/test.csv')
test_df.head()
print('There are 2 columns that are in training set but not in the test set. They are',set(train_df.columns).difference(set(test_df.columns)))
#make the revenue all floats
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype('float')
train_df['totals.transactionRevenue'] = np.log1p(train_df['totals.transactionRevenue'].values)
#we want to explore how many people purchase repetitively first, to see we can use user indead of session as our index,.reset_index() will return a new dataframe instead of just an array
visitor_revenue_count = train_df.groupby('fullVisitorId')['totals.transactionRevenue'].count().sort_values(ascending=False).reset_index()
visitor_revenue_count.head()
#purchase time count chart
visitor_purchase_count = visitor_revenue_count.groupby('totals.transactionRevenue').count()

plt.figure(figsize=(12,5))
x_visitor = visitor_purchase_count.index[:20]
y_visitor = visitor_purchase_count ['fullVisitorId'][:20]
plt.title('Purchase time count chart')
plt.xlabel('Purchase time')
plt.ylabel('Count')
plt.bar(x_visitor,y_visitor)
plt.show()
#Before we process the data, we can also check what's the ratio of epople who make purchase in a session.
all_user_revenue = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(12,5))
plt.title('All user revenue')
x_all_revenue = all_user_revenue.index
y_all_revenue = all_user_revenue['totals.transactionRevenue'].values
plt.scatter(x_all_revenue, y_all_revenue)
plt.xlabel('user')
plt.ylabel('Revenue')
plt.show()
all_user = all_user_revenue.shape[0]

purchased_user = all_user_revenue[all_user_revenue['totals.transactionRevenue'] > 0].shape[0]

all_session = train_df.shape[0]

#pd.notnull() will return a list of boolearn to check if each row is null.But if you apply a .sum(), False will be 0, True will be 1, and you get the total number of True
purchased_session = train_df[train_df['totals.transactionRevenue'] > 0].shape[0]

print('There are',purchased_user,'users with the purchase, which is {percent:.2%} of all the users. There are'.format(percent=purchased_user/all_user),purchased_session,'sessions with the purchase, which is {percent:.2%} of all the sessions.'.format(percent=purchased_session/all_session))
#nunique() will return the unique numbers of a column, by default, dropna = False, but here, we make it True, because we don't want to ignore any null values
unhelpful_columns = []

for column in train_df.columns:
    if train_df[column].nunique(dropna = False) == 1:
        unhelpful_columns.append(column)

print('There are',train_df.shape[1],'columns.',len(unhelpful_columns),'of them have identical values for all rows. And they are as follows:')
print(unhelpful_columns)
unhelpful_columns.append('trafficSource.campaignCode')
train_df = train_df.drop(unhelpful_columns, axis = 1)
print('After removing the unhelpful columns, we have',train_df.shape[1],'columns, and they are',train_df.columns)
len(train_df.columns)
for column in unhelpful_columns:
    if column in test_df.columns:
        test_df = test_df.drop(column, axis = 1)
test_df.shape
print(test_df.shape,train_df.shape)
#sort the device transaction value by grouping device browser (if you use .count to only check the numerb, you don't need to put the 'by' parameter in the sort_values() function)
#.agg() let you check multiple info at one time, size is the total number of rows of a browser, count is number of rows that has revunue, mean is the average revenue
train_df['device.browser'] = train_df['device.browser'].fillna('NA')
device_browser = train_df.groupby('device.browser')['totals.transactionRevenue'].agg(['size','count','min','max','mean']).sort_values(by='mean',ascending = False)
device_browser.columns = ['total rows','rows with revenue','min','max','average revenue']

#type(device_browser)
#print(device_browser)
#below in comment is the simple pandas way to draw bar charts
# ax_device_browser1=device_browser.head(10).plot.bar(y='total rows')
# ax_device_browser2=device_browser.head(10).plot.bar(y='rows with revenue')

# we are still going to use plt to do the bar chart, making them line up in one row and saving space
x = device_browser.index[:20]
y1 = device_browser['total rows'][:20]
y2 = device_browser['rows with revenue'][:20]
y3 = device_browser['min'][:20]
y4 = device_browser['max'][:20]
y5 = device_browser['average revenue'][:20]

#subplot(3,1,3)means this plot has one row, 3 plots in a row,  this subplot is the first one
print('The effect of device browser on revenue')
plt.figure(figsize=(25,12))

plt.subplot(2,3,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x,y3)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x,y4)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x,y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

device_browser.head(20)
train_df['device.deviceCategory'] = train_df['device.deviceCategory'].fillna('NA')
device_category = train_df.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size','count','min','max','mean']).sort_values(by='count',ascending = False)
device_category.columns = ['total rows','rows with revenue','min','max','average revenue']

x = device_category.index[:20]
y1 = device_category['total rows'][:20]
y2 = device_category['rows with revenue'][:20]
y3 = device_category['min'][:20]
y4 = device_category['max'][:20]
y5 = device_category['average revenue'][:20]

print('The effect of device category on revenue')
plt.figure(figsize=(25,12))

plt.subplot(2,3,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x,y3)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x,y4)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x,y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

device_category.head()
#train_df['device_isMobile'] = train_df['device_isMobile'].fillna('NA')
device_isMobile = train_df.groupby('device.isMobile')['totals.transactionRevenue'].agg(['size','count','mean']).sort_values(by='count',ascending = False)
device_isMobile.columns = ['total rows','rows with revenue','average revenue']

x = device_isMobile.index
y1 = device_isMobile['total rows']
y2 = device_isMobile['rows with revenue']
y3 = device_isMobile['average revenue']

print('The effect of device category on revenue')
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(1,3,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(1,3,3)
plt.bar(x,y3)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

device_isMobile.head()
train_df['device.operatingSystem'] = train_df['device.operatingSystem'].fillna('NA')
device_operatingSystem = train_df.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size','count','min','max','mean']).sort_values(by='mean',ascending = False)
device_operatingSystem.columns = ['total rows','rows with revenue','min','max','average revenue']

x = device_operatingSystem.index[:20]
y1 = device_operatingSystem['total rows'][:20]
y2 = device_operatingSystem['rows with revenue'][:20]
y3 = device_operatingSystem['min'][:20]
y4 = device_operatingSystem['max'][:20]
y5 = device_operatingSystem['average revenue'][:20]

print('The effect of device category on revenue')
plt.figure(figsize=(25,12))

plt.subplot(2,3,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x,y3)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x,y4)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x,y5)
plt.title('average revenue')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

device_operatingSystem.head(40)
train_df['geoNetwork.city'] = train_df['geoNetwork.city'].fillna('NA')
city = train_df.groupby('geoNetwork.city')['totals.transactionRevenue'].agg(['size','count','min','max','mean']).sort_values(by='count',ascending = True)
city.columns = ['total rows','rows with revenue','min','max','average revenue']

x = city.index[:100]
y1 = city['total rows'][:100]
y2 = city['rows with revenue'][:100]
y3 = city['min'][:100]
y4 = city['max'][:100]
y5 = city['average revenue'][:100]

print('The effect of city')
plt.figure(figsize=(24,35))

plt.subplot(5,1,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(5,1,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(5,1,3)
plt.bar(x,y3)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(5,1,4)
plt.bar(x,y4)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(5,1,5)
plt.bar(x,y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

city.head(100)
#country
train_df['geoNetwork.country'] = train_df['geoNetwork.country'].fillna('NA')
country_or_region = train_df.groupby('geoNetwork.country')['totals.transactionRevenue'].agg(['size','count','min','max','mean']).sort_values(by='count',ascending = True)
country_or_region.columns = ['total rows','rows with revenue','min','max','average revenue']

x = country_or_region.index[:100]
y1 = country_or_region['total rows'][:100]
y2 = country_or_region['rows with revenue'][:100]
y3 = country_or_region['min'][:100]
y4 = country_or_region['max'][:100]
y5 = country_or_region['average revenue'][:100]

print('The effect of country')
plt.figure(figsize=(24,35))

plt.subplot(5,1,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(5,1,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(5,1,3)
plt.bar(x,y3)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(5,1,4)
plt.bar(x,y4)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(5,1,5)
plt.bar(x,y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

country_or_region.head(150)
#country
train_df['geoNetwork.subContinent'] = train_df['geoNetwork.subContinent'].fillna('NA')
country_or_region = train_df.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(['size','count','min','max','mean']).sort_values(by='count',ascending = True)
country_or_region.columns = ['total rows','rows with revenue','min','max','average revenue']

x = country_or_region.index[:100]
y1 = country_or_region['total rows'][:100]
y2 = country_or_region['rows with revenue'][:100]
y3 = country_or_region['min'][:100]
y4 = country_or_region['max'][:100]
y5 = country_or_region['average revenue'][:100]

print('The effect of sub continent')
plt.figure(figsize=(24,35))

plt.subplot(5,1,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(5,1,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(5,1,3)
plt.bar(x,y3)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(5,1,4)
plt.bar(x,y4)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(5,1,5)
plt.bar(x,y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

country_or_region.head(150)
train_df['geoNetwork.networkDomain'] = train_df['geoNetwork.networkDomain'].fillna('NA')
network_domain = train_df.groupby('geoNetwork.networkDomain')['totals.transactionRevenue'].agg(['size','count','mean']).sort_values(by='count',ascending = True)
network_domain.columns = ['total rows','rows with revenue','average revenue']

x = network_domain.index[:100]
y1 = network_domain['total rows'][:100]
y2 = network_domain['rows with revenue'][:100]
y3 = network_domain['average revenue'][:100]

print('The effect of network domain')
plt.figure(figsize=(24,15))

plt.subplot(3,1,1)
plt.bar(x,y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(3,1,2)
plt.bar(x,y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(3,1,3)
plt.bar(x,y3)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

network_domain.head(50)
train_df['visitNumber'] = train_df['visitNumber'].fillna(0)
visit_number = train_df.groupby('visitNumber')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
visit_number.columns = ['total rows','rows with revenue','max','min','average revenue']

x = visit_number.index[:40]
y1 = visit_number['total rows'][:40]
y2 = visit_number['rows with revenue'][:40]
y3 = visit_number['max'][:40]
y4 = visit_number['min'][:40]
y5 = visit_number['average revenue'][:40]

print('The effect of visit number')
plt.figure(figsize=(24,16))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average avenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

visit_number.head(10)
train_df['totals.bounces'] = train_df['totals.bounces'].fillna(0)
total_bounces = train_df.groupby('totals.bounces')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
total_bounces.columns = ['total rows','rows with revenue','max','min','average revenue']

x = total_bounces.index[:40]
y1 = total_bounces['total rows'][:40]
y2 = total_bounces['rows with revenue'][:40]
y3 = total_bounces['max'][:40]
y4 = total_bounces['max'][:40]
y5 = total_bounces['average revenue'][:40]

print('The effect of total bounces')
plt.figure(figsize=(24,8))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

total_bounces.head(10)
train_df['totals.hits'] = train_df['totals.hits'].fillna(0)
total_hits = train_df.groupby('totals.hits')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = True)
total_hits.columns = ['total rows','rows with revenue','max','min','average revenue']

x = total_hits.index[:65]
y1 = total_hits['total rows'][:65]
y2 = total_hits['rows with revenue'][:65]
y3 = total_hits['max'][:65]
y4 = total_hits['min'][:65]
y5 = total_hits['average revenue'][:65]

print('The effect of total_hits')
plt.figure(figsize=(24,15))

plt.subplot(3,2,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(3,2,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(3,2,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(3,2,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(3,2,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

total_hits.head(10)
train_df['totals.newVisits'] = train_df['totals.newVisits'].fillna(0)

total_new_visits = train_df.groupby('totals.newVisits')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
total_new_visits.columns = ['total rows','rows with revenue','max','min','average revenue']

x = total_new_visits.index[:40]
y1 = total_new_visits['total rows'][:40]
y2 = total_new_visits['rows with revenue'][:40]
y3 = total_new_visits['max'][:40]
y4 = total_new_visits['min'][:40]
y5 = total_new_visits['average revenue'][:40]

print('The effect of total new visits')
plt.figure(figsize=(24,16))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

total_new_visits.head(10)
train_df['totals.pageviews'] = train_df['totals.pageviews'].fillna(0)

total_pageviews = train_df.groupby('totals.pageviews')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = True)
total_pageviews.columns = ['total rows','rows with revenue','max','min','average revenue']

x = total_pageviews.index[:50]
y1 = total_pageviews['total rows'][:50]
y2 = total_pageviews['rows with revenue'][:50]
y3 = total_pageviews['max'][:50]
y4 = total_pageviews['min'][:50]
y5 = total_pageviews['average revenue'][:50]

print('The effect of total pageviews')
plt.figure(figsize=(24,15))

plt.subplot(3,2,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(3,2,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(3,2,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(3,2,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(3,2,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

total_pageviews.head()
bounces1 = train_df[train_df['totals.bounces']=='1']
hitonce = train_df[train_df['totals.hits']=='1']

filter1 = train_df['totals.pageviews']== '1'
filter2 = train_df['totals.pageviews']== 0
visit0_1 = train_df[filter1|filter2]

all_non_purchase = set(bounces1.index)&set(hitonce.index)&set(visit0_1.index)
print('There are',bounces1.shape[0],'people who bounced,',hitonce.shape[0],'people who only hit once,',visit0_1.shape[0],'people who only visited zero time or once, and',len(all_non_purchase),'people who did all of these.')
train_df['channelGrouping'] = train_df['channelGrouping'].fillna('NA')
channel = train_df.groupby('channelGrouping')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
channel.columns = ['total rows','rows with revenue','max','min','average revenue']

x = channel.index[:40]
y1 = channel['total rows'][:40]
y2 = channel['rows with revenue'][:40]
y3 = channel['max'][:40]
y4 = channel['min'][:40]
y5 = channel['average revenue'][:40]

print('The effect of channel')
plt.figure(figsize=(24,16))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

channel.head(10)
train_df['trafficSource.adContent'] = train_df['trafficSource.adContent'].fillna('NA')
ad_content = train_df.groupby('trafficSource.adContent')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = True)
ad_content.columns = ['total rows','rows with revenue','max','min','average revenue']

x = ad_content.index[:40]
y1 = ad_content['total rows'][:40]
y2 = ad_content['rows with revenue'][:40]
y3 = ad_content['max'][:40]
y4 = ad_content['min'][:40]
y5 = ad_content['average revenue'][:40]

print('The effect of ad content')
plt.figure(figsize=(24,15))

#in the plot, the x can only be str, so we convert "0/1" to strings
plt.subplot(3,2,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(3,2,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(3,2,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(3,2,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(3,2,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

ad_content.head(50)
aa = len(set(test_df['trafficSource.adContent']))
bb = len(set(test_df['trafficSource.adContent']) & set(ad_content[ad_content['average revenue']>0].index))
bb/aa
set(test_df['trafficSource.adContent']) & set(ad_content[ad_content['average revenue']>0].index)
train_df['trafficSource.adwordsClickInfo.page'] = train_df['trafficSource.adwordsClickInfo.page'].fillna(0)
ad_click_page = train_df.groupby('trafficSource.adwordsClickInfo.page')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
ad_click_page.columns = ['total rows','rows with revenue','max','min','average revenue']

x = ad_click_page.index[:40]
y1 = ad_click_page['total rows'][:40]
y2 = ad_click_page['rows with revenue'][:40]
y3 = ad_click_page['max'][:40]
y4 = ad_click_page['min'][:40]
y5 = ad_click_page['average revenue'][:40]

print('The effect of ad click page')
plt.figure(figsize=(24,16))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

ad_click_page.head(10)
train_df['trafficSource.isTrueDirect'] = train_df['trafficSource.isTrueDirect'].fillna('False')
true_direct = train_df.groupby('trafficSource.isTrueDirect')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
true_direct.columns = ['total rows','rows with revenue','max','min','average revenue']

x = true_direct.index[:40]
y1 = true_direct['total rows'][:40]
y2 = true_direct['rows with revenue'][:40]
y3 = true_direct['max'][:40]
y4 = true_direct['min'][:40]
y5 = true_direct['average revenue'][:40]

print('The effect of true direct')
plt.figure(figsize=(24,16))

#in the plot, the x can only be str, so we convert "0/1" to strings
plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

true_direct.head(10)
train_df['trafficSource.keyword'] = train_df['trafficSource.keyword'].fillna('NA')
keywords = train_df.groupby('trafficSource.keyword')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = True)
keywords.columns = ['total rows','rows with revenue','max','min','average revenue']

x = keywords.index[:40]
y1 = keywords['total rows'][:40]
y2 = keywords['rows with revenue'][:40]
y3 = keywords['max'][:40]
y4 = keywords['min'][:40]
y5 = keywords['average revenue'][:40]

print('The effect of keywords')
plt.figure(figsize=(24,16))

#in the plot, the x can only be str, so we convert "0/1" to strings
plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

keywords.head()
train_df['trafficSource.medium'] = train_df['trafficSource.medium'].fillna('NA')
medium = train_df.groupby('trafficSource.medium')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = False)
medium.columns = ['total rows','rows with revenue','max','min','average revenue']

x = medium.index[:40]
y1 = medium['total rows'][:40]
y2 = medium['rows with revenue'][:40]
y3 = medium['max'][:40]
y4 = medium['min'][:40]
y5 = medium['average revenue'][:40]

print('The effect of medium')
plt.figure(figsize=(24,16))

#in the plot, the x can only be str, so we convert "0/1" to strings
plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

medium.head(10)
train_df['trafficSource.referralPath'] = train_df['trafficSource.referralPath'].fillna('NA')
referral_path = train_df.groupby('trafficSource.referralPath')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='count',ascending = True)
referral_path.columns = ['total rows','rows with revenue','max','min','average revenue']

x = referral_path.index[:80]
y1 = referral_path['total rows'][:80]
y2 = referral_path['rows with revenue'][:80]
y3 = referral_path['max'][:80]
y4 = referral_path['min'][:80]
y5 = referral_path['average revenue'][:80]

print('The effect of referral path')
plt.figure(figsize=(24,30))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y2)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y4)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

referral_path.head(10)
train_df['trafficSource.source'] = train_df['trafficSource.source'].fillna('NA')
source = train_df.groupby('trafficSource.source')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='mean',ascending = True)
source.columns = ['total rows','rows with revenue','max','min','average revenue']

x = source.index[:40]
y1 = source['total rows'][:40]
y2 = source['rows with revenue'][:40]
y3 = source['max'][:40]
y4 = source['min'][:40]
y5 = source['average revenue'][:40]

print('The effect of source')
plt.figure(figsize=(24,15))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y5)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

source.head(10)
train_df['trafficSource.adwordsClickInfo.adNetworkType'] = train_df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('NA')
adNetworkType = train_df.groupby('trafficSource.adwordsClickInfo.adNetworkType')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='mean',ascending = True)
adNetworkType.columns = ['total rows','rows with revenue','max','min','average revenue']

x = adNetworkType.index[:40]
y1 = adNetworkType['total rows'][:40]
y2 = adNetworkType['rows with revenue'][:40]
y3 = adNetworkType['max'][:40]
y4 = adNetworkType['min'][:40]
y5 = adNetworkType['average revenue'][:40]

print('The effect of adNetworkType')
plt.figure(figsize=(24,15))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y5)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

adNetworkType.head(10)
train_df['trafficSource.adwordsClickInfo.gclId'] = train_df['trafficSource.adwordsClickInfo.gclId'].fillna('NA')
gclId = train_df.groupby('trafficSource.adwordsClickInfo.gclId')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='mean',ascending = True)
gclId.columns = ['total rows','rows with revenue','max','min','average revenue']

x = gclId.index[:40]
y1 = gclId['total rows'][:40]
y2 = gclId['rows with revenue'][:40]
y3 = gclId['max'][:40]
y4 = gclId['min'][:40]
y5 = gclId['average revenue'][:40]

print('The effect of gclId')
plt.figure(figsize=(24,15))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y5)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

gclId.head(10)
train_df['trafficSource.adwordsClickInfo.isVideoAd'] = train_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
isVideoAd = train_df.groupby('trafficSource.adwordsClickInfo.isVideoAd')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='mean',ascending = True)
isVideoAd.columns = ['total rows','rows with revenue','max','min','average revenue']

x = isVideoAd.index[:40]
y1 = isVideoAd['total rows'][:40]
y2 = isVideoAd['rows with revenue'][:40]
y3 = isVideoAd['max'][:40]
y4 = isVideoAd['min'][:40]
y5 = isVideoAd['average revenue'][:40]

print('The effect of isVideoAd')
plt.figure(figsize=(24,15))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y5)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

isVideoAd.head(10)
train_df['trafficSource.adwordsClickInfo.slot'] = train_df['trafficSource.adwordsClickInfo.slot'].fillna('NA')
slot = train_df.groupby('trafficSource.adwordsClickInfo.slot')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='mean',ascending = True)
slot.columns = ['total rows','rows with revenue','max','min','average revenue']

x = slot.index[:40]
y1 = slot['total rows'][:40]
y2 = slot['rows with revenue'][:40]
y3 = slot['max'][:40]
y4 = slot['min'][:40]
y5 = slot['average revenue'][:40]

print('The effect of slot')
plt.figure(figsize=(24,15))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y5)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

slot.head(10)
train_df['trafficSource.campaign'] = train_df['trafficSource.campaign'].fillna('NA')
campaign = train_df.groupby('trafficSource.campaign')['totals.transactionRevenue'].agg(['size','count','max','min','mean']).sort_values(by='mean',ascending = True)
campaign.columns = ['total rows','rows with revenue','max','min','average revenue']

x = campaign.index[:40]
y1 = campaign['total rows'][:40]
y2 = campaign['rows with revenue'][:40]
y3 = campaign['max'][:40]
y4 = campaign['min'][:40]
y5 = campaign['average revenue'][:40]

print('The effect of campaign')
plt.figure(figsize=(24,15))

plt.subplot(2,3,1)
plt.bar(x.astype('str'),y1)
plt.title('total rows')
plt.xticks(rotation=90)

plt.subplot(2,3,2)
plt.bar(x.astype('str'),y2)
plt.title('rows with revenue')
plt.xticks(rotation=90)

plt.subplot(2,3,3)
plt.bar(x.astype('str'),y3)
plt.title('max')
plt.xticks(rotation=90)

plt.subplot(2,3,4)
plt.bar(x.astype('str'),y5)
plt.title('min')
plt.xticks(rotation=90)

plt.subplot(2,3,5)
plt.bar(x.astype('str'),y5)
plt.title('average revenue')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

campaign.head(10)
train_df.columns