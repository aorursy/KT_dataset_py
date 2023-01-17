"""
1) Prepare Problem
a) Load libraries
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Lets see what other libraries I will be using
# Keras
# sklearn
"""
1) Prepare Problem
b) Load dataset
"""
items=pd.read_csv('../input/items.csv')
item_categories=pd.read_csv('../input/item_categories.csv')
shops=pd.read_csv('../input/shops.csv')

# What about these *.gz files?
# It is a compressed format: "For on-the-fly decompression of on-disk data"
test=pd.read_csv('../input/test.csv.gz',compression='gzip')
sample_submission=pd.read_csv('../input/sample_submission.csv.gz',compression='gzip')
sales_train=pd.read_csv('../input/sales_train.csv.gz',compression='gzip')
items.info()
items.head()
items.describe()
item_categories.info()
item_categories.head()
item_categories.describe()
# Gives descriptive statistics on quantitative features
shops.info()
shops.head()
shops.describe()
test.info()
test.head()
test.describe()
sample_submission.info()
sample_submission.head()
sample_submission.describe()
sales_train.info()
sales_train.head()
sales_train.describe()
sales_train.info()
sales_train.describe()
sales_train.item_price.hist()
sales_train.item_price.value_counts()
sales_train.item_price.nunique()
sales_train.item_price.max()
print(sales_train[sales_train.item_price==sales_train.item_price.max()])
print(sales_train[sales_train.item_price==sales_train.item_price.max()].item_id)
print(items[items.item_id==6066])
# Radmin is a remote control software - dont think that it is that expensive. Let's check if it was sold for "normal" prices
print(sales_train[sales_train.item_id==6066])
# Only this one time. Interesting. Then maybe it is right. One huge license?
# Let's see if there are other Radmin versions
# and if this is the only outlier in price
print(sales_train[sales_train.item_price>50000])
# ok lets leave it in for now.
sales_train[sales_train.item_price<60000].item_price.hist()
sales_train[sales_train.item_price<30000].item_price.hist()
sales_train[sales_train.item_price<15000].item_price.hist()
sales_train[sales_train.item_price<5000].item_price.hist()
sales_train[sales_train.item_price<3000].item_price.hist()
sales_train[sales_train.item_price<1000].item_price.hist()
# So definetly lets build some categories on price. There seems to be mayority is small B2C business but there are also big B2B deals.

# In general I should understand more what actually the products are:
print(item_categories.head(300))
# Playstation, X-Box and kyrillic things.
# Lets translate the column (and also the shop column - check if we can see cities)
"""from textblob import TextBlob

item_categories['english'] = item_categories['item_category_name'].str.encode('cp437', 'ignore').apply(lambda x:TextBlob(x.strip()).translate(to='en'))
"""
"""
from unidecode import unidecode
item_categories['english'] = unidecode(item_categories['item_category_name'])
"""
# 3rd solution worked out: https://stackoverflow.com/questions/14173421/use-string-translate-in-python-to-transliterate-cyrillic
symbols=(u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")
english = {ord(a):ord(b) for a, b in zip(*symbols)}

item_categories['items_english'] = item_categories['item_category_name'].apply(lambda x: x.translate(english))

print(item_categories.items_english.head(100))

# Observations:
# In categories are meta-categories: Accessories, Console, PC, programs, music...
# Added to to-do list: Take these meta-categories as features

#Split the metacategories with the "-"
item_categories["meta_category"]=item_categories.items_english.apply(lambda x:x.split(" - ")[0])
print(item_categories.meta_category.head(100))
item_categories.head()
print(item_categories.meta_category.unique())
print(item_categories.meta_category.nunique())
#Great! Only 20 makro-categories
print(item_categories.meta_category.value_counts())
print(item_categories.meta_category)
# Of course: I need to put the makro-categories into the data
shops.info()

# Translate shop names
shops['shops_english'] = shops['shop_name'].apply(lambda x: x.translate(english))
print(shops.shops_english.head(100))

# YES! First word is the city! Great feature to extract! Another "Makro-category"
"""
# And because it is only 60 objects this can even be done and checked manually
shops["town"] =["Yakutsk","Yakutsk","Adygea","Balasiha","Volzhskij","Vologda","Voronej","Voronej",]
"""
# No this was to stupid:
shops["town"]=shops.shops_english.apply(lambda x:x.split()[0])
print(shops.town)

# While doing this and researching cities next idea: Another makro feature of "regions" eg Balashiha belongs to moscow region
shops["region"]=["Sakha","Sakha","Adygea","Moscow","Volgograd", "Vologda", "voronezh","voronezh","voronezh","Vyezdnaa", "Moscow", "Moscow","Internet", "tatarstan", "tatarstan","Kaluga", "Moscow", "Moscow", "Moscow", "Kursk", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow","Moscow","novgorod","novgorod","novosibirsk","novosibirsk","omsk","rostov","rostov","rostov","Saint Peterburg","Saint Peterburg","samara","samara","moscow","Khanty-Mansi","Tomsk","Tyumen","tyumen","tyumen","Bashkortostan","Bashkortostan","moscow","zifrovoj","moscow","sakha","sakha","yaroslavl"]
print(shops.town.nunique())
print(shops.region.nunique())
# hmmm didn't help much - only 6 towns that belong to Moscow region

shops.to_csv('final_shops.csv',index=False)
# Always print (parts of) data that you are examining just to get an idea
# done
print(test.shape)
print(sales_train.shape)

# 3 features missing in test
print(test.columns)
print(sales_train.columns)

print(test.head())
# ok test is really only the form I need to fill. per shop per item forecast revenue for the specific month
# Therefore need to split later training data into train & validation set
# Feedback on test-set will be the evaluation via Kaggle and/or coursera
sales_train.columns
# Let's start with the dates column
sales_train['day'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.day
sales_train['month'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.month
sales_train['year'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.year
sales_train['weekday'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.dayofweek
sales_train.columns
print(sales_train.head())
# Dates look ordered and shops also
sales_train.date_block_num.plot()
sales_train.shop_id.plot(figsize=(20,4))
# Interesting. There is a rythm to it
sales_train.weekday[0:100].plot(figsize=(20,4))
sales_train.head(100)
"""
Aha. Order of train set is by
- month
- shops per month
- item_id per shop
- dates of items per shop
"""
sales_train[3000:4000]
# No hypotheses from above is wrong. Shop 25 appears again after shop 24
sales_train.item_id[1:10000].plot(figsize=(20,4))
# There seem to be groups. Maybe it has to do with categories?
sales_train=sales_train.merge(items, how='left')
sales_train=sales_train.merge(item_categories,how="left")
sales_train=sales_train.merge(shops,how="left")
sales_train.head()
sales_train.drop("item_name",axis=1,inplace=True)
sales_train.drop("shop_name",axis=1,inplace=True)
sales_train.drop("item_category_name",axis=1,inplace=True)
sales_train.head()
sales_train.isnull().values.any()
# No values NaN
# Maybe NaN values have been replaced by a number? Carefully check for outliers
sales_train.head(200)
# Can't see a strict order. Kind of per shop. Kind of per item_id. 
sales_train["revenue"]=sales_train.item_cnt_day * sales_train.item_price
sales_train.groupby("year").sum()
# Interesting: 2015 lower revenue - no! We dont have full years. Reaches from beginning of 2013 to October 2015
sales_train.groupby("date_block_num").sum()["revenue"].plot()
# Very clearly typical retail pattern: Peak at christmas and low in Summer
sales_train.groupby("weekday").sum()["revenue"].plot()
# looks like Friday, Saturday and Sunday are the busiest days
sales_train.groupby("shop_id").sum()["revenue"].plot.bar()
# Significant differences
# But careful: Could be that shops opened later or closed earlier
# And what is with the online shop?
sales_train[sales_train["shops_english"]=="Internet-magazin CS"]["revenue"].sum()
# 1.1 in above scale so not the most big one
sales_train[sales_train["shops_english"]=="Internet-magazin CS"]["revenue"].plot()
# funny spikes
# I expected steady and increasing sales if it would be an online shop
sales_train[sales_train["shops_english"]=="Internet-magazin CS"].groupby("date_block_num").sum()["revenue"].plot()
# How does this look for other shops?
sales_train[sales_train["shops_english"]=="Vyezdnaa Torgovla"]["revenue"].plot()
# Spikes seem to be rather normal when larger things are being sold.
# These "things" need to be looked into much deeper and when they occure. They have a mayor impact! 
# How can tihs be modelled? Is this yearly licenses? Or random occurence and I should model a random? 
sales_train[sales_train["shops_english"]=="Vyezdnaa Torgovla"].groupby("date_block_num").sum()["revenue"].plot()
# This looks very strange
# Has it to do with a test-train split? No data for monthes 23-31?
sales_train.groupby("shop_id").sum()["revenue"]
# Understand the training data structure a bit better
sales_train.shop_id.plot(figsize=(20,4))
# Why so irregular?
sales_train.groupby("date_block_num").count().town.plot()
# The number of transactions is shrinking!
sales_train.groupby("date_block_num").mean().item_price.plot()
# But because average price is increasing stronger the revenue is increasing
sales_train.groupby("date_block_num").mean().revenue.plot()
sales_train.groupby("date_block_num").sum().item_cnt_day.plot()
#Just doublechecking it is not only number of transactions but also total items
# https://tradingeconomics.com/russia/inflation-rate-mom
# Inflation rate in russia over the period was oscillating between 0 and 1%
# except one huge peak (to 4% per month very shortly) end of 2014, beginning 2015
# Due to lower oil prices and Western sanctions imposed over Ukraine

# https://www.quora.com/Economy-of-Russia-What-caused-the-high-inflation-in-Russia-in-2014-and-2015
# It translates to a yearly inflation rate of 17%    
sns.pairplot(sales_train)

# if you work in the kernel you should de-activate this as it takes a long time

"""Observations:
- one extreme outlier in price
- one extreme outlier in cnt_day
- one month (~10 date_block_num) has a lot of high revenue items
- certain item_ids have wide range of revenues, some have outliers
"""
sales_train.groupby(["shop_id","date_block_num"]).sum()
# Definetly big difference in how long shops are on the market
shop_life=pd.DataFrame(columns=["shop_id","Start", "Stop"])
shop_life["shop_id"]=np.arange(60)
shop_life["Start"]=sales_train.groupby("shop_id")["date_block_num"].min()
shop_life["Stop"]=sales_train.groupby("shop_id")["date_block_num"].max()
shop_life.merge(shops, how="left").drop("shop_name",axis=1)
print(shop_life)
"""
Observations:
- shops 10 and 11 have the same name, just ^2 and ? -> Check if shop 10 is empty at month 25 (a)
- shops 39 and 40 seem to be the same? (b)
- definetly need to check what shops are in the test-set (c)
- should closed shops be considered? (d)
"""
# (a)
sales_train[(sales_train["shop_id"]==10) & (sales_train["date_block_num"]==25)]
sales_train[(sales_train["shop_id"]==11) & (sales_train["date_block_num"]==25)]
sales_train[(sales_train["shop_id"]==10) & (sales_train["date_block_num"]==24)]
sales_train[(sales_train["shop_id"]==10) & (sales_train["date_block_num"]==26)]
sales_train[(sales_train["shop_id"]==11) & (sales_train["date_block_num"]==24)]
sales_train[(sales_train["shop_id"]==11) & (sales_train["date_block_num"]==26)]
# Good. Brute-force but clear.
# Let's have a 100% picture:
sales_train[(sales_train["shop_id"]==10) | (sales_train["shop_id"]==11)].groupby(["shop_id","date_block_num"]).sum()
# Yes, definetly.
sales_train.loc[sales_train["shop_id"]==11,"shop_id"]=10
sales_train[sales_train["shop_id"]==11]
sales_train[(sales_train["shop_id"]==10)&(sales_train["date_block_num"]==25)]
sales_train.to_csv('sales_train.csv',index=False)
# Good. Next one:
# b) shops 39 and 40 seem to be the same?
sales_train[(sales_train["shop_id"]==39) | (sales_train["shop_id"]==40)].groupby(["shop_id","date_block_num"]).sum()
# No, seems to be two separate shops. Both opened in month 14, one closed earlier than the other
#c) Check what shops are in the test-set
print(sorted(test.shop_id.unique()))
test_list=list(test.shop_id.unique())
complete_list=list(range(60))
out_of_test=[x for x in complete_list if x not in test_list]
print(out_of_test)
print(shop_life[shop_life["Stop"]<33])
# 9, 11, 20, are not in test but were active in time_period 33
# What could be the reason?
# Maybe they closed then?
# A good question is whether the train model should also look at the shops that are in test!?
print(shops.loc[9])
#print(shops.loc[11])
# Yes of course this one I deleted manually
print(shops.loc[20])
sales_train[(sales_train["shop_id"]==9) | (sales_train["shop_id"]==20)].groupby(["shop_id","date_block_num"]).sum()["revenue"]
# Aha, yet another trick. There is data only for limited periods for these shops. 
# Lets check if this is the reason and others are consistently in business
sales_train[(sales_train["shop_id"]==3) | (sales_train["shop_id"]==24)].groupby(["shop_id","date_block_num"]).sum()["revenue"]
# Yes looks fine

# I think it depends what I want to achieve whether i include these shops or not.
# Definetly a kind of different distribution in train and test

# I want to see KPIs over time (prices, revenue per shop per month)
# Let's start with prices
sales_train.groupby("item_id").sum()
# Clearly to be seen some items only very short time in sale
sales_train.groupby("item_id").sum()["revenue"].hist(figsize=(20,4),bins=100)
# many many items with little revenue
sales_train.groupby("item_category_id").sum()
sales_train.groupby("item_category_id").sum()["revenue"].hist(figsize=(20,4),bins=100)
# many many items with little revenue
sales_train.groupby(["date_block_num","item_category_id"]).sum()["revenue"].unstack()
# unstack is a great function!
# https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html
# https://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures/

sales_train.groupby(["date_block_num","item_category_id"]).sum()["revenue"].unstack().plot(figsize=(20,20))
# Let's just look at growth rates of categories (CAGR's)
sales_train.groupby(["date_block_num","meta_category"]).sum()["revenue"].unstack().plot(figsize=(20,20))
# Consoles with 2 big spikes
# Exactly at Christmas! :-) Playstations from Santa Claud: Or Father Frost as it is in Russia I think
sales_train.groupby(["date_block_num","shop_id"]).sum()["revenue"].unstack().plot(figsize=(20,20))
# of course same spikes. Let's normalize
sales_train.groupby(["date","shop_id"]).sum()["revenue"].unstack().plot(figsize=(20,20))
# Very spiky (see above where I saw spikes for first time)
# Looks difficult to predict
# Probably a good idea to modell low-cost predictable items separately and then model random big sales
# What is total revenue of company?
sales_train.groupby(["date_block_num"]).sum()["revenue"].plot(figsize=(20,5))
# I am working through the question list now - so maybe a bit random
# Slowly I start to think more about how actually to do the modelling
# At the moment I have lost a bit the track on how this will look like and how this actually works.

# Products out of sale:
#All item ids that are being sold in the last month in the train data
sales_train[sales_train.date_block_num==33]["item_id"]
print(sales_train[sales_train.date_block_num==33]["item_id"].nunique())
# All item ids from all times
print(sales_train.item_id.nunique())

# Wow: more than 3/4 of items out of sale. But makes sense. Music titles, old programs, old consoles and PCs

# How about the test data?
print(test.item_id.nunique())
# 300 less than was sold in the month before
# Are there items that are new?
a=set(test.item_id)
b=set(sales_train[sales_train.date_block_num==33]["item_id"])
new_in_test_items=[x for x in a if x not in b]
print(len(new_in_test_items))
print(new_in_test_items[0:100])
print(sales_train[sales_train["item_id"]==8214])
# ok this is an item that is not often sold (only 2 times in dataset)
print(sales_train[sales_train["item_id"]==4893]) #(18 times in dataset)

# a) Let's check how many items are just sold <100 times - maybe different categories? FMCC vs. B2B

# Really new items:
c=set(sales_train["item_id"])
print(len(c))

new_in_test_items2=[x for x in a if x not in c]
print(len(new_in_test_items2))
print(new_in_test_items2[0:100])

print(sales_train[sales_train["item_id"]==83])
print(sales_train[sales_train["item_id"]==430])

# ok it is working. But funny that such low IDs appear for the first time. 
# b) Aren't the IDs an ever growing item? Or are certain ID-number-blocks reserved for categories?
# How many items are only sold rarely?
# Aren't the IDs an ever growing item? Or are certain ID-number-blocks reserved for categories?
sales_train[sales_train.item_category_id==11]
# Interesting, the same item (PS3) has many different product ids
print(sales_train[sales_train.items_english=="Igrovye konsoli - PS3"])
print(sales_train[sales_train.items_english=="Igrovye konsoli - PS3"]["item_id"].unique())
print(sales_train[sales_train.items_english=="Igrovye konsoli - PS3"]["item_id"].nunique())

print(sales_train[sales_train.items_english=="Igrovye konsoli - PS3"]["item_price"].unique())
print(sales_train[sales_train.items_english=="Igrovye konsoli - PS3"]["item_price"].nunique())
prices_PS3=sales_train[sales_train.items_english=="Igrovye konsoli - PS3"]["item_price"]

plt.figure(figsize=(20, 8), dpi=80)
plt.scatter(prices_PS3.index, prices_PS3,s=0.1)
# quite a spread
# Why so many item_ids? Does this correlate price per ID?
sales_train["value"]=1
pivot=pd.pivot_table(sales_train[sales_train.items_english=="Igrovye konsoli - PS3"], values="value", index=["item_id"], columns="item_price", fill_value=0) 
# No, it is not one item id per price

print(new_in_test_items)
# Now examine price changes and price developments of items
sales_train.groupby(["item_id","item_price"]).sum()
# Now examine the categories

# How many items in each category
#sales_train.groupby("category_id","item_id").count()
# Save the final dataset (to not always calculate everything above when restarting the Kernel)
sales_train.to_csv('mycsvfile.csv',index=False)

print(os.listdir("../"))
print(os.listdir("../working"))
# Load the pre-processed dataset to continue from here without always calculating everything above
train=pd.read_csv('../working/mycsvfile.csv')

sample_submission.head(100)
# Do I have to predict only the amount, not the revenue / price???
# Indeed "We are asking you to predict total sales for every product and store in the next month."
# So price information is helpful only as a feature
test.head()
# submission.to_csv('submission.csv',index=False)
# Very first submission resulted in a score of 1,8 something - an extremely bad score place 863 of 950
# I had the sum of items per month completely wrong
interim= sales_train[sales_train["date_block_num"]==33].groupby(["shop_id", "item_id"],as_index=False).sum()[["shop_id","item_id","item_cnt_day"]]
interim["item_cnt_day"].clip(0,20,inplace=True)
interim
# the item_cnt_month are not properly entered into the grid
interim2=pd.merge(test, interim, how="left", left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])
interim2.info()
interim2=interim2[["ID","item_cnt_day"]]
interim2.columns=["ID","item_cnt_month"]
interim2.fillna(0,inplace=True)
interim2
interim2.to_csv('submission2.csv',index=False)
# "Your submission scored 16.05675" ? -> Forgot to clip values
# 1.96214: Still worse than before and lower than mentioned!? -> I had the summing up completely wrong

# v2: Yes: 1.02172, place 376
#Let's try November values from last year
interim= sales_train[sales_train["date_block_num"]==22].groupby(["shop_id", "item_id"],as_index=False).sum()[["shop_id","item_id","item_cnt_day"]]
interim["item_cnt_day"].clip(0,20,inplace=True)
interim2=pd.merge(test, interim, how="left", left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])
interim2=interim2[["ID","item_cnt_day"]]
interim2.columns=["ID","item_cnt_month"]
interim2.fillna(0,inplace=True)
interim2.to_csv('submission3.csv',index=False)

# Interesting: 1.60233: Much worse. So October-November seasonal effect smaller than November-November between 2 years
# Before starting with more complicated methods lets model something meaningful for items that are sold for the first time
# And lets check if there werer items in the data sold in September, but not in October, but then in November again

# Data that was for the first time in test:
new_in_test_items2
print(test[test.item_id.isin(new_in_test_items2)])
# No further information on the items. Can categories be learned from ids? At the example of 5320
sales_train[sales_train.item_id.isin(range(5310,5330))].groupby("item_id").max()
# Not really.
# One more example 3405-3408
sales_train[sales_train.item_id.isin(range(3400,3415))].groupby("item_id").max()
# Here the category seems to be Igry PC. But prices and counts vary very much

# Idea could be:
# - if category before and after the ID is the same use the average of this category
# - if they do not match take some average (eg of both categories or of all categories)
# Now for the next idea: items that were sold in september but not october
september = set(sales_train[sales_train.date_block_num==32].item_id)
october = set(sales_train[sales_train.date_block_num==33].item_id)
november = set(test.item_id)
sep_but_not_oct=[x for x in november if x not in october and x not in new_in_test_items2]
sep_but_not_oct
print(len(september))
print(len(october))
print(len(november))
print(len(sep_but_not_oct))
# 746 items were we could use the september figures
interim= sales_train[sales_train["date_block_num"]==33].groupby(["shop_id", "item_id"],as_index=False).sum()[["shop_id","item_id","item_cnt_day"]]
interim["item_cnt_day"].clip(0,20,inplace=True)

interim2=sales_train[sales_train["date_block_num"]==32].groupby(["shop_id", "item_id"],as_index=False).sum()[["shop_id","item_id","item_cnt_day"]]
interim2=interim2[interim2.item_id.isin(sep_but_not_oct)]
interim2["item_cnt_day"].clip(0,20,inplace=True)

interim3=pd.merge(test, interim, how="left", left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])
interim3=pd.merge(interim3, interim2, how="left", left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])

interim3.fillna(0,inplace=True)

interim3["item_cnt_month"]=interim3[["item_cnt_day_x","item_cnt_day_y"]].max(axis=1) 

print(interim3)
interim4=interim3[["ID","item_cnt_month"]]
print(interim4)
interim4.to_csv('submission4.csv',index=False)

# Scored worse: 1.16602 - but I am not sure that the operations above did what I want them to do. 
# Have to check more closely in next working session
sales_train.groupby(["item_id","shop_id","date_block_num"],as_index=False).sum()[["item_id","shop_id","date_block_num","item_cnt_day"]]

# Need to figure out how to only include the latest date_block_num per item per shop in table as lookup
# value for test
abc=sales_train.groupby(["item_id","shop_id","date_block_num"],as_index=False).sum()[["item_id","shop_id","date_block_num","item_cnt_day"]].head(10)
abc
# Now find out how to get only the rows where item_id and shop_id are the same and date_block_num is max
# I.e. I want to have a table with the most recent item_cnt of a specific item per shop as lookup table to fill this
# most recent item_cnt into the test.
abc.groupby(["item_id","shop_id"]).last()
#YES!
interim5=sales_train.groupby(["item_id","shop_id","date_block_num"],as_index=False).sum()[["item_id","shop_id","date_block_num","item_cnt_day"]].groupby(["item_id","shop_id"],as_index=False).last()
interim5=interim5[["item_id","shop_id","item_cnt_day"]]
interim5
interim5["item_cnt_day"].clip(0,20,inplace=True)
interim6=pd.merge(test, interim5, how="left", left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])
interim6=interim6[["ID","item_cnt_day"]]
interim6.columns=["ID","item_cnt_month"]
interim6.fillna(0,inplace=True)
interim6.to_csv('submission5.csv',index=False)
interim6
# scored 1.38739, better than september and october together but worse than october alone. Probably some items way to old
# lets restrict age
interim=sales_train.groupby(["item_id","shop_id","date_block_num"],as_index=False).sum()[["item_id","shop_id","date_block_num","item_cnt_day"]].groupby(["item_id","shop_id"],as_index=False).last()
interim=interim[interim["date_block_num"]<25]
interim=interim[["item_id","shop_id","item_cnt_day"]]
interim["item_cnt_day"].clip(0,20,inplace=True)
interim7=pd.merge(test, interim, how="left", left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])
interim7=interim7[["ID","item_cnt_day"]]
interim7.columns=["ID","item_cnt_month"]
interim7.fillna(0,inplace=True)
interim7.to_csv('submission6.csv',index=False)
interim7
# Only slightly better: 1.32017
# So this is a dead end. Lets start with the modelling
sales_train.to_csv('sales_train.csv',index=False)