import pandas as pd
import itertools 
from bs4 import BeautifulSoup
import requests
from requests import get
import time
from random import seed
from random import random
from random import randint

# specify the url format
url = 'https://www.pararius.com/apartments/amsterdam/page-'
# initialize a list called houses 
houses = []
# initialize variable count at 1
count = 1

# first while loop that will run 100 times (adjust this to how many pages you want to scrape)
while count <= 100:
    # initialize variable new_count at 0
    new_count = 0
    # if loop that specifies the first page separately (many websites have a first page url format different than other pages)
    if count == 1:
        first_page = 'https://www.pararius.com/apartments/amsterdam/page-1'
        # request the response
        response = get(first_page)
        # parse through the html 
        html_soup = BeautifulSoup(response.text, 'html.parser')
        # in the html of the page, find all the bins with <li> and class:
        house_data = html_soup.find_all('li', class_="search-list__item search-list__item--listing")
        # I like to print where the program is on the screen so we can follow its progress and where any errors happened
        print(first_page)
        
        # if the response was not empty (if something was actually scraped)
        if house_data != []:
            # add to the list houses
            houses.extend(house_data)
            # random wait times
            value = random()
            scaled_value = 1 + (value * (9 - 5))
            print(scaled_value)
            time.sleep(scaled_value)
    # pages other than the first
    elif count != 1:
    # collect four and wait random times 
        url = 'https://www.pararius.com/apartments/amsterdam/page-' + str(count)
        print(url)
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        print(response)
        house_data = html_soup.find_all('li', class_="search-list__item search-list__item--listing")

        if house_data != []:
            houses.extend(house_data)
            value = random()
            scaled_value = 1 + (value * (9 - 5))
            print(scaled_value)
            time.sleep(scaled_value)

        # if you get empty response, stop the loop
        else:
            print('empty')
            break
            

    count += 1
count = 0
house_price = []
rental_agency = []
location = []
city = []
bedrooms = []
surface = []

n = int(len(houses)) - 1

while count <= n:
    
    num = houses[int(count)]
    
    price = num.find_all('span',{"class":"listing-search-item__price"})[0].text
    house_price.append(price)
    df1 = pd.DataFrame({'house_price':house_price})
    df1['house_price'] = df1['house_price'].str.replace("\D","")
    df1['house_price'] = df1['house_price'].str.replace("per month","")
    
    try:
        agency = num.find_all('a', href=True)[2].text
    except IndexError:
        agency = 'none'
    rental_agency.append(agency)
    df2 = pd.DataFrame({'rental_agency':rental_agency})
    

    postcode = num.find('div',{"class":"listing-search-item__location"}).text
    location.append(postcode)
    df3 = pd.DataFrame({'postcode':location})
    df3['postcode'] = df3['postcode'].str.replace("\nApartment\n ","")
    df3['postcode'] = df3['postcode'].str.replace("\n","")
    df3['postcode'] = df3['postcode'].str.replace("\s","")
    df3['postcode'] = df3['postcode'].str.replace("               ","")
    df3['postcode'] = df3['postcode'].str.replace("new","")
    df3['postcode'] = df3['postcode'].str[0:6]
    
    bedrooms_num = num.find_all('dd',{"class":"illustrated-features__description"})[1].text
    bedrooms.append(bedrooms_num)
    df4 = pd.DataFrame({'bedrooms':bedrooms})
    df4['bedrooms'] = df4['bedrooms'].str.replace("\D","")
    
    size = num.find_all('dd',{"class":"illustrated-features__description"})[0].text
    surface.append(size)
    df5 = pd.DataFrame({'surface':surface})
    df5['surface'] = df5['surface'].str.replace("\D","")
    
    print(count)
    count += 1
    
result = pd.concat([df1, df2], axis=1, sort=False)
result2 = pd.concat([result, df3], axis=1, sort=False)
result3 = pd.concat([result2, df4], axis=1, sort=False)
dfa = pd.concat([result3, df5], axis=1, sort=False)
df = dfa.copy()

from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
import geopy.geocoders
from geopy.geocoders import Nominatim
import geopy
import geopandas
import pandas as pd
import time 


list_of_points = []

df['address'] = df['postcode']

df['address2'] = df['postcode'].str.replace('\s','')

locator = Nominatim(user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9')

df2 = pd.DataFrame(columns=['house_price', 'rental_agency',
                            'postcode','bedrooms','surface',
                            'address','address2'])
n = int(len(houses)) - 1

count = 1
while count <= n: 
    if count == 0:
        df_new = df[0:1]
    else:
        a = int(count)
        n = int(count) +1
        print(a)
        print(n)
        df_new = df[a:n]
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    try:
        df_new['location'] = df_new['address'].apply(geocode)
        df_new['point'] = df_new['location'].apply(lambda loc: tuple(loc.point) if loc else None)
        df_new[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df_new['point'].tolist(), index=df_new.index)
    except ValueError:
        try:
            df_new['location'] = df_new['address2'].apply(geocode)
            print('trying second address')
            df_new['point'] = df_new['location'].apply(lambda loc: tuple(loc.point) if loc else None)
            df_new[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df_new['point'].tolist(), index=df_new.index)
        except ValueError:
            df_new = df_new.dropna(subset=['location'])
            list_of_points.append(a)
                
    df2 = pd.concat([df2, df_new], sort=False)
    time.sleep(1)
    count += 1
# Importing the complete datasets to skip running the time-consuming operations above
import pandas as pd
df = pd.read_excel("../input/amsterdam-rental-listings/amsterdam_rental_data1.xlsx")
df2 = pd.read_excel("../input/amsterdam-rental-listings/amsterdam_rental_data2.xlsx")
del df2['address2']

df2 = pd.concat([df,df2])
df2 = df2.drop_duplicates()
df2 = df2.dropna()
# Getting distance of the apartments from the city center
import geopy.distance

df5 = df2.copy()

df5['lat2'] = 52.370216
df5['lon2'] = 4.895168

df5['coord1'] = df5['latitude'].astype(str) + ',' + df5['longitude'].astype(str)
df5['coord2'] = df5['lat2'].astype(str) + ',' + df5['lon2'].astype(str)

def get_distance(coord1,coord2):
    dist = geopy.distance.vincenty(coord1, coord2).km
    return dist
df5['dist'] = [get_distance(**df5[['coord1','coord2']].iloc[i].to_dict()) for i in range(df5.shape[0])]
# deleting columns we don't need anymore

del df5['address']
del df5['altitude']
del df5['latitude']
del df5['longitude']
del df5['point']
del df5['lat2']
del df5['lon2']
del df5['coord1']
del df5['coord2']
del df5['location']
import requests
import json

rating = []
zipcode = []
prices =[]

api_key=''
headers = {'Authorization': 'Bearer %s' % api_key}
url='https://api.yelp.com/v3/businesses/search'

offset = 0
while offset <= 1000:
    print(offset)
    params={'term':'Restaurants', 'location': 'amsterdam', 'limit': 50, 'offset': offset}
    req = requests.get(url, params=params, headers=headers)
    parsed = json.loads(req.text)
    for n in range(0,51):
        try:
            price_data = parsed["businesses"][n]['price']
            ratings_data = parsed["businesses"][n]['rating']
            zipcode_data = parsed["businesses"][n]["location"]["zip_code"]

            rating.append(ratings_data)
            zipcode.append(zipcode_data)
            prices.append(price_data)
        except:
            pass
    offset += 1

yelp1 = pd.DataFrame({'rating':rating,'zipcode':zipcode,'prices':prices})

yelp1 = pd.read_excel('../input/yelp-amsterdam/yelp_updated.xlsx')
## get the length of the zipcode
yelp1['zip_len'] = yelp1.zipcode.str.len()
## NL postcodes have more than 4 digits, so make sure we only keep those
yelp1 = yelp1[yelp1['zip_len'] > 4]
## I like to copy data frames when I make radical changes to data that took a while to be generated
## to ensure I can go back to the original data if I need
yelp = yelp1.copy()

## elimate whitespaces
yelp['postcode2'] = yelp['zipcode'].str.replace('\s','')
## only get 4 first digits of the postcode
yelp['postcode2']= yelp.postcode2.str[0:4]

## prices in yelp are represented by $, $$, $$$ or $$$$ so the length of the string can tell us how 
## expensive a restaurant is
yelp['price_len'] = yelp.prices.str.len()

## group by and get means by postcode area
yelp_prices = yelp.groupby(['postcode2']).price_len.mean()
yelp_rate = yelp.groupby(['postcode2']).rating.mean()

## create two dataframes and transform them into dictionaries 
yelp_prices = pd.DataFrame(data=yelp_prices)
yelp_rate = pd.DataFrame(data=yelp_rate)
dict1 = yelp_prices.to_dict()['price_len']
dict2 = yelp_rate.to_dict()['rating']
df5['postcode'] = df5['postcode'].str.replace('\s','')
df5['postcode2'] = df5['postcode'].str[0:4]

## delete the non-digit characters of the postcodes and copy dataframe to amsmodel1
amsmodel1 = df5.copy()

## map the yelp price means and ratings means into our rental data frame
amsmodel1['yelp_prices'] = amsmodel1['postcode2'].map(dict1)
amsmodel1['yelp_ratings'] = amsmodel1['postcode2'].map(dict2)

## make sure all the integer columns are in fact integer types
amsmodel1['house_price'] = pd.to_numeric(amsmodel1['house_price'])
amsmodel1['bedrooms'] = pd.to_numeric(amsmodel1['bedrooms'])
amsmodel1['surface'] = pd.to_numeric(amsmodel1['surface'])
## how many columns do we have?
len(amsmodel1.columns)
amsmodel1 = amsmodel1.dropna()
## creating dummy variables for the categorical columns "rental agency" and "postcode"

dummies = pd.get_dummies(amsmodel1.postcode2,prefix=['p'])
amsmodel1 = pd.concat([amsmodel1,dummies],axis = 1)
dummies2 = pd.get_dummies(amsmodel1.rental_agency,prefix=['ag'])
amsmodel1 = pd.concat([amsmodel1,dummies2],axis = 1)

del amsmodel1['rental_agency']
del amsmodel1['postcode2']
del amsmodel1['postcode']

amsmodel1['house_price'] = pd.to_numeric(amsmodel1['house_price'])
amsmodel1 = amsmodel1.dropna()
len(amsmodel1.columns)
import numpy as np

target= np.array(amsmodel1['house_price'])
features = amsmodel1.drop('house_price', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

## RANDOM FOREST - KFOLD AND MODEL 

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
    
kf = KFold(n_splits=10,random_state=42,shuffle=True)
accuracies = []
for train_index, test_index in kf.split(features):

    data_train   = features[train_index]
    target_train = target[train_index]

    data_test    = features[test_index]
    target_test  = target[test_index]

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, criterion = 'mse',  bootstrap=True)
    
    rf.fit(data_train, target_train)

    predictions = rf.predict(data_test)

    errors = abs(predictions - target_test)

    print('Mean Absolute Error:', round(np.mean(errors), 2))
    
    mape = 100 * (errors / target_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
print('Average accuracy:', average_accuracy)
# SAVING THE DECISION TREE 

from sklearn.tree import export_graphviz
import pydot
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree_amsterdam.png')
y = rf.feature_importances_
list_y = [a for a in y if a > 0.01]
print(list_y)

list_of_index = []
for i in list_y:
    a = np.where(y==i)
    list_of_index.append(a)
    print(a)
list_of_index = [0,1,2,3,4,24,90,105,131,170,173,230,238,259,282]
col = []
for i in feature_list:
    col.append(i)
labels = []
for i in list_of_index:
    b = col[i]
    labels.append(b)
import matplotlib.pyplot as plt 

y = list_y
fig, ax = plt.subplots() 
width = 0.8
ind = np.arange(len(y)) 
ax.barh(ind, y,width, color="pink")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(labels, minor=False)
plt.title('Feature importance in Random Forest Regression')
plt.xlabel('Relative importance')
plt.ylabel('feature') 
plt.figure(figsize=(10,8.5))
fig.set_size_inches(10, 8.5, forward=True)
# create a new dataframe that is indexed like the trained model
newdata = pd.DataFrame().reindex_like(amsmodel1)
newdata.fillna(value=0, inplace=True)

# delete the variable to be predicted
del newdata['house_price']
newdata = newdata.iloc[[1]]

# insert information about your apartment 
newdata['bedrooms'] = 1
newdata['surface'] = 45
newdata['yelp_prices'] = 2.234043
newdata['yelp_ratings'] = 4.113475

# only change the number values in the postcode 
# and string values after the _ for the rental agency
newdata["['p']_1018"]= 1
newdata["['ag']_JLG Real Estate"] = 1

rf.predict(newdata)