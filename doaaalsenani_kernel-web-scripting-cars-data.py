import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from html.parser import HTMLParser
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from time import sleep
from random import randint
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
link_list=[]

llist_price=[]
list_mark=[]
list_model=[]
list_year=[]
list_engine_capacity=[]
list_mileage=[]
list_fuel=[]
list_drive=[]
list_type_drive=[]
list_transmission=[]
for i in range(0,100):
    link=('https://www.tc-v.com/used_car/all/all/?pn={}'.format(i))
    link_list.append(link)
driver_1 = webdriver.Chrome(executable_path='chromedriver/chromedriver')
for i in range(0,100):
    link=link_list[i]
    driver_1.get(link) 
    html = driver_1.page_source
    soup = BeautifulSoup(html,'html.parser')
    item_list1 = soup.find_all('div',class_="vehicle-item-info-area")
    sleep(randint(2,10))
    
    
    for i in item_list1:
    
        model=i.find( 'span', class_="vehicle-item-ttl-model").text
        list_model.append(model)

        year=i.find( 'span', class_="vehicle-item-ttl-year").text
        list_year.append(year)

        mark=i.find( 'span', class_="vehicle-item-ttl-make").text
        list_mark.append(mark)

        price=i.find( 'div', class_="vehicle-item-main-info").text.split()[2]
        llist_price.append(price)


        engine_capacity=i.find('div', class_="vehicle-item-main-info").text.split()[1]
        list_engine_capacity.append(engine_capacity)

        mileage=i.find('div', class_="vehicle-item-main-info").text.split()[1]
        list_mileage.append(mileage)


        fuel=i.find('div', class_="vehicle-item-sub-info").text.split()[0]
        list_fuel.append(fuel)

        transmission=i.find('div', class_="vehicle-item-sub-info").text.split()[0]
        list_transmission.append(transmission)

        drive=i.find('div',class_="vehicle-item-sub-info").text.split()[0]
        list_drive.append(drive)

        type_drive=i.find('div', class_="vehicle-item-sub-info").text.split()[0]
        list_type_drive.append(type_drive)
df = pd.DataFrame({'price':llist_price,
                         'mark': list_mark,
                         'mode': list_model,
                         'year': list_year,
                         'engine_capacity': list_engine_capacity,
                         'mileage': list_mileage,
                         'fuel': list_fuel,
                         'drive': list_drive,
                         'type_drive':list_type_drive,
                         'transmission': list_transmission })
df.to_csv('./cars_datasets.csv')
cars_data = pd.read_csv('../input/cars_datasets.csv')
cars_data.head()
cars_data.tail()
cars_data.info()
missing= cars_data.isnull().sum().sort_values(ascending=False)
percentage = (cars_data.isnull().sum()/ cars_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])
missing_data.head(10)
cars_data['capacity_1'] = cars_data['engine_capacity'].apply(lambda x: x.split('Mileage')[0]
                                                             .replace('Capacity','')
                                                             .replace(',','')
                                                             .replace('cc',''))

cars_data['mileage_1'] = cars_data['engine_capacity'].apply(lambda x: x.split('Mileage')[1]
                                                           .replace('kmFOB','')
                                                           .replace(',',''))
cars_data['price_1'] = cars_data['price'].apply(lambda x: x.split('Price')[1]
                                               .replace('US$','')
                                               .replace(',','')
                                               .replace('Estimated',''))
# hand_drive
cars_data['drive'] =cars_data['drive'].str.replace('RHD','')
cars_data['drive'] =cars_data['drive'].str.replace('Center','')
cars_data['drive'] =cars_data['drive'].str.replace('LHD','')
# fuel
cars_data['drive'] =cars_data['drive'].str.replace('Gasoline','')
cars_data['drive'] =cars_data['drive'].str.replace('Hybrid','')
cars_data['drive'] =cars_data['drive'].str.replace('CNG','')
cars_data['drive'] =cars_data['drive'].str.replace('CNG','')
cars_data['drive'] =cars_data['drive'].str.replace('LPG','')
cars_data['drive'] =cars_data['drive'].str.replace('Diesel','')
# transmission
cars_data['drive'] =cars_data['drive'].str.replace('AT','')
cars_data['drive'] =cars_data['drive'].str.replace('MT','')
cars_data['drive'] =cars_data['drive'].str.replace('Automanual','')
cars_data['drive'] =cars_data['drive'].str.replace('CVT','')
# fuel
cars_data['type_drive'] =cars_data['type_drive'].str.replace('Gasoline','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('Hybrid','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('CNG','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('CNG','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('LPG','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('Diesel','')
# transmission
cars_data['type_drive'] =cars_data['type_drive'].str.replace('AT','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('MT','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('Automanual','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('CVT','')
# drive
cars_data['type_drive'] =cars_data['type_drive'].str.replace('2WD','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('4WD','')
cars_data['type_drive'] =cars_data['type_drive'].str.replace('AWD','')
# hand_drive
cars_data['fuel'] = cars_data['fuel'].str.replace('LHD','')
cars_data['fuel'] = cars_data['fuel'].str.replace('RHD','')
cars_data['fuel'] = cars_data['fuel'].str.replace('Center','')
# transmission
cars_data['fuel'] = cars_data['fuel'].str.replace('AT','')
cars_data['fuel'] =  cars_data['fuel'].str.replace('MT','')
cars_data['fuel'] = cars_data['fuel'].str.replace('Automanual','')
cars_data['fuel'] = cars_data['fuel'].str.replace('CVT','')
# drive
cars_data['fuel'] = cars_data['fuel'].str.replace('2WD','')
cars_data['fuel'] = cars_data['fuel'].str.replace('4WD','')
cars_data['fuel'] = cars_data['fuel'].str.replace('AWD','')

cars_data['fuel'] = cars_data['fuel'].str.replace('CV','')
cars_data['fuel'] = cars_data['fuel'].str.replace('M','')
# hand_drive
cars_data['transmission'] =cars_data['transmission'].str.replace('RHD','')
cars_data['transmission'] =cars_data['transmission'].str.replace('Center','')
cars_data['transmission'] =cars_data['transmission'].str.replace('LHD','')
# fuel
cars_data['transmission'] =cars_data['transmission'].str.replace('Gasoline','')
cars_data['transmission'] =cars_data['transmission'].str.replace('Hybrid','')
cars_data['transmission'] =cars_data['transmission'].str.replace('CNG','')
cars_data['transmission'] =cars_data['transmission'].str.replace('CNG','')
cars_data['transmission'] =cars_data['transmission'].str.replace('LPG','')
cars_data['transmission'] =cars_data['transmission'].str.replace('Diesel','')
# drive
cars_data['transmission'] = cars_data['transmission'].str.replace('2WD','')
cars_data['transmission'] = cars_data['transmission'].str.replace('4WD','')
cars_data['transmission'] = cars_data['transmission'].str.replace('AWD','')

cars_data['transmission'] = cars_data['transmission'].str.replace('Automanual','AT')
cars_data['fuel'] = cars_data['fuel'].apply(lambda x:x.lower())
cars_data['type_drive'] =cars_data['type_drive'].apply(lambda x:x.lower())
cars_data['drive'] =cars_data['drive'].apply(lambda x:x.lower())
cars_data['transmission'] = cars_data['transmission'].apply(lambda x:x.lower()) 
cars_data['mark'] =cars_data['mark'].apply(lambda x:x.lower())
cars_data['mode'] = cars_data['mode'].apply(lambda x:x.lower())
cars_data['drive'].unique() #   has 162 empty values
cars_data['drive'].value_counts()
#Drop empty values in drive 
cars_data=cars_data.loc[cars_data['drive'] != '']
#Drop unknownFOB values in mileage
cars_data.loc[cars_data['mileage_1'] == 'unknownFOB']
cars_data=cars_data.loc[cars_data['mileage_1'] != 'unknownFOB']
# reset index after drop
cars_data.reset_index(drop=True, inplace=True)
#Drop columns
del cars_data["engine_capacity"]
del cars_data["mileage"]
del cars_data['price']
del cars_data['Unnamed: 0']
cars_data.rename(columns={'capacity_1':'engine_capacity',
                         'mileage_1':'mileage',
                          'price_1':'price',
                          'type_drive':'hand_drive'
                          },inplace=True)
cars_data.head(1)
cars_data['engine_capacity']=cars_data['engine_capacity'].apply(pd.to_numeric)
print('The type of engine_capacity:',cars_data['engine_capacity'].dtypes)
cars_data['mileage']=cars_data['mileage'].apply(pd.to_numeric)
print('The type of mileage:',cars_data['mileage'].dtypes)
cars_data['price']=cars_data['price'].apply(pd.to_numeric)
print('The type of price:',cars_data['price'].dtypes)
# to show the duplicate row
duplicateRowsDF = cars_data[cars_data.duplicated()]
duplicateRowsDF
# to drop duplicates in data 
cars_data.drop_duplicates(keep=False,inplace = True)
cars_data.shape
cars_data = cars_data[["price", "mark", "mode","year","mileage","engine_capacity","transmission","drive","hand_drive","fuel"]]
cars_data.to_csv('./final_cars_datasets.csv')
display(cars_data.head())
display(cars_data.shape)