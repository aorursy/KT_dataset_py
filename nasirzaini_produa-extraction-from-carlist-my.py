import pandas as pd

from time import sleep

from random import randint

from requests import get

from bs4 import BeautifulSoup

import warnings

warnings.warn("Warning Simulation")

from IPython.core.display import clear_output

import time 

start_time = time.time()

requests = 0

pages = [str(i) for i in range(0,1)]

print(pages)



cars = []

years = []

mileages = []

transmissions =[]

makes = []

models =[]

prices =[]

listing_ids = []



for page in pages:



        # Make a get request

       

        newpage = get('https://www.carlist.my/used-cars-for-sale/perodua/malaysia?page_number=' +page + '&page_size=25')

        

        

            

            

        # Pause the loop

        sleep(randint(8,15))



        # Monitor the requests

        requests += 1

        elapsed_time = time.time() - start_time

        print('Request:{}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))

        clear_output(wait = True)



        # Throw a warning for non-200 status codes

        if newpage.status_code != 200:

            warn('Request: {}; Status code: {}'.format(requests, newpage.status_code))



        # Break the loop if the number of requests is greater than expected

        if requests > 2000:

            warn('Number of requests was greater than expected.')

            break



        # Parse the content of the request with BeautifulSoup

     

        soup = BeautifulSoup(newpage.content, 'html.parser')



        # Select all the 50 movie containers from a single page

        car_list = soup.find(id="classified-listings-result")



        # For every movie of these 50

        car = [c["data-display-title"] for c in car_list.select("article")]

        cars.extend (car)

        

        year = [y["data-year"] for y in car_list.select("article")]

        years.extend(year)

        

        mileage = [m["data-mileage"] for m in car_list.select("article")]

        mileages.extend(mileage)

        

        

        transmission = [t["data-transmission"] for t in car_list.select("article")]

        transmissions.extend(transmission)

        

        make = [mk["data-make"] for mk in car_list.select("article")]

        makes.extend(make)

        

        model = [md["data-model"] for md in car_list.select("article")]

        #price = [pr["data-price"] for pr in car_list.select("article")]

        models.extend (model)

        

        price = [pr.get_text() for pr in car_list.select(".listing__content .flexbox .listing__price")]

        prices.extend(price)

        

        listing_id = [lis["data-listing-id"] for lis in car_list.select("article")]

        listing_ids.extend(listing_id)

        

        
print(len(models))

print(len(listing_ids))

print(len(makes))

print(len(years))

print(len(mileages))

print(len(prices))

print(len(transmissions))

perodua = pd.DataFrame({'id': listing_ids,

'model' : models,

'make' : makes ,

'version' : cars,

'year': years,

'mileage': mileages,

'price': prices,

'transmission': transmissions

})

print(perodua.info())

perodua.head(10)
perodua2 = perodua
perodua2.loc[:, 'version'] = perodua2['version'].str[12:].astype(str)

perodua2.head()
perodua2  = perodua2.to_csv (r'perodua_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
