import requests as rq

import csv

import bs4

import numpy as np

import pandas as pd

import random

from selenium import webdriver
chromeOptions = webdriver.ChromeOptions()

prefs = {'profile.default_content_setting_values': {'cookies': 2, 'images': 2, 'javascript': 2, 

                            'plugins': 2, 'popups': 2, 'geolocation': 2, 

                            'notifications': 2, 'auto_select_certificate': 2, 'fullscreen': 2, 

                            'mouselock': 2, 'mixed_script': 2, 'media_stream': 2, 

                            'media_stream_mic': 2, 'media_stream_camera': 2, 'protocol_handlers': 2, 

                            'ppapi_broker': 2, 'automatic_downloads': 2, 'midi_sysex': 2, 

                            'push_messaging': 2, 'ssl_cert_decisions': 2, 'metro_switch_to_desktop': 2, 

                            'protected_media_identifier': 2, 'app_banner': 2, 'site_engagement': 2, 

                            'durable_storage': 2}}

chromeOptions.add_experimental_option('prefs', prefs)



driver = webdriver.Chrome(executable_path=r'C:\Users\manna\Desktop\lco-bootcamp\Day17-webscrapping-numpybasics2\Assignment\chromedriver.exe', options=chromeOptions)
def get_devices_list(link_url = r'https://www.gsmarena.com/makers.php3'):

    try:

        global devices_urls

        devices_urls = []

        source_code=rq.get(link_url)

    #     driver.get(link_url)

    #     plain_text=driver.page_source

        plain_text=source_code.text

        soup = bs4.BeautifulSoup(plain_text)

        for table in soup.find_all('div',{'class':'st-text'}):

               for td in table.find_all('td'):

                    for anc in td.find_all('a'):

                        anc_src = r'http://www.gsmarena.com/' + anc.get('href')

                        print("\nwe are on", list(anc.stripped_strings)[0], "page")

                        brand_devices(anc_src)

                        print(list(anc.stripped_strings)[0], "page ends")

        print("list Completed")

    except Exception as e:

        print(str(e)) 
def brand_devices(hrefs):

    try:

        global devices_urls

        source_code=rq.get(hrefs)

        plain_text=source_code.text

    #     driver.get(hrefs)

    #     plain_text=driver.page_source

        soup = bs4.BeautifulSoup(plain_text)

        for link in soup.find_all('div',{'class':'makers'}):

            for li in link.find_all('li'):

                for anc in li.find_all('a'):

                    anc_src = r'http://www.gsmarena.com/' + anc.get('href')

                    devices_urls.append(anc_src)

                    if li == link.find_all('li')[-1]:

                        print("device urls listed on the page added to the list and page ends...calling next page")

                        if soup.find_all('a',{'class':'pages-next'}):

                            for next_page in soup.find_all('a',{'class':'pages-next'}):

                                next_page_src = r'http://www.gsmarena.com/' + next_page.get('href')

                                brand_devices(next_page_src)

    except Exception as e:

        print(str(e))

    finally:

        store_data_list()
def store_data_list():

    try:

        df = pd.DataFrame(devices_urls, columns=["devices list"])

        df.to_csv("devices-list.csv", index=False)

        print("stylesheet created!!")

    except Exception as e:

        print(str(e))    

    

def get_device_features():

    try:

        global all_products, all_products_heading

        all_products = []

        all_products_heading = []

        data =pd.read_csv("devices-list1.csv")

        devices_list = data['devices list'].values

        for device in devices_list:

            device_features(device)

        print("Dataset completed!!")

    except Exception as e:

        print(str(e)) 

    finally:

        make_dataset()

    

def device_features(hrefs):

    try:

        global all_products, all_products_heading

        driver.get(hrefs)

        pt=driver.page_source

        soupy= bs4.BeautifulSoup(pt, "html.parser")

        tables = soupy.findAll('table')

        product_name = list(soupy.find("h1",{'class':'specs-phone-name-title'}).stripped_strings)

        specs = [product_name[0]]

        features_heads = ['Product Name']

        for table in tables:

            for features_head in table.findAll("td",{'class':'ttl'}):

                if list(features_head.stripped_strings) == []:

                    features_head.append('Untitled')

                    features_heads.append(list(features_head)[1])

                else:

                    features_heads.append(list(features_head.stripped_strings)[0])





            for spec in table.findAll("td",{'class':'nfo'}):

                if list(spec.stripped_strings) == []:

                    spec.append('_')

                    specs.append(list(spec)[0])

                else:

                    specs.append(list(spec.stripped_strings)[0])



        all_products_heading.append(features_heads)

        product_data = dict(zip(features_heads, specs))

        all_products.append(product_data)

        print("product added")

    except Exception as e:

        print(str(e))    

    

def make_dataset():

    try:

        global all_products, all_products_heading

        product_cols = []

        for i in all_products_heading:

            product_col_len = len(i)

            product_cols.append(product_col_len)



        index_max = np.argmax(product_cols)

        columns = all_products_heading[index_max]

        df = pd.DataFrame(all_products , columns = columns)

        df.to_csv("gsmarena_dataset.csv", index=False)

        print("dataset created!!")

    except Exception as e:

        print(str(e))





get_devices_list()
get_device_features()