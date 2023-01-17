import sqlalchemy

import pandas as pd

from pandas.io import sql

import calendar

import os

import requests

import json

from bs4 import BeautifulSoup





from time import sleep

import random



import re

import ast

import datetime as datetime

from pytz import timezone

import requests, zipfile, io



import csv

import numpy as np



import zipfile

import sys

import time

#import camelot

from urllib3.util.retry import Retry

from requests.adapters import HTTPAdapter

from selenium import webdriver

import warnings

warnings.filterwarnings('ignore')

import xml.etree.ElementTree as ET







from selenium import webdriver

from selenium.common.exceptions import TimeoutException

from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys



from selenium.common.exceptions import WebDriverException





import re

import datetime as datetime

from pytz import timezone

from bs4 import BeautifulSoup

import json

import requests

import os

import sqlalchemy

import pandas as pd



import random

import time

import numpy as np

import math

from selenium.common.exceptions import NoSuchElementException

from selenium import webdriver

from selenium.webdriver.chrome.options import Options



import shutil

from selenium.webdriver.common.action_chains import ActionChains

from selenium import webdriver

from tqdm import tqdm

import sys



site_url = "https://www.talabat.com/uae/restaurants"

driver = webdriver.Chrome('/Users/Bharath/chromedriver')  

driver.get(site_url)

driver.maximize_window()

final_df = pd.DataFrame()



i=1

while (i<301):

    driver.find_element_by_tag_name('body').send_keys(Keys.END)

    i = i+1

    sleep(0.2)

main_soup = BeautifulSoup(driver.page_source)



soup = main_soup.findAll('div', attrs = {'class':'col-lg-3 col-md-3 col-sm-4 col-xs-8 new-rest-item ng-scope'})
len(soup)
Restaurant = []

Cuisine = []

for i in range(len(soup)):

    Restaurant.append(soup[i].findAll('p')[0].text)

    Cuisine.append(soup[i].findAll('p')[1].text)

    

    
restaurant_database = pd.DataFrame({'Restaurant':Restaurant,'Cuisine':Cuisine})
restaurant_database.to_excel('Talabat_restaurants_data.xlsx',index=False)
restaurant_database.to_csv('Talabat_restaurants_data.csv',index=False)
driver.quit()