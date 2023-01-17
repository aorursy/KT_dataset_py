

# Import required libraries

import os.path

from os.path import join, splitext

from tempfile import mkdtemp

import pandas as pd

from werkzeug.utils import secure_filename

import time

# Importing packages

from selenium import webdriver

# import pandas as pd

from bs4 import BeautifulSoup

# from selenium.webdriver.common.keys import Keys



def GetExtractedData(inputCompany):

    customoptions = webdriver.ChromeOptions()

    customoptions.add_argument('--ignore-certificate-errors')

    # customoptions.add_argument('--incognito')

    # customoptions.add_argument('--headless')



    driver = webdriver.Chrome('D:\Downloads\chromedriver_win32\chromedriver.exe', options=customoptions)



    time.sleep(5)  

    driver.get('https://www.testsamplesite.com/sitequote/' + inputCompany)

    time.sleep(5)

    driver.maximize_window()

    time.sleep(5)  # Let the user actually see something!

    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'lxml')

    time.sleep(5)



    # print(soup.prettify())

    reviews_selector = soup.find_all('section', class_='left__1234_description__5678')

    

    time.sleep(5)

    userid = reviews_selector

    time.sleep(20)

    # print(userid[0].text)

    time.sleep(5)  # Let the user actually see something!

    driver.quit()

    return userid[0].text



#=========================================================================================


