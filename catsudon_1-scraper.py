import os

import chromedriver_binary



from tabulate import tabulate

from selenium import webdriver

from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import Select

from selenium.webdriver.chrome.options import Options

from selenium.common.exceptions import NoSuchElementException
# get_data() Function to get datasets 

def get_data():

 submit = driver.find_element_by_xpath('//*[@id="btnSubmit"]')

 driver.implicitly_wait(30)

 driver.execute_script("arguments[0].click();", submit)

 driver.implicitly_wait(5)

    

 try:

        save= driver.find_element_by_xpath('//*[@id="DL_CSV"]')

        driver.implicitly_wait(20)

        # multiple_down()

        driver.implicitly_wait(10)

        # download_loc()

        driver.implicitly_wait(10)

        driver.execute_script("arguments[0].click();", save)

        driver.implicitly_wait(30)

        print("Grabbed success...")

        driver.implicitly_wait(30)    

        

 except NoSuchElementException:

        print("No data found...")
# Does not work

# Direct to a directory

def download_loc():

    options = Options()

    prefs = {"download.default_directory": r"C:/Users/Yuki/Documents/Git Repositories/Airline Delay Prediction Project/Datasets"}

    options.add_experimental_option("prefs",prefs)

# Function not used

# Enables chromedriver to download multiple files

def multiple_down():

    options = Options()

    prefs = {'profile.default_content_setting_values.automatic_downloads': 1}

    options.add_experimental_option("prefs",prefs)



    
# Checklist 

def checklist():

    stats = driver.find_element_by_xpath('//*[@id="chkAllStatistics"]')

    # driver.find_element_by_id('chkAllStatistics').click()

    driver.execute_script("arguments[0].click();", stats)



    # Click 'All Days' checkbox

    months = driver.find_element_by_xpath('//*[@id="chkAllMonths"]')

    driver.execute_script("arguments[0].click();", months)





    days = driver.find_element_by_xpath('//*[@id="chkAllDays"]')

    driver.execute_script("arguments[0].click();", days)



    # Click '2019' checkbox

    year = driver.find_element_by_id('chkYears_32')

    driver.execute_script("arguments[0].click();", year)



    # Destination

    dest = driver.find_element_by_xpath('//*[@id="cboAirport"]/option[text()="Los Angeles, CA: Los Angeles International (LAX)"]').click()

url = "https://transtats.bts.gov/ONTIME/Arrivals.aspx"

variable = 0

airport = "Los Angeles, CA: Los Angeles International (LAX)"



# Downloaded datasets are located in "../Downloads" 

options = Options()

prefs = {'profile.default_content_setting_values.automatic_downloads': 1}

options.add_experimental_option("prefs",prefs)



# create a new Chrome session

driver = webdriver.Chrome(options=options)

driver.implicitly_wait(10)

driver.get(url)



checklist()



# 33 items in picketlist

while (variable < 33):

    variable =variable+1 

    airline = driver.find_element_by_xpath("//*[@id='cboAirline']/option["+str(variable)+"]").click()

    print(driver.find_element_by_xpath("//*[@id='cboAirline']/option["+str(variable)+"]").get_attribute("value"))

    print("Grabing data...")

    get_data()

else:

    print("")

    print("Fin.")