# !pip install selenium
import re #Regular expression
import requests
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

## Step 1 : Download Html page 
response = requests.get('https://shop.tamimimarkets.com/category/food--beverages')
response.status_code
# Get scroll height
# Scroll down to bottom 

# open browser
#driver = webdriver.Chrome(executable_path='chromedriver/chromedriver.dms')
driver = webdriver.Chrome(executable_path='chromedriver/chromedriver')
driver.get('https://shop.tamimimarkets.com/category/food--beverages')

sleep(3) # sleep for few seconds to make sure that the initial page is opened



#**********************************************************************************************************#
################################## Start of the scroll page code ###########################################

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to scroll the page down
    sleep(5)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    #print ('new height',new_height)  # print new height after scroll for debugging purposes
    
    if new_height > 101366:
        break
    last_height = new_height
    
################################## End of the scroll page code #############################################
#**********************************************************************************************************#


html = driver.page_source
driver.close() # close the opened browser page after done.
# Beautiful Soup code
response = BeautifulSoup(html,'html')
# Create response object finding all 'div' which is where data we need 
item = response.find_all('div', attrs={'class': 'Product__StyledContainer-sc-13egllk-2 gBoTUw'})
len(item)
# create empty lists to handle all the data
prices=[]
company_name = []
product_name = []
# Loop through item to get prices,company_name and product_name
for i in item:
    prices.append(i.find('span',class_="Text-sc-1bsd7ul-0 loTFgW").text)
    company_name.append(i.find('span',class_="Text-sc-1bsd7ul-0 fOTkAa" ).text )
    product_name.append(i.find('span' ,class_="Text-sc-1bsd7ul-0 eKTmJK").text)
import pandas as pd
# create dataframe
medium = pd.DataFrame({'product name':product_name ,
                       'company name':company_name,
                       'prices':prices}) 
#display data 
medium.head()
#save data as csv file 
medium.to_csv('market.csv', encoding='utf-8')