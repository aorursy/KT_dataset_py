# import liabraries  
# open the shop page 
#start to grap the reviews 
# clean Data
!pip install selenium
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
import numpy as np
from scipy import  stats
import matplotlib.pyplot  as plt
import seaborn as sns
# create browser obj 
# !you should download chromedriver  and change executable_path based on chromedriver.exe location
driver = webdriver.Chrome(executable_path='../chromedriver/chromedriver.exe')
driver.get('https://www.etsy.com/listing/243266250/macbook-decal-macbook-starry-night-decal?ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=&ref=sr_gallery-1-11')
sleep(5)
# save the code for the page
html = driver.page_source
soup = BeautifulSoup(html,'html')
# create two list for the review and number of star
# list_of_reviews = []
# list_of_stars   = []
#scrape over all reviews 
while(True):
    try:
        next_button =driver.find_element_by_xpath('//*[@id="reviews"]/div[2]/nav/ul/li[7]/a')
        next_button.click()
        sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html,'html')
        reviews = soup.find_all('div',class_='wt-mb-xs-3 wt-mb-md-1 wt-display-flex-md')
        for review in reviews:
            list_of_reviews.append(review.find('div',class_='wt-break-word').text.strip())
            list_of_stars.append(review.find('input')['value'])
    except:
        print('finsish')
        break
df = pd.DataFrame({'review':list_of_reviews,'rating':list_of_stars})
df.to_csv('../data/review.csv')
df['rating'].value_counts()
df2.review
df2 = pd.read_csv('../input/etsy-seller-reviews/review.csv')
# cleaned_review = []
# df2.drop('Unnamed: 0',axis=1,inplace=True)
import re
for i in df2.review:
    emoji_pattern= re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    cleaned_review.append(emoji_pattern.sub(r'', i))
      
len(cleaned_review)
from googletrans import Translator
translator = Translator()
# translated_review = []
# for i in cleaned_review:
#     result = translator.translate(i)
#     translated_review.append(result.text)

df2.review = translated_review 
df2.to_csv('../data/review.csv')
sns.countplot(df2.rating)