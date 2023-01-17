# This notebook contains the code that I used to scrape the website. Feel free to use it! 

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

URL = "https://www.theramenrater.com/resources-2/the-list/"
page = requests.get(URL)
soup = BeautifulSoup(page.text, 'html.parser')
table_contents = soup.find_all("tr")[1:]
table_dict = {"ID": [], "URL": [],  "Brand": [], "Variety": [], "Style": [], 
              "Country": [], "Stars": []}
for table_content in table_contents:
    attrs = table_content.find_all("td")
    table_dict["URL"].append(attrs[0].find_all("a")[0]["href"])
    table_dict["ID"].append(attrs[0].find_all("a")[0].text)
    table_dict["Brand"].append(attrs[1].text)
    table_dict["Variety"].append(attrs[2].text)
    table_dict["Style"].append(attrs[3].text)
    table_dict["Country"].append(attrs[4].text)
    table_dict["Stars"].append(attrs[5].text)
df = pd.DataFrame(table_dict)
df.head()
df.to_csv("ramen_ratings_2020.csv", index=False)
