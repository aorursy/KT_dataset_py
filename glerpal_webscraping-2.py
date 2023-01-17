import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
import matplotlib.pyplot as plt
url='http://webscraper.io/test-sites/e-commerce/allinone'
content = requests.get(url).text
soup = BeautifulSoup(content, 'html.parser')
row = soup.find(class_="row")
shopping = row.find_all(class_="col-sm-4 col-lg-4 col-md-4")
i1 = shopping[1]
print(i1.prettify())
title = row.find(class_='title').get_text()
price = row.find(class_='pull-right price').get_text()
desc = row.find(class_='description').get_text()

print(title)
print(price)
print(desc)
title = [t.get_text() for t in row.select('.caption .title')]
price = [p.get_text() for p in row.select('.caption .pull-right.price')]
desc = [d.get_text() for d in row.select(".caption .description")]

print(title)
print(price)
print(desc)
computers = pd.DataFrame({
    'Model': title,
    'price': price,
    'desc': desc
    })

computers