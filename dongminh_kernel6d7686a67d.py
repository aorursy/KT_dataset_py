import pandas as pd

import requests

import urllib.request

from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}

table = pd.DataFrame()

for i in range(70000,75000):

    url = "https://hosocongty.vn/1814673/page-"+str(i)

    r = requests.get(url=url,headers=headers)

    html_content = r.text

    soup = BeautifulSoup(html_content,"html.parser")

    soup = soup.find_all("ul", class_="hsdn")

    try:

      soup = soup[0].find_all("li")

      for li in soup:

          try:

              k = li.div.text.find('Mã số thuế:')

              line = pd.DataFrame({'Mã số thuế':[li.div.a.text],'Địa chỉ':[li.div.text[9:k]]})

              table = table.append(line,sort=False,ignore_index=True)

          except:

              pass

    except:

      pass

table.to_excel(r'/So71 75.xlsx',index=False)