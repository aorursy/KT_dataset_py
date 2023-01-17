import numpy as np

import pandas as pd

import requests

from bs4 import BeautifulSoup as Soup



col = ['Name','Company','City','Ratings','Summary','Date']

indeed = pd.DataFrame(columns = col)





for page in range(0,10):

    

    urls = "https://www.indeed.fr/emplois?q=Data+Alternance&_ga=2.128418826.724289500.1594917590-804461608.1590582612"

    url = urls + str(page*10)

    P_url = requests.get(url)

    P_html = P_url.text

    P_soup = Soup(P_html, "html.parser")

    containers = P_soup.findAll("div", {"data-tn-component": "organicJob"}) 

    #print(len(containers))

    #print(Soup.prettify(containers[0]))

    container = containers[0]

    for container in containers:

        Name = container.findAll("a",{"class": "jobtitle turnstileLink"})

        if len(Name) != 0:

          name = Name[0].text.strip()

        else:

          name = "NaN"  



        Company = container.findAll("span",{"class":"company"})

        if len(Company) != 0:

          comp = Company[0].text.strip()

        else:

          comp = "NaN"  

        City =  container.findAll('span',{"class":"location accessible-contrast-color-location"})

        if len(City) != 0:

          city = City[0].text.strip()

        else:

          city = "NaN"  

    

        ratings = container.findAll("span",{"class":"ratingsDisplay"})

        if len(ratings) != 0:

          rat = ratings[0].text.strip()

        else:

          rat = "NaN"

        

        Summ = container.findAll("div",{"class":"summary"})

        if len(Summ) != 0:

          summ = Summ[0].text.strip()

        else:

          summ = "NaN"  

    

        date = container.findAll('span',{"class":"date"})

        if len(date) != 0:

          dat = date[0].text.strip()

        else:

          dat = "NaN"  



        data = pd.DataFrame([[name, comp, city, rat, summ, dat]])

        data.columns = col

        indeed = indeed.append(data, ignore_index = True)



print(indeed)    

indeed.to_csv('Indeed_10k.csv')

indeed.head()