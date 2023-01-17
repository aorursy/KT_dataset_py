from urllib.request import urlopen as uReq

from bs4 import BeautifulSoup as soup

import csv
url="https://in.bookmyshow.com/hyderabad/movies"

client=uReq(url)

page=client.read()

client.close()

bs=soup(page,"html.parser")

containers=bs.findAll("div",{"class":"cards"})

#print(len(containers))

container=containers[0]

print(container.div.img["alt"])

language=container.findAll("li",{"class":"__language"})

print(language[0].text)

censor=container.findAll("span",{"class":"censor"})

print(censor[0].text.replace("|",""))

filename="movies.csv"

f=open(filename,"w")

header="name,language,censor\n"

f.write(header)

for container in containers:

    name=container.div.img["alt"]

    name=name.replace(",","")

    lang=container.findAll("li",{"class":"__language"})

    language=lang[0].text

    cen=container.findAll("span",{"class":"censor"})

    censor=cen[0].text

    censor=censor.replace("|","")

    f.write(name+","+language+","+censor+"\n")

f.close()

    

    
from urllib.request import urlopen as uReq

from bs4 import BeautifulSoup as soup

import csv
url="https://in.bookmyshow.com/hyderabad/movies"

client=uReq(url)

page=client.read()

client.close()

bs=soup(page,"html.parser")

containers=bs.findAll("div",{"class":"cards"})

print(len(containers))

#container=containers[0]

#print(container.div.data-mobile["alt"])

#language=container.findAll("li",{"class":"__language"})

#print(language[0].text)

#censor=container.findAll("span",{"class":"censor"})

#print(censor[0].text.replace("|",""))