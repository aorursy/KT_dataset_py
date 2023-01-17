from bs4 import BeautifulSoup
import requests
url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&p=1&u=/netahtml/PTO/srchnum.html&r=1&f=G&l=50&d=PALL&s1=10618288.PN.'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc) #Create a BeautifulSoup object from the HTML
pretty_soup = soup.prettify() #Prettify the BeautifulSoup object
#print(pretty_soup)
soup.head.title.text
tables = soup.body.find_all('table')
b_tables = tables[2].find_all('b')
patent = b_tables[1].text
pub_date = b_tables[4].text
print("Patent Number: " + str(patent))
print("Publication Date: " + str(pub_date))
title = soup.body.find_all('font')
title = title[3].text
print("Title: " + str(title))
abstract = soup.p.text
print("Abstract: \n" + str(abstract))
tr = tables[3].find_all('tr')
inventors = tr[0].find_all('b')
print("Inventors: ")
for inventor in inventors:
    print(inventor.text)
assignee = tr[1].b.text
print("Assignee: " + str(assignee))
file = tr[7].b.text
print("Filing Date: " + str(file))
coma = soup.coma.text
p_coma = soup.coma.prettify()
claim = p_coma.split("<br/>")
claims = claim[7:34]
for i in claims:
    print(i)
desc = claim[35:96]
for i in desc:
    print(i)
a_tags = soup.find_all('a')

for link in a_tags:
    print("http://patft.uspto.gov/" + str(link.get('href')))