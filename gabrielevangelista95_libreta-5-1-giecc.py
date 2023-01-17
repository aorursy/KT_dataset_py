import requests
from bs4 import BeautifulSoup
url = 'https://www.misprofesores.com/profesores/Huertas-Jose-Ignacio_83196'
page = requests.get(url)
page
soup = BeautifulSoup(page.text, "html.parser")
print(soup.prettify())
soup.title
soup.title.name
soup.title.string
proftags = soup.findAll("span", {"class": "rating-type" })
proftags
for mytag in proftags:
    print(mytag.get_text())
url2 ='https://www.ratemyprofessors.com/ShowRatings.jsp?tid=941931'
page2 = requests.get(url2)
page2

soup = BeautifulSoup(page2.text, "html.parser")
print(soup.prettify())
profratings = soup.findAll("div", {"class": "RatingValues__RatingValue-sc-6dc747-3 iZENup" })
profratings
for myrating in profratings:
    print(myrating.get_text())
proftags = soup.findAll("span", {"class": "Tag-bs9vf4-0" })
proftags
for mytag in proftags:
    print(mytag.get_text())