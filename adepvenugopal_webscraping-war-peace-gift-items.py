from urllib.request import urlopen

from bs4 import BeautifulSoup

import re
html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')

#html.read()
bs = BeautifulSoup(html, "html.parser")

print(bs)
titles = bs.find_all(['h1', 'h2','h3','h4','h5','h6'])

print([title for title in titles])
nameList = bs.findAll('span', {'class': 'green'})

for name in nameList:

    print(name.get_text())
nameList = bs.find_all('span', {'class':{'green', 'red'}})

for name in nameList:

    print(name.get_text())
nameList = bs.find_all(text='the prince')

print(len(nameList))
nameList = bs.find_all(text='Prince Vasili')

print(len(nameList))
allText = bs.find_all(id='title', class_='text')

print([text for text in allText])
html = urlopen('http://www.pythonscraping.com/pages/page3.html')

#html.read()
bs = BeautifulSoup(html, 'html.parser')

bs
for child in bs.find('table',{'id':'giftList'}).children:

    print(child)
for sibling in bs.find('table', {'id':'giftList'}).tr.next_siblings:

    print(sibling) 
print(bs.find('img', {'src':'../img/gifts/img1.jpg'}).parent.previous_sibling.get_text())

print(bs.find('img', {'src':'../img/gifts/img2.jpg'}).parent.previous_sibling.get_text())

print(bs.find('img', {'src':'../img/gifts/img3.jpg'}).parent.previous_sibling.get_text())

print(bs.find('img', {'src':'../img/gifts/img4.jpg'}).parent.previous_sibling.get_text())

print(bs.find('img', {'src':'../img/gifts/img5.jpg'}).parent.previous_sibling.get_text())
images = bs.find_all('img', {'src':re.compile('\.\.\/img\/gifts/img.*\.jpg')})

for image in images: 

    print(image['src'])
bs.find_all(lambda tag: len(tag.attrs) == 2)
bs.find_all(lambda tag: tag.get_text() == 'Or maybe he\'s only resting?')
bs.find_all('', text='Or maybe he\'s only resting?')