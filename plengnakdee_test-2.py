import requests

from bs4 import BeautifulSoup

import csv
source = requests.get('https://pantip.com/topic/38505531').text

soup = BeautifulSoup(source)

title = soup.find('h2',{'class':'display-post-title'}).text

article = soup.find('div',{'class':'display-post-story'}).text

like = soup.find('div', {'class':'display-post-vote'}).text

print(title)

print(article)

print(like)
csv_file = open('pantip.csv', 'a', encoding='utf-8')

csv_writer = csv.writer(csv_file)

#csv_writer.writerow(['title', 'article', 'like'])

csv_writer.writerow([title, article, like])

csv_file.close()