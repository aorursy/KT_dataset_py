import requests
response = requests.get('http://www1.cuny.edu/mu/forum/2020/03/06/cuny-colleges-mark-womens-history-month-with-over-60-events-across-campuses/')
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.content)
rows = soup.find_all('div')

print(rows)
for row in rows:

    print(row.find_all('h1',"title"))
for row in rows:

    cells = row.find('h1')

    if cells:

        for cell in cells:

            Title = cell

if Title:

    print(title.text)