import requests
from bs4 import BeautifulSoup
import time
BASE_URL_1 = 'https://stackoverflow.com/questions/tagged/python?tab=newest&page='
SIZE = 'size=50'

resp = requests.get(f"https://stackoverflow.com/questions/tagged/python?tab=newest&page=1size=50")
soup = BeautifulSoup(resp.content, 'html.parser')
pages = soup.find_all('a', attrs={'class':'s-pagination--item js-pagination-item'})
maxline = pages[len(pages)-2]
maxpages = int(str(maxline).split('>')[1].split('<')[0])
POST_LINKS = []

for i in range(maxpages):
    resp = requests.get(f"{BASE_URL_1}{i}{SIZE}")
    soup = BeautifulSoup(resp.content, 'html.parser')
    posts = soup.find_all('a', attrs = {'class':'question-hyperlink'})
    print(x for x in posts)
#    postlinks = [x[href] for x in posts if '[closed]' in x[content]]
#    POST_LINKS += postlinks
    time.sleep(0.25)

print(len(POST_LINKS))