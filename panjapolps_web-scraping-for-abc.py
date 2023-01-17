!pip install requests beautifulsoup4 fake-useragent --quiet



import os

import time

from bs4 import BeautifulSoup

import requests

from urllib.request import Request, urlopen

from fake_useragent import UserAgent

import random



print(os.listdir('./'))
ua = UserAgent() 



def get_proxies():

    req = Request('https://www.sslproxies.org/')

    req.add_header('User-Agent', ua.random)

    doc = urlopen(req).read().decode('utf8')

    soup = BeautifulSoup(doc, 'html.parser')

    table = soup.find(id='proxylisttable')

    proxies = []

    for row in table.tbody.find_all('tr'):

        proxies.append({

            'protocol': 'https' if row.find_all('td')[6].string == 'yes' else 'http',

            'ip': row.find_all('td')[0].string, 

            'port': row.find_all('td')[1].string 

        })

    return proxies
base_url = 'http://abcnotation.com'



# Edit these values

start_page = 200

last_page = 300
for i in range(start_page, last_page):

    cur_page = str(i)

    while len(cur_page) != 4:

        cur_page = '0' + cur_page

    page_link = base_url + '/browseTunes?n=' + cur_page

    page_response = requests.get(page_link)

    page_soup = BeautifulSoup(page_response.text, 'html.parser')

    pre_tag = page_soup.find('pre')

    a_tag = pre_tag.find_all('a', href=True)

    

    print('Page:', i)

    file_number = 0

    proxies = get_proxies()

    prev_proxy_index = 0

    

    for link in a_tag:

        file_link = base_url + link['href']

        

#         Fake Proxy & UA => Can't break web security.

        cur_proxy_index = random.randint(0, len(proxies) - 1)

        while cur_proxy_index == prev_proxy_index:

            cur_proxy_index = random.randint(0, len(proxies) - 1)

        proxy = {proxies[cur_proxy_index]['protocol']: 'http://' + proxies[cur_proxy_index]['ip'] + ':' + proxies[cur_proxy_index]['port']}

        prev_proxy_index = cur_proxy_index

        headers = requests.utils.default_headers()

        headers.update({'User-Agent': ua.random})

        file_response = requests.get(file_link, proxies=proxy, headers=headers)



        if file_response.status_code == 200:

            file_soup = BeautifulSoup(file_response.text, 'html.parser')

    #         print(file_soup.prettify())

            text_area = file_soup.find('textarea')

    #         print(text_area.contents[0])

            with open('./scraping_' + str(i) + '_' + str(file_number) + '.abc', 'a') as file:

                file.write(text_area.contents[0])

#                 print('scraping_' + str(i) + '_' + str(file_number) + '.abc created') 

        else:

            print('Failed at page ' + str(i) + ' file ' + str(file_number))

        file_number+=1

        time.sleep(1)
!tar -zcf scraping.tar.gz ./

!rm -rf ./scraping_*

# !tar -zxf scraping.tar.gz # For extraction

!ls ./