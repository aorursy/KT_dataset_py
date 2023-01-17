import requests, gc#, re

from bs4 import BeautifulSoup

import pandas as pd

def get_text(url):

    ### copied from https://www.cnblogs.com/sheng-247/p/7686014.html

    req = requests.get(url)

    if req.encoding == 'ISO-8859-1':

        encodings = requests.utils.get_encodings_from_content(req.text)

        if encodings:

            encoding = encodings[0]

        else:

            encoding = req.apparent_encoding



        # encode_content = req.content.decode(encoding, 'replace').encode('utf-8', 'replace')

        global encode_content

    encode_content = req.content.decode(encoding, 'replace') #如果设置为replace，则会用?取代非法字符；

    return BeautifulSoup(encode_content, 'html.parser')



bgm = 'https://mirror.bgm.rin.cat'



# find total number of pages

temp = get_text('https://mirror.bgm.rin.cat/group/category/all?page=1')

pages = int(temp.find('span', class_ = 'p_edge').get_text().split('\xa0')[-2])
def one_page(id_):

    this_page = get_text(f'{bgm}/group/category/all?page={id_}')

    groups = this_page.find('ul', id = 'memberGroupList')

    group_lists = groups.find_all('li', class_ = 'user') + groups.find_all('li', class_ = 'user odd')

    

    return [[group.find('a', class_ = 'avatar').get('href')[7:],  # 小组id

         group.find('a', class_ = 'avatar').get_text()[2:],       # 小组名字

         group.find('small').get_text().split(' ')[0],            # 小组人数

            ] for group in group_lists]
result = []

for i in range(1, pages + 1):

    result += one_page(i)

    

pd.DataFrame(result, columns = ['groupId', 'groupName', 'groupNum']).set_index('groupId').to_csv('group_id.csv')