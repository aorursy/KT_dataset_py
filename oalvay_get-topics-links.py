import requests, gc, re, time

from bs4 import BeautifulSoup

import pandas as pd 



group_id = pd.read_csv('/kaggle/input/bangumi-group-project/group_id.csv').drop_duplicates()



def get_text(url, count = 1):

    try:

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

    except:

        if count >= 10:

            raise RuntimeError('request failed too many times')

        time.sleep(count)

        return get_text(url, count + 2)



bgm = 'https://mirror.bgm.rin.cat'

groupIdList = group_id['groupId'].to_list()



def one_page(url):

    # 爬小组某一页的数据

    # 输出格式：list(list(帖子id, 作者id, 帖子标题))

    thisPage = get_text(url)

    topics = thisPage.find_all('tr', class_ = 'topic odd') + thisPage.find_all('tr', class_ = 'topic even')

    return [[i.get('href').split('/')[-1] for i in topic.find_all('a', class_ = 'l')] + 

            [topic.find('a', class_ = 'l').get_text()] for topic in topics]



def one_group(id):

    # 爬某个小组的数据

    # 输出格式：list(list(小组id, 帖子id, 作者id, 帖子标题))

    url = f'{bgm}/group/{id}/forum?page='

    count = 1

    result = []

    newPage = one_page(f'{url}{count}')

    

    while newPage != []:

        result += newPage # this is actually previous page

        count += 1

        newPage = one_page(f'{url}{count}')

        

    return [[id] + i for i in result]
ouput = []

for groupId in groupIdList:

    ouput += one_group(groupId)

    

pd.DataFrame(ouput, columns = ['groupId', 'topicId', 'authorId', 'title']).set_index('topicId').to_csv('topic_id.csv')