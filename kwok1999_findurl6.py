# coding=UTF-8

!pip install bs4

!pip install aiohttp

!pip install nest_asyncio

!pip install pattern3

import nest_asyncio

import ssl

from tqdm import tqdm

import aiohttp

import asyncio

import time

from bs4 import BeautifulSoup

import os

import pickle

from pattern3.web import plaintext



ssl.match_hostname = lambda cert, hostname: True





def readPickle(path):

    with open(path, "rb") as f:

        return pickle.load(f)

    

def clean(s):

    """去掉换行符 \t"""

    return " ".join(s.split())



async def getContent(session, url, index):

    print(str(index), url)

    try:

        # res = requests.get(url, headers=headers,timeout = 10)

        res = await session.get(url)

        text = await res.text()

        cleand_page = str(index)+"\t"+url+"\t"+clean(plaintext(text))

    except:

        cleand_page = str(index)+"\t"+url+"\t"+"CAN'T GET CONNECT"

    print(cleand_page)

    return cleand_page



async def getUrl(session, url, index):

    url = url

    print(str(index), url)

    title = str()

    text = str()

    try:

        # res = requests.get(url, headers=headers,timeout = 10)

        res = await session.get(url)

        text = await res.text()

        try:

            soup = BeautifulSoup(text, 'lxml')

            title = str(index)+"\t"+url+"\t"+clean(soup.title.text)

#             print(title)

            return title

        except:

            title = str(index)+"\t"+url+"\t"+"CAN'T GET TITLE"

#             print(title)

            return title

    except:

        title = str(index)+"\t"+url+"\t"+"CAN'T GET CONNECT"

#         print(title)

        return title





async def main(loop, start, end, data):

    async with aiohttp.ClientSession(headers=headers) as session:

        tasks = [loop.create_task(getUrl(session, line, index))

                 for index, line in tqdm(enumerate(data))]  # 创建任务, 但是不执行

        finished, unfinished = await asyncio.wait(tasks)

        urlSet = [r.result() for r in finished]

        with open("/kaggle/working/findTitleUrl_0708_"+str(start)+"-"+str(end)+".txt", "w", encoding="utf-8") as f:

            new_list = [s for s in urlSet if s is not None]

            f.write("\n".join(new_list))

        print(time.time()-t1)

t1 = time.time()

headers = {

    'User-Agent': 'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',

}

start_from, end_from = 360000,420000

step = 1250

res = "/kaggle/input/nottitlelist/notTitleList.pickle"

data = readPickle(res)

print(len(data))

urlSet = []

nest_asyncio.apply()

for i in range(start_from, end_from, step):

    print("目前正在爬取{}，{}".format(i, i+step))

    loop = asyncio.get_event_loop()

    loop.run_until_complete(main(loop, i, i+step, data[i:i+step]))

loop.close()

print("Async total time:", time.time() - t1)