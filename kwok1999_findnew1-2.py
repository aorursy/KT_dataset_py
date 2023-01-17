import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        res = os.path.join(dirname, filename)

print(res)

!pip install bs4

!pip install aiohttp

from bs4 import BeautifulSoup

import time

import asyncio

import aiohttp

from tqdm import tqdm

import ssl

ssl.match_hostname = lambda cert, hostname: True



"""

保留句子中的数字，英文字母，空格

转为小写

注意！！！ 日文和中文都会被清洗掉



符合条件会返回字符串

不符合条件会返回空字符串 ""

"""





def senToB(sentence):

    res = ''

    for ch in sentence:

        res += Q2B(ch)

    return res





"""全角转半角"""





def Q2B(uchar):

    inside_code = ord(uchar)

    if inside_code == 0x3000:

        inside_code = 0x0020

    else:

        inside_code -= 0xfee0

    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符

        return uchar

    return chr(inside_code)





"""判断一个unicode是否是英文字母"""





def is_alphabet(uchar):

    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):

        return True

    else:

        return False





"""判断一个unicode是否是数字"""





def is_number(uchar):

    if uchar >= u'\u0030' and uchar <= u'\u0039':

        return True

    else:

        return False





"""

去除一行中的\t

"""





def delTFromLine(sentence):

    temp = sentence.split("\t")

    if len(temp) == 2:

        url = temp[0]

        title = temp[1].strip()

        if title.isspace() or title == "":

            return None

    else:

        url = temp[0]

        title = [line.strip()

                 for line in temp[1:] if line != " " and line != ""]

        # 删去多余的空格

        title = " ".join(title)

    return url+"\t"+title





"""

去除多余的token 如\n 和空格

"""





def delTokenFromLine(title, token):

    if token != "\t":

        title_temp = title.split(token)

        title_temp = [line for line in title_temp if len(line) > 0]

        title_temp = " ".join(title_temp)

    else:

        title_temp = delTFromLine(title)

    return title_temp





"""

输入：字符串

输出：字符串

"""





def change(sentence):

    res = ""

    sentence = sentence.strip()

    sentence = senToB(sentence)

    for char in sentence:

        #         字符全角转半角

        #         判断你是否是英文字母和数据

        #         print(char)

        if is_alphabet(char):  # 数字和空格符合条件

            #             print("en num space "+char)

            res += char

        elif is_number(char):

            res += char

        elif char.isspace():

            res += char

        else:  # 除了英文字母、数字、空格都不符合条件

            #             print("不符合条件 "+char)

            res += " "

    # 将字符串按照空格分割，去除头尾的空格，再用空格凭借

    res = delTokenFromLine(res, "\n")

    res = delTokenFromLine(res, " ")

    return res.lower()





# coding=UTF-8



async def getUrl(session, url, index):

    url = url[:-1]

    print(str(index), url)

    title = str()

    text = str()

    try:

        # res = requests.get(url, headers=headers,timeout = 10)

        res = await session.get(url)

        text = await res.text()

        try:

            soup = BeautifulSoup(text, 'lxml')

            title = str(index)+"\t"+url+"\t"+change(soup.title.text)

            print(title)

            return title

        except:

            title = str(index)+"\t"+url+"\t"+"CAN'T GET TITLE"

            print(title)

            return title

    except:

        title = str(index)+"\t"+url+"\t"+"CAN'T CONNECT"

        print(title)

        return title





async def main(loop):

    async with aiohttp.ClientSession(headers=headers) as session:

        tasks = [loop.create_task(getUrl(session, line, index))

                 for index, line in tqdm(enumerate(data))]  # 创建任务, 但是不执行

        finished, unfinished = await asyncio.wait(tasks)

        urlSet = [r.result() for r in finished]

        with open("/kaggle/working/findTitleUrl_1203_210000-220000.txt", "w", encoding="utf-8") as f:

            new_list = [s for s in urlSet if s is not None]

            f.write("\n".join(new_list))

        print(time.time()-t1)



t1 = time.time()

headers = {

    'User-Agent': 'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',

}

with open(res, 'r', encoding="utf-8") as f:

    data = f.readlines()[210000:220000]

print(len(data))

urlSet = set()

loop = asyncio.get_event_loop()

loop.run_until_complete(main(loop))

loop.close()

# writer.close()

print("Async total time:", time.time() - t1)