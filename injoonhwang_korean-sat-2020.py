import re
import matplotlib.pyplot as plt
import pandas as pd
f = open('../input/korean-sat/2020-korean-sat.txt', 'r')
lines = f.read().replace('\n', ' ')
f.close
lines
lines = re.sub('[-_=+,#/\?:^$.@*\"※~&%ㆍ!\\‘|\(\)\[\]\<\>`\'“”ⓐⓑⓒⓓⓔ①②③④⑤]', ' ', lines)
lines = lines.lower()
lines = lines.split()
lines
len(lines), len(set(lines))
nums = dict([(ln, lines.count(ln)) for ln in set(lines)])
nums = sorted(nums.items(), key=lambda x: x[1])
nums
len(nums)
nums[-10:]
df = pd.DataFrame(nums)
df.columns = ['word', 'number written']
df
import requests
from bs4 import BeautifulSoup
meanings = list()
i = 1

for word in df['word']:
    # 진행 상황 파악
    # 시간이 꽤 오래 걸린다
    if i % 10 == 0: print(f'{i:>4}/{len(nums)}')
    i += 1
    
    url = "http://endic.naver.com/search.nhn?query=" + word
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")
    try:
        result = soup.find('dl', {'class':'list_e2'}).find('dd').find('span', {'class':'fnt_k05'}).get_text()
    except:
        result = None
    meanings.append(result)

df['meaning'] = meanings
df
df.to_excel('./result.xlsx')