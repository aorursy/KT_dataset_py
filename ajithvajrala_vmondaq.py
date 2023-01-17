import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from bs4 import BeautifulSoup

data = pd.read_csv("/kaggle/input/vdmondaq/articles.csv", encoding= 'unicode_escape')
data.head()
data['body']
def parse_text(html):

    soup = BeautifulSoup(html)

    text = soup.get_text()

    return text
data.isna().sum()
data.fillna("sample", inplace=True)
data['clean_text'] = data['body'].apply(lambda x: parse_text(x))
data.head()