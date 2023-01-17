

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')

df
from bs4 import BeautifulSoup



def fix_html(html):

    soup = BeautifulSoup(html)



    for a in soup.findAll('code'):

        a.replaceWith("CODE")



    return (soup.get_text()).replace("\\n",'\n')

    

fix_html('''<p>I have <code>Dictionary&lt;int key, int sum</code></p>\n\n<p>asd</p>''')
df['Body'] = df['Body'].apply(lambda x: fix_html(x))

df.head()