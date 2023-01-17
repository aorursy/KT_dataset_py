# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import requests
from bs4 import BeautifulSoup
Images=[]
df=pd.read_csv("Desktop/urls.csv")
for index,row in df.iterrows():
    webpage= row['urls']
    response = requests.get(webpage)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    Images.append(html_soup.find_all('img', class_="product-description"))
df['Images']=Images
df.head()
#df.to_csv("Desktop/urls.csv",header="infer")






