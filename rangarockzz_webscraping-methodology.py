# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import urllib.request
from bs4 import BeautifulSoup
import csv

f=open('03-05-2020','w',newline='')
writer=csv.writer(f)

soup=BeautifulSoup(urllib.request.urlopen("https://www.mohfw.gov.in/").read(),'lxml')



tbody=soup('table',{"class":"table table-striped"})[0].find_all("tr")
for rows in tbody:
    cols=rows.findChildren(recursive=False)
    cols=[ele.text.strip() for ele in cols]
    writer.writerow(cols)
    print(cols)
