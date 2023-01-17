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
!pip install requests
!pip install bs4
import requests

from bs4 import BeautifulSoup

# we will analyze the whitehouse url and get the header

result= requests.get("https://www.whitehouse.gov/briefings-statements/") # getting the response for this URL i.e. status code 200

print(result)
src = result.content   # To see the html content

print(src)

# Parsing the content



soup = BeautifulSoup(src,"html5")  # html5 is a type of parsing technique

print(soup)
print(soup.prettify())  # prettify function will show the html content in better way and is more readable

# Now we have to get the header and store it in some variable, Below code will do the operation

urls=[]

for h2_tag in soup.find_all("h2"):

    a_tag = h2_tag.find("a")

    urls.append(a_tag.attrs["href"])



print(urls)