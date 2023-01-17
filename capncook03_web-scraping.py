# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import bs4

import urllib

import requests

from skimage import io

import matplotlib.pyplot as plt

%matplotlib inline
# creating a database of Indian Movie Celebrities with their images



url = "https://www.imdb.com/list/ls025929404/"



# getting the source code from the website using the requests module



source = requests.get(url).text



# creating a bs4 object



soup = bs4.BeautifulSoup(source,"lxml")
print(soup)
for img in soup.find_all('div',class_ = "lister-item mode-detail"):

    

    imagelink = img.find('img').get('src')

    nametemp = img.find('img').get('alt')

    

    imagefile = open(nametemp + ".jpg",'wb')

    imagefile.write(urllib.request.urlopen(imagelink).read())

    imagefile.close()
# grabbing the header 

header = soup.find('h1',class_ = "header list-name")

print(header.text)

print()

i=0



# grabbing the actors info

for actor in soup.find_all('div',class_ = "lister-item mode-detail"):

    

    name = actor.find('h3',class_ = "lister-item-header")

    print(name.a.text)

    

    image = actor.find("img").get('alt')

    i = io.imread(image+".jpg")

    plt.imshow(i)

    plt.axis("off")

    plt.show()

    

    for para in actor.find_all('p'):

        newpara = para

    print(newpara.text)

    print()