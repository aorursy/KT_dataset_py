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
import requests

from bs4 import BeautifulSoup
response = requests.get("http://www1.cuny.edu/mu/forum/?sf_paged=1")

#print(response.content)



soup = BeautifulSoup(response.content)

type(soup)
#for row in rows:

    #spans = soup.find_all('span', attrs=('date'))

    #for span in spans:

        #print (span.text)
#My loop function that finds the info 



def infogetter(x):

    

    response= requests.get(x)

    soup = BeautifulSoup(response.content)



    highlight = soup.find('ul', class_="post-list-container wpb_row vc_row-fluid post-list-container filter")

    #print(highlight) 

    

    rows = highlight.find_all("li")

    #print(rows)





    

    for row in rows:

                                                             #title and link

        cells = row.find_all('h2')

        if cells:

            for cell in cells:

            

                title = cell.a

                if title:

                

                    print(title.text)

                    

                    print(title.get('href'))  

                                                             #date

        dates = row.find_all('span', attrs=('date'))

        for span in dates:

            print (span.text)

            

#loop for url



Pages=[]



for x in range (1,6):

    Pages.append("http://www1.cuny.edu/mu/forum/?sf_paged=%d" % (x))

    

print(Pages)
for x in Pages:

    print ("Page ", x[-1])

    infogetter(x)