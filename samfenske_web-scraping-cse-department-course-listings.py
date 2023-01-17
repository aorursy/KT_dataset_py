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
import requests

from bs4 import BeautifulSoup



response = requests.get('http://bulletin.wustl.edu/undergrad/engineering/computerscience/#courses')

soup = BeautifulSoup(response.text, "html.parser")
print(soup.text)
soup
full=soup.find_all('div', attrs={'class':'courseblock'})

len(full)
#regular expressions (find patterns in text, such as numbers)

import re



#create blank dataframe in which we can append data for each course block

df=pd.DataFrame()



#go through every course

for course in soup.find_all('div', attrs={'class':'courseblock'}):

    

    #first content is the title- extract text

    title=course.contents[1].text

    

    #second content is the course description- extract text

    desc=course.contents[3].text.strip()

    

    #number of credits is a little tricky, first extract the information, then use re.sub to extract the number of credits

    #specifically

    credits_info=course.contents[4].text.strip()

    res = re.sub("\D", "", credits_info)

    

    #append the three data points to the dataframe

    df=df.append({'Title':title,'Description':desc,'Credits':res},ignore_index=True)

df.head()
#displays full contents of each cell

pd.set_option('display.max_colwidth', -1)

df[['Title','Description','Credits']].head()
df['Prerequisites']=''

for index in range(len(df)):

    if ('Prerequisite' or 'Prerequisites') in df['Description'][index]:

        df['Prerequisites'][index]=df['Description'][index].split('Prereq',1)[-1].split(':',1)[-1]

    else:

        df['Prerequisites'][index]=None
pd.set_option('display.max_rows', None)

df.head()
response = requests.get('http://bulletin.wustl.edu/undergrad/engineering/biomedical/#courses')

soup = BeautifulSoup(response.text, "html.parser")



bme=pd.DataFrame()

for entry in soup.find_all('div', attrs={'class':'courseblock'}):

    title=entry.contents[1].text

    desc=entry.contents[3].text.strip()

    credits_info=entry.contents[4].text.strip()

    res = re.sub("\D", "", credits_info)

    bme=bme.append({'Title':title,'Description':desc,'Credits':res},ignore_index=True)



bme['Prerequisites']=''

for index in range(len(bme)):

    if ('Prerequisite' or 'Prerequisites') in bme['Description'][index]:

        bme['Prerequisites'][index]=bme['Description'][index].split('Prereq',1)[-1].split(':',1)[-1]

    else:

        bme['Prerequisites'][index]=None
bme.head()
response = requests.get('http://bulletin.wustl.edu/undergrad/artsci/biology/#courses')

soup = BeautifulSoup(response.text, "html.parser")



bio=pd.DataFrame()

for entry in soup.find_all('div', attrs={'class':'courseblock'}):

    title=entry.contents[1].text

    desc=entry.contents[3].text.strip()

    credits_info=entry.contents[4].text.strip()

    res = re.sub("\D", "", credits_info)

    bio=bio.append({'Title':title,'Description':desc,'Credits':res},ignore_index=True)



bio['Prerequisites']=''

for index in range(len(bio)):

    if ('Prerequisite' or 'Prerequisites') in bio['Description'][index]:

        bio['Prerequisites'][index]=bio['Description'][index].split('Prereq',1)[-1].split(':',1)[-1]

    else:

        bio['Prerequisites'][index]=None
bio.head()