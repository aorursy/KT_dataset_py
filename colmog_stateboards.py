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
!pip install selenium

!pip install beautifulsoup4

!pip install requests
from bs4 import BeautifulSoup

import requests



URL = 'http://membership.stateboards.ie/'

page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')



departments=[]

url=[]

for child in soup.find_all('ul')[2]:

    if child.name == 'li':

        departments.append(child.string)

        url.append(child.a['href'])
df_depts = pd.DataFrame({'Department':departments,'URL':url})
depts=[]

boards=[]

boards_url=[]



for index, row in df_depts.iterrows():

    dept = row['Department']

    dept_url = row['URL']

    # URL = 'http://membership.stateboards.ie/'

    dept_page = requests.get(dept_url)

    soup = BeautifulSoup(dept_page.content, 'html.parser')

    for child in soup.find_all('ul')[2]:

        if child.name == 'li':

            depts.append(dept)

            boards.append(child.string)

            boards_url.append(child.a['href'])

    df_boards = pd.DataFrame({'Department':depts,'Board':boards,'URL':boards_url})    
depts=[]

boards=[]

Name=[]

First_Appointed=[]

Reappointed=[]

Expiry_Date=[]

Position_type=[]

Basis_of_appointment=[]

#dept = 'Department of Agriculture, Food and the Marine'

#board = 'An Bord Bia'

#board_url = 'http://membership.stateboards.ie/board/An%20Bord%20Bia/'

for index, row in df_boards.iterrows():

    dept = row['Department']

    board = row['Board']

    board_url = row['URL']

    board_page = requests.get(board_url)

    soup = BeautifulSoup(board_page.content, 'html.parser')

    for child in soup.find_all('table')[0]:

        if child.name == 'tr':

            depts.append(dept)

            boards.append(board)

            Name.append(child.contents[1].string) #print(child.contents[1].string) # Name

            First_Appointed.append(child.contents[3].string) #print(child.contents[3].string) # First Appointed

            Reappointed.append(child.contents[5].string) #print(child.contents[5].string) # Reappointed

            Expiry_Date.append(child.contents[7].string) #print(child.contents[7].string) # Expiry Date

            Position_type.append(child.contents[9].string) #print(child.contents[9].string) # Position type

            Basis_of_appointment.append(child.contents[11].string) #print(child.contents[11].string)# Basis of appointment

df_members = pd.DataFrame({'Department':depts,'Board':boards,'Name':Name,'First_Appointed':First_Appointed,'Reappointed':Reappointed,'Expiry_Date':Expiry_Date,'Position_type':Position_type,'Basis_of_appointment':Basis_of_appointment})    
# df_members

df_members.to_csv(r'df_members.csv')  

from IPython.display import FileLink

FileLink(r'df_members.csv')