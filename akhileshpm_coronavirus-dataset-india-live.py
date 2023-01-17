# libraries
!pip install gspread oauth2client
from datetime import datetime
import os
import re
import glob
import requests 
import pandas as pd
from bs4 import BeautifulSoup
import gspread
from oauth2client.service_account import ServiceAccountCredentials
link = 'https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vSz8Qs1gE_IYpzlkFkCXGcL_BqR8hZieWVi-rphN1gfrO3H4lDtVZs4kd0C3P8Y9lhsT1rhoB-Q_cP4/pubhtml?urp=gmail_link#'
req = requests.get(link)
soup = BeautifulSoup(req.content, "html.parser")
tbody = soup.find_all('tbody')[-8] #Here we need to insert Index Number from last then we get the dataset of specific sheets.
body = tbody.find_all('tr')

# print(rows)
body_rows = []
    
for tr in body:
    td = tr.find_all(['th', 'td'])
    row = [i.text for i in td]
    body_rows.append(row)
    
df_bs = pd.DataFrame(body_rows, index=None)
df_bs.head()