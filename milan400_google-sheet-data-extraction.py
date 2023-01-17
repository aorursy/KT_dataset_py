!pip install gspread oauth2client

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("/kaggle/input/google form-c0a03acfc626.json", scope)
client = gspread.authorize(creds)
sheet = client.open('googleform').sheet1
data = sheet.get_all_records()
data
import pandas as ps
import numpy as np
df_location = pd.DataFrame()
name = []
from_target = []
to = []

for values in data:
    name.append(values['name'])
    from_target.append(values['from'])
    to.append(values['to'])
df_location['Name'] = name
df_location['Start location'] = from_target
df_location['Destination location'] = to
df_location
data
sheet.row_values(3)
sheet.col_values(3)
sheet.cell(2,1).value
insert_row = ["karan", "delhi", "ktm"]
sheet.insert_row(insert_row, 4)
data = sheet.get_all_records()
data
sheet.delete_row(4)
data = sheet.get_all_records()
data
sheet.update_cell(3,2,"gorkha")
data = sheet.get_all_records()
data