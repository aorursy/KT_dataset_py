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
!pip install jovian --upgrade --quiet #installs jovian package
import jovian
jovian.commit(project='Course Project: Philippines Covid Overview') #creates jovian.ml notebook
#import packages

import requests #for data pull
import pandas as pd #for data manipulation
import json
import time
df_caseinfo = pd.DataFrame()
page = 1
row = 1000
new_results  = False


while new_results == False:
    url = 'https://covid19-api-philippines.herokuapp.com/api/get?&page=' + str(page) + '&limit=1000'
    response = requests.get(url)
    result = response.status_code
    
    d = json.loads(response.text) #views json dictionary file
    new_result = d.get("data",[]) == [] #check if there is no further data to pull

    if new_result == False:
        
        caseinfo = pd.DataFrame.from_dict(d["data"]) #saves json file into dataframe            
        df_caseinfo = df_caseinfo.append(caseinfo)

    else: 
        print('Completed')
        break

    print('Record',row,'| Page',page,'completed. Looping...')

    page += 1
    row += 1000
    
    time.sleep(20)

from pprint import pprint
pprint(response)
df_caseinfo
url = 'https://covid19-api-philippines.herokuapp.com/api/get?&page=1'
response = requests.get(url)
result = response.status_code

d = json.loads(response.text) #views json dictionary file
new_result = d.get("data",[]) == [] #check if there is no further data to pull
d.get("data",[]) == []
new_results = response.get("data", [])
new_results
df_caseinfo.head()
df_caseinfo.shape
df_caseinfo.describe(include='all')
df_caseinfo
pd.set_option('display.max_rows', 20)
df_caseinfo.case_code
import ijson
filename = "md_traffic.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)