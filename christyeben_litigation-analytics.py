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
case_id = '11273792'
response = requests.get(

    'https://api.case.law/v1/cases/{}/?full_case=true'.format(case_id),

#     headers={'Authorization': 'Token abcd12345'}

)



case_data = response.json()

response.close()
for k,v in case_data.items():

    

    print(k, '\t', type(v))

    if isinstance(v, dict):

        

        for k1, v1 in v.items():

            print('\t', k1, '\t', type(v1))

            

    print()
case_data
case_data.keys()