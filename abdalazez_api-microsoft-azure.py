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
# !pip install urllib2
# import urllib2
import urllib.request as urllib2
import json

data = {
        "Inputs": {
                "input1":
                [
                    {
                            'sepal_length': "7",   
                            'sepal_width': "7.1",   
                            'petal_length': "7.4",   
                            'petal_width': "7",   
                            'species': "",   
                    }
                ],
        },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/b8debfb034224bd89c9cd9c55e927320/services/2831423029624b099927d5c1f9d3c9dc/execute?api-version=2.0&format=swagger'
api_key = 'wzCxi4sk5lEIaLqGIZ0gLi8pJJtvN8Y/CWe3DNmExSejyEkhvgX+lpi1wGQgcBG8kHxWo9BPOIs/BF3fesmE0Q==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers)

try:
    response = urllib2.urlopen(req)

    result = response.read()
    print(result)
except urllib2.HTTPError:
    print("The request failed with status code: " )
