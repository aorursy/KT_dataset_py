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
!pwd
%cd 
!wget -nc https://paperswithcode.com/media/about/papers-with-abstracts.json.gz
!gunzip -f papers-with-abstracts.json.gz
!ls -lhS
!head -n 30 papers-with-abstracts.json
!pip install -U pip
!pip install pandas
!pip install tqdm
!pip install setuptools
!pip install tfkit
import pandas as pd 

import json
import os.path

file_name = "papers-with-abstracts.json"
cols = ['title', 'tasks']
df = pd.DataFrame(columns=cols)

with open(file_name, encoding='utf-8') as f:
    docs = json.load(f)
    for doc in docs:
      # print(doc)
      if doc['title'] != '' and len(doc['tasks']) > 0:
        lst_dict=({'content': doc['title'], 'tasks': "/".join(doc['tasks'])})
        df = df.append(lst_dict, ignore_index=True)

df.to_csv('papers-with-abstracts.csv', index=False)
!head -n 50 papers-with-abstracts.csv
df.info()

