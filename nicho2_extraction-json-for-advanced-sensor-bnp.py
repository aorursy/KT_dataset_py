# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import json 

import pandas as pd 

from pandas.io.json import json_normalize #package for flattening json in pandas df



#load json object

with open('../input/export-data2/export.json') as f:

    d = json.load(f)



#works_data = json_normalize(data=d['programs'], record_path='works', 

#                            meta=['id', 'orchestra','programID', 'season'])

#works_data.head(3)

print(len(d))

works_data = json_normalize(data=d)



works_data.head(5)
works_data.describe()

works_data.corr()
#load json object

print(os.listdir("../input/export-data"))

with open('../input/export-data/monexport_Datas.json') as f:

    datas = json.load(f)

print(len(datas))
works_data = json_normalize(data=datas)



works_data.head(5)

listColumns = works_data.columns

listColumns = [el if 'Datas.' not in el else el.replace('Datas.','') for el in listColumns ]

works_data.columns = listColumns

works_data.head(5)