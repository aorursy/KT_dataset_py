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
import json
df=pd.read_json('/kaggle/input/chipotle-locations/us-states.json')
df.head()
df['type'].value_counts()
df['features'].head()
df['features'][0]
df['features'][0].keys()
len(df['features'][0]['geometry']['coordinates'][0])
frame=pd.DataFrame()
for i in range(len(df['features'])):
    ID=df['features'][i]['id']
    state=df['features'][i]['properties']['name']
    shape=len(df['features'][i]['geometry']['coordinates'])
    geo=df['features'][i]['geometry']['type']
    shape=len(df['features'][i]['geometry']['coordinates'])
    if shape>1:
        num=0
        for j in range(shape):
            num+=len(df['features'][i]['geometry']['coordinates'][j][0])
    else:
        num=len(df['features'][i]['geometry']['coordinates'][0])
    frame=frame.append({'id':ID,'state':state,'geometry':geo,'shape':shape,'coordinate pairs':num},ignore_index=True)  
frame
ger=pd.read_json('/kaggle/input/german-recipes-dataset/recipes.json')
ger.head()
ger['Ingredients'][0]
ger['num']=[len(ger['Ingredients'][i]) for i in range(len(ger))]
ger.head()
sk=ger[ger['Name'].str.contains("Sauerkraut")]
sk.head()
len(sk)
Min=sk[sk['num']==sk['num'].min()]
Min.head()
len(Min)
#compute number of letters as length of instructions
len(Min.reset_index()['Instructions'][0])
pd.options.mode.chained_assignment = None
Min['length']=[len(Min['Instructions'][index]) for index in Min.reset_index()['index']]
Min
fix=ger.drop_duplicates(subset=['Instructions'])
len(fix)
sk2=fix[fix['Name'].str.contains("Sauerkraut")]
sk2
sk2['letters']=[len(sk2['Instructions'][index]) for index in sk2.reset_index()['index']]
sk2['words']=[len(sk2['Instructions'][index].split()) for index in sk2.reset_index()['index']]
sk2