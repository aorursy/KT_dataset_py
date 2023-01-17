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
!pip install -U turicreate
import turicreate as tc
dir = "/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/"
villagers = pd.read_csv(dir+'villagers.csv')
pref = []
for index, row in villagers.iterrows():
    for item in row['Furniture List'].split(';'):
        pref.append({"user_id": row['Name'],"item_id": item})
df = pd.DataFrame(pref)
fields = ['user_id','Species',
 'Gender',
 'Personality',
 'Hobby',
 'Birthday',
 'Catchphrase',
 'Favorite Song',
 'Style 1',
 'Style 2',
 'Color 1',
 'Color 2']
villagers['user_id'] = villagers['Name']
user_data = tc.SFrame(villagers[fields])
item_data = tc.SFrame.read_csv(dir + 'housewares.csv', header=True, delimiter=',')
item_data = item_data.rename({'Internal ID': 'item_id'})
item_data = item_data.dropna()
data =  tc.SFrame(df)
m = tc.item_similarity_recommender.create(data,user_data=user_data,item_data=item_data)

recommendations = m.recommend()
m.recommend(users=['Anchovy'])