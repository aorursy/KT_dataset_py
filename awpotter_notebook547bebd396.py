# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

base = "../input/"

data = open(base+"train.json").read()

parsed_json = pd.read_json(data)

df = pd.DataFrame.from_dict(parsed_json)

for row in df.head(10)[['bathrooms','bedrooms','features','interest_level','price']].iterrows():

    print(row[1]['features'],row[1]['price'])

# Any results you write to the current directory are saved as output.