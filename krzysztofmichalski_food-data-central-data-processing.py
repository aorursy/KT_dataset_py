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
fn = pd.read_csv('/kaggle/input/food_nutrient.csv')
fn.head()
nt = pd.read_csv(r"/kaggle/input/nutrient.csv")
nt.head()
fd = pd.read_csv(r"/kaggle/input/food.csv")
fd.head()
fp = pd.read_csv(r"/kaggle/input/food_portion.csv")
fp.head()
fc = pd.read_csv(r"/kaggle/input/food_category.csv")
fc.head()
fn = fn.loc[:, 'fdc_id':'amount']
fn.head()
nt = nt.loc[:, :'unit_name']
nt.head()
fp = fp.loc[:, ['fdc_id', 'gram_weight', 'portion_description']]
fp.head()
fnn = fn.merge(nt, left_on='nutrient_id', right_on='id', how='left').drop(columns=['nutrient_id', 'id'])
fnn.head()
foods = pd.pivot_table(fnn, values='amount', index='fdc_id', columns=['name', 'unit_name'])
foods.head()
foods = foods.merge(fd, on='fdc_id', how='left')
foods.head()
foods = foods.merge(fp, on='fdc_id', how='left')
foods.head()
foods = foods.merge(fc, left_on='food_category_id', right_on='id', how='left')
foods.head()
foods.count().sort_values(ascending=False).plot()
len(foods)