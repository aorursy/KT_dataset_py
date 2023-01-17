# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # for plotting data



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#reading data

data = pd.read_csv("/kaggle/input/demand/train_0irEZ2H.csv",index_col = 'week')



#total/base = total_base

data['total_base'] = data.total_price / data.base_price



#fine tuning total_base

def tune(dat):

    if dat.total_base > 1:

        return 2;

    elif dat.total_base < 1:

        return 0;

    else:

        return 1;

        

new_tb= data.apply(tune, axis = 1)



#list data with sku_id -- #sku($sku_id)

sku= list(data.groupby(['store_id']))



##Units sold each week

#Usold_w = pd.DataFrame(data.groupby('week')['units_sold'].sum())

#Usold_w.head()



##Array having info of units_sold of a given sku_id

sku_id= []

sku_unit= []

graph =list(data.groupby('store_id')['units_sold'])

for i in np.arange(len(data.sku_id.unique())):

    sku_unit.append(pd.DataFrame(graph[i][1]))

    sku_id.append(graph[i][0])



sku_unit[3]

#graph[1][1]
data.sku_id.unique()

data.groupby('sku_id')['units_sold'].count()
data.head()

data.store_id == 8091
data.total_base = new_tb
sold_items_per_store = data.groupby('store_id')['units_sold'].size()
sold_items_per_store
data.total_base.head()
count_a=count_b=0

def num_tb(data):

    global count_a 

    global count_b 

    if data.total_base > 1:

        count_a= count_a + 1;

    else:

        count_b= count_b + 1;
data.apply(num_tb, axis = 1) 

print(count_a,'\n')

print(count_b)


f, ax = plt.subplots(figsize = (15,6))

sns.barplot(x= data.index,y = data['units_sold'], data= data)
data.head()
data8091 = data[data.store_id == 8091]
nw= data8091.groupby('sku_id')['units_sold']

new =pd.DataFrame(data.groupby(['week'])['units_sold'].count())

new.head()
new_dat= data.groupby(['week','total_base'])['units_sold'].count()

new_dat.head()
f, ax = plt.subplots(figsize = (20,6))

sns.countplot(x = 'sku_id', hue = 'total_base', data = data, palette = 'magma')
sku =data[(data['sku_id']==216233) & (data['total_base']==0)]



f, ax1 = plt.subplots(figsize = (20,6))

ax1.bar(sku.index,sku.units_sold)

plt.show()



sns.distplot(sku.units_sold)

plt.show()
sku =sku[(sku['sku_id']==216233) & (sku['total_base']==0)]
data.store_id.unique()