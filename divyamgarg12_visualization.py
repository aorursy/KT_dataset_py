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
# import some important libraries



import pandas as pd # For Data processing and importing csv as well as other format files

import numpy as np # for statistics
# import sales_train file

sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv"

                    ,index_col='date',parse_dates=['date'])
sales.head()
import matplotlib.pyplot as plt
sales["item_price"][:'2014-01-01'].plot(figsize=(16,10), legend= True, color = 'g')

sales["item_price"]['2014-01-01':'2015-01-01'].plot(figsize=(16,10), legend=True, color= 'r')

sales["item_price"]['2015-01-01':].plot(figsize=(16,10), legend = True, color = 'b')

plt.xlabel('Dates')

plt.ylabel('Item_price')

plt.title('Date vs Item_price')
import seaborn as sns

sns.distplot(sales["item_price"])

plt.figsize=(20,20)

plt.show()
sns.scatterplot(sales["item_price"], sales["item_cnt_day"])
shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

shops.head()
items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

items.head()
cat=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

cat.head()
def r(x):

    if 'PC' in x:

        return 'PC'

    elif 'PS2' in x:

        return 'PS2'

    elif 'PS3' in x:

        return 'PS3'

    elif 'PSP' in x:

        return 'PSP'

    elif 'PS4' in x:

        return 'PS4'

    elif 'PSVita' in x:

        return 'PSVita'

    elif 'XBOX 360' in x:

        return 'XBOX 360'

    elif 'XBOX ONE' in x:

        return 'XBOX ONE'

    elif 'Blu-Ray 3D' in x:

        return 'Blu-Ray 3D'

    elif 'Blu-Ray 4K' in x:

        return 'Blu-Ray 4K'

    

    else:

        return 'Others'

cat['item_category_name']=cat['item_category_name'].apply(r)
cat.head()
cat["item_category_name"].value_counts()
plt.figsize=(20,20)

sns.countplot(cat["item_category_name"])
sales.corr()
sns.heatmap(sales.corr(), annot= True, cmap = "YlGnBu")
sales.date_block_num.value_counts()
plt.plot(sales.date_block_num.value_counts().values, linestyle = '--')
sales.item_cnt_day.value_counts()