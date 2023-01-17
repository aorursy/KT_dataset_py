# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/aiosogbo-certification-competition/Train.csv')
perishable = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables",

              "Meat", "Seafood", 'Snack Foods']



non_perishable = ["Baking Goods", "Canned","Frozen Foods", "Hard Drinks",

                  "Health and Hygeine", "Household", "Soft Drinks"]



# create a new feature 'Item_Type_new'

Item_Type_new = []

for item in df['Item_Type']:

    if item in perishable:

        Item_Type_new.append('perishable')

    elif item in non_perishable:

        Item_Type_new.append('non_perishable')

    else:

        Item_Type_new.append('not_sure')
# create a new feature 'Item_Type_new'

df['Item_Type_new'] = Item_Type_new
# create a new feature 'Outlet_Years' from 'Outlet_Establishment_Year'



df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

df['Outlet_Years'].head()
# create a new feature 'Price_Per_Unit_wt'

df['Price_Per_Unit_wt'] = df['Item_MRP'] / df['Item_Weight']

df['Price_Per_Unit_wt'].head()


# create a new feature 'Item_MRP_Clusters'

# 'Item_MRP_Clusters' this feature will group expensive products and the less expensive ones. This will inform the model of product that fall in the range of higher prices and product within the lower price 

# df['Item_MRP_Clusters']

Item_MRP_Clusters = []

for item in df['Item_MRP']:

    if item < 69:

        Item_MRP_Clusters.append(1)

    elif item >= 69 and item < 136:

        Item_MRP_Clusters.append(2)

    elif item >=136 and item < 203:

        Item_MRP_Clusters.append(3)

    else:

        Item_MRP_Clusters.append(4)

df['Item_MRP_Clusters'] = Item_MRP_Clusters

################################ Using Label Code Encoder ###############################

Outlet_Size_num = []

for item in df['Outlet_Size']:

    if item == 'Small':

        Outlet_Size_num.append(0)

    elif item == 'Medium':

        Outlet_Size_num.append(1)

    else:

        Outlet_Size_num.append(2)

df['Outlet_Size_num'] = Outlet_Size_num

df['Outlet_Size_num'].head()
Outlet_Location_Size_num = []

for item in df['Outlet_Location_Type']:

    if item == 'Tier 3':

        Outlet_Location_Size_num.append(3)

    elif item == 'Tier 2':

        Outlet_Location_Size_num.append(2)

    else:

        Outlet_Location_Size_num.append(0)

df['Outlet_Location_Size_num'] = Outlet_Location_Size_num

df['Outlet_Location_Size_num'].head()
outlet_Type = pd.DataFrame(df['Outlet_Type'])

fe = outlet_Type.groupby('Outlet_Type').size() / len(outlet_Type)

outlet_Type.loc[:, 'Outlet_Type_fe'] = outlet_Type['Outlet_Type'].map(fe)



df['Outlet_Type'] = outlet_Type['Outlet_Type_fe']
# removing categorical variable after label encoding

df = df.drop(['Outlet_Location_Type', 'Outlet_Size'], axis=1)

df.columns
################################# One Nice Hot Encoding -- Baba Dummy ###############################



dummy_cols = ['Item_Type']

combi_cat = pd.get_dummies(df[dummy_cols])

combi_cat.head()
# Let combine the dummy dataframe and the other dataframe while dropping the affected column 'Item_Type'

df_new = df.drop(dummy_cols, axis=1)

df = df_new.join(combi_cat)

df.head()
import matplotlib.pyplot as plt

Item_Visibility = df['Item_Visibility']

fig, ax = plt.subplots()

ax.hist(Item_Visibility.dropna(), color='green', bins=80, alpha=0.9)

plt.xlabel('Item_Visibility')

plt.ylabel('count')

plt.title('Histogram of Item_Visibility')

# Removing Skewness by taking logs

df['Item_Visibility'] = np.log(df['Item_Visibility'] + 1) # log + 1 to avoid division by zero

df['Price_Per_Unit_wt'] = np.log(df['Price_Per_Unit_wt'] + 1)