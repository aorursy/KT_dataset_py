# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

display(df.head(5))
df.describe()
df.info()
df = df.rename(columns={"parking spaces": "parking", "hoa (R$)": "hoa", "rent amount (R$)" : "rent", "property tax (R$)" : "tax", "fire insurance (R$)" : "insurance", "total (R$)" : "total"})

df.info()
df.shape
from sklearn.preprocessing import LabelEncoder 

cityLE = LabelEncoder()
df['city'] = cityLE.fit_transform(df['city'])

cityLE.inverse_transform(df['city'])
list(cityLE.classes_)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(9, 6))

plt.bar(list(cityLE.classes_), df['city'].value_counts().values)

plt.suptitle('Count of cities records')

plt.show()
df[(df['city'] == 0)]["total"].values
def get_df_by_conditions(city, total=1000000000, area=50000):

    return df.loc[(df['city'] == city) & (df['total'] <= total) & (df['area'] <= area)]
belo_prices = get_df_by_conditions(0)["total"].values

belo_area = get_df_by_conditions(0)["area"].values



campinas_prices = get_df_by_conditions(1)["total"].values

campinas_area = get_df_by_conditions(1)["area"].values



porto_prices = get_df_by_conditions(2)["total"].values

porto_area = get_df_by_conditions(2)["area"].values



rio_prices = get_df_by_conditions(3)["total"].values

rio_area = get_df_by_conditions(3)["area"].values



sao_prices = get_df_by_conditions(4)["total"].values

sao_area = get_df_by_conditions(4)["area"].values



plt.figure(figsize=(18, 6))

plt.scatter(belo_area, belo_prices, color='r')

plt.scatter(campinas_area, campinas_prices, color='b')

plt.scatter(porto_area, porto_prices, color='g')

plt.scatter(rio_area, rio_prices, color='y')

plt.scatter(sao_area, sao_prices, color='c')



plt.xlabel('Area')

plt.ylabel('Total amount')

plt.suptitle('Area/Cost in every city')

plt.show()
def set_size(w,h, ax=None):

    """ w, h: width, height in inches """

    if not ax: ax=plt.gca()

    l = ax.figure.subplotpars.left

    r = ax.figure.subplotpars.right

    t = ax.figure.subplotpars.top

    b = ax.figure.subplotpars.bottom

    figw = float(w)/(r-l)

    figh = float(h)/(t-b)

    ax.figure.set_size_inches(figw, figh)
max_total = 40000

max_area = 1250



belo_prices = get_df_by_conditions(0, max_total, max_area)["total"].values

belo_area = get_df_by_conditions(0, max_total, max_area)["area"].values



campinas_prices = get_df_by_conditions(1, max_total, max_area)["total"].values

campinas_area = get_df_by_conditions(1, max_total, max_area)["area"].values



porto_prices = get_df_by_conditions(2, max_total, max_area)["total"].values

porto_area = get_df_by_conditions(2, max_total, max_area)["area"].values



rio_prices = get_df_by_conditions(3, max_total, max_area)["total"].values

rio_area = get_df_by_conditions(3, max_total, max_area)["area"].values



sao_prices = get_df_by_conditions(4, max_total, max_area)["total"].values

sao_area = get_df_by_conditions(4, max_total, max_area)["area"].values



fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)



ax[0, 0].scatter(belo_area, belo_prices, color='r')

ax[0, 1].scatter(campinas_area, campinas_prices, color='b')

ax[1, 0].scatter(porto_area, porto_prices, color='g')

ax[1, 1].scatter(rio_area, rio_prices, color='y')

ax[2, 0].scatter(sao_area, sao_prices, color='c')



set_size(50,20)

plt.xlabel('Area')

plt.ylabel('Total amount')

plt.suptitle('Area/Cost in every city')

plt.show()
df.loc[df['floor'] == '-'].shape[0]
df.loc[df['floor'] == '0'].shape[0]
df.loc[df['floor'] == '-', 'floor'] = '0'
df['floor'] = df['floor'].astype('float').astype('Int64')
df.info()
animalLE = LabelEncoder()

df['animal'] = animalLE.fit_transform(df['animal'])

animalLE.inverse_transform(df['animal'])
plt.figure(figsize=(9, 6))

plt.bar(list(animalLE.classes_), df['animal'].value_counts().values)

plt.suptitle('Count of animal records')

plt.show()
furnitureLE = LabelEncoder()

df['furniture'] = furnitureLE.fit_transform(df['furniture'])

furnitureLE.inverse_transform(df['furniture'])
plt.figure(figsize=(9, 6))

plt.bar(list(furnitureLE.classes_), df['furniture'].value_counts().values)

plt.suptitle('Count of furniture records')

plt.show()
df.info()
df.to_csv('house2rent.csv',index=False)
# !pip install jovian --upgrade --quiet
# import jovian

# project_name='01-renting-price-data-cleaning'

# jovian.commit(project=project_name, environment=None)