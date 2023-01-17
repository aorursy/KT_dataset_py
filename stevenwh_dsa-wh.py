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

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_excel("/kaggle/input/used_car_data.xlsx")
df
merek = []

for name in df['Name']:

    merek.append(name.split(" ")[0].upper())

    

df['Merek'] = merek
sns.distplot(df['Year'])
df['Kilometers_Driven'].median()
category = []

for km in df['Kilometers_Driven']:

    if km >= df['Kilometers_Driven'].median():

        category.append("Tinggi")

    else:

        category.append("Rendah")

        

df['Category'] = category
df
sns.boxplot(x=df['Kilometers_Driven'])
df.sort_values(by='Kilometers_Driven', ascending=False)
index = df.loc[df['Kilometers_Driven']== 6500000]

df.drop(2328, inplace=True)

sns.distplot(df['Kilometers_Driven'])
sns.boxplot(x=df['Kilometers_Driven'])
# Fungsi untuk menyamakan satuan menjadi km/kg

# Menggunakan asumsi 1 liter = 0,8 kg

def samakan_satuan(mileage):

    if pd.notna(mileage):

        satuan= mileage.split(" ")[1]

        jarak = float(mileage.split(" ")[0])

        if satuan == "kmpl":

            jarak = jarak / 0.8

            return jarak

        return jarak
# Membuat suatu kolom yang memuat jarak tempuh bahan bakar yang telah disamakan satuannya menjadi km/kg 

df['Mileage'] = df.apply(lambda row: samakan_satuan(row['Mileage']), axis=1)
# Membuat DataFrame yang menyortir rata-rata jarak yang ditempuh dari setiap bahan bakar yang ada 

df.groupby('Fuel_Type').mean().sort_values(by='Mileage', ascending=False)
average_price = df.groupby(['Merek']).Price.agg(['mean'])

average_price