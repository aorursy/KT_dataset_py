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
df = pd.read_excel("/kaggle/input/used_car_data.xlsx")
df
merek = []

for name in df['Name']:

    merek.append(name.split(" ")[0].upper())

    

df['Merek'] = merek
df
# Fungsi untuk menyamakan satuan menjadi km/kg

# Menggunakan asumsi 1 liter minyak = 0,8 kg

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
# Fungsi untuk membuang satuan CC

def buang_satuan_CC(engine):

    if pd.notna(engine) and engine != 'null CC' :

        return float(engine.split(" ")[0])

    else:

        return np.nan
# Membuang satuan CC pada kolom Engine

df['Engine'] = df.apply(lambda row: buang_satuan_CC(row['Engine']), axis=1)
# Fungsi untuk membuang satuan bhp

def buang_satuan_bhp(power):

    if pd.notna(power) and power != 'null bhp' :

        return float(power.split(" ")[0])

    else:

        return np.nan
# Membuang satuan bhp pada kolom Power

df['Power'] = df.apply(lambda row: buang_satuan_bhp(row['Power']), axis=1)
df
df['Price'].groupby(df['Merek']).count()
df['Location'].groupby(df['Location']).count().sort_values(ascending=False)
import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(df['Year'])
df[df['Kilometers_Driven'] < 100000].count()
category = []

for km in df['Kilometers_Driven']:

    if km >= df['Kilometers_Driven'].median():

        category.append("Tinggi")

    else:

        category.append("Rendah")

        

df['Category'] = category
df
sns.set(rc={'figure.figsize':(16,8)})

sns.boxplot(x=df['Kilometers_Driven'])
fig, ax = plt.subplots(figsize=(16,8))

ax.scatter(df['Year'], df['Kilometers_Driven'])

ax.set_xlabel('Year')

ax.set_ylabel('kilometres Driven')

plt.show()
data = sorted(df['Kilometers_Driven'])

quantile1, quantile3 = np.percentile(data, [25, 75])



iqr = quantile3 - quantile1

lower_bound = quantile1 - (1.5 * iqr)

upper_bound = quantile3 + (1.5 * iqr)



outliers = sorted(df[(df['Kilometers_Driven'] > upper_bound) | (df['Kilometers_Driven'] < lower_bound)].Kilometers_Driven)



for i in outliers:

    print(i,end=", ")



print()

print()

print("Jumlah outlier :",len(outliers))
df['Year'].corr(df['Kilometers_Driven'])
df[(df['Owner_Type'] == 'Third') | (df['Owner_Type'] == 'Fourth & Above')].count()
# Membuat DataFrame yang menyortir rata-rata jarak yang ditempuh dari setiap bahan bakar yang ada 

df.groupby('Fuel_Type')['Mileage'].mean().sort_values(ascending=False)
df.isnull().sum()
df.drop(columns = ['Category', 'Merek'])
df = df.dropna(subset = ['Mileage', 'Engine', 'Power', 'Seats'])
df.isnull().sum()
df['Price'].corr(df['Year'])
df['Price'].corr(df['Kilometers_Driven'])
df['Price'].corr(df['Mileage'])
df['Price'].corr(df['Engine'])
df['Price'].corr(df['Power'])
df['Price'].corr(df['Seats'])
import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.multivariate.manova import MANOVA
maov = MANOVA.from_formula('Name+Location+Fuel_Type+Transmission+Owner_Type~Price', data = df)

print(maov.mv_test())
reg = ols('Price ~ Name', data = df).fit()

aov = sm.stats.anova_lm(reg, type = 2)

print(aov)
reg = ols('Price ~ Location', data = df).fit()

aov = sm.stats.anova_lm(reg, type = 2)

print(aov)
reg = ols('Price ~ Fuel_Type', data = df).fit()

aov = sm.stats.anova_lm(reg, type = 2)

print(aov)
reg = ols('Price ~ Transmission', data = df).fit()

aov = sm.stats.anova_lm(reg, type = 2)

print(aov)
reg = ols('Price ~ Owner_Type', data = df).fit()

aov = sm.stats.anova_lm(reg, type = 2)

print(aov)