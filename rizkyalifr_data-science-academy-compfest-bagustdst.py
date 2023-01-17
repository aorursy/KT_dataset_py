# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/dsacomp/used_car_data.csv")
df.sample(50)
model=df['Name']
merek=[]
for x in model :
    a = x.split()[0]
    if a == "ISUZU" :
        a ="Isuzu"
    merek.append(a)
merek=pd.Series(merek)
merek.value_counts()
df["Location"].value_counts()
df["Year"].describe()
df["Year"].plot(kind='hist',title="Distribusi Mobil Berdasarkan Tahun")
condition = df["Kilometers_Driven"] < 100000
c=condition.value_counts().max()
print('Mobil yang pemakaiannya dibawah 100.000 Kilometer sebanyak',c)
df["Kilometers_Driven"].describe()
Q1=df["Kilometers_Driven"].quantile(0.25)
Q3=df["Kilometers_Driven"].quantile(0.75)
print('Batas kilometer rendah adalah',Q1)
print('Batas kilometer tinggi adalah',Q3)
IQR=Q3-Q1
minimum = Q1 - 1.5*IQR
maximum = Q3 + 1.5*IQR
condition = (df["Kilometers_Driven"] >= minimum) & (df["Kilometers_Driven"] <= maximum)
df=df[condition]
condition.value_counts()
df["Kilometers_Driven"].describe()
# find the mean and std dev
#mean = df['Kilometers_Driven'].mean()
#std_dev = df['Kilometers_Driven'].std()

# find z scores
#z_scores = (df['Kilometers_Driven'] - mean) / std_dev
#z_scores = np.abs(z_scores)

#df=df[z_scores<3] ##Remove Outlier

#condition = z_scores >= 3
#condition.value_counts()


df["Kilometers_Driven"].plot(kind='box',title="Distribusi Mobil Berdasarkan Tahun")
x=df["Kilometers_Driven"]
y=df["Price"]
plt.scatter(x,y)
y=df['Kilometers_Driven']
x=df['Year']
plt.scatter(x,y)
x.corr(y)
condition = (df['Owner_Type'] == "Third") | (df['Owner_Type'] == "Fourth & Above")
df.loc[condition,'Owner_Type'] = 'Third & Above'
df['Owner_Type'].value_counts(ascending=True)
angka=df['Mileage']
satuan=[]
for x in angka :
    x =str(x)
    a = x.split()[0]
    satuan.append(a)

new_list = []
for item in satuan:
    new_list.append(float(item))
satuan=new_list

df['Mileage']=satuan
df.groupby(['Fuel_Type'])['Mileage'].mean().dropna()