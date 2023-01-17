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


data_2019=pd.read_csv('../input/world-happiness/2019.csv')
data_2019.head()
data_2019.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data_2019.columns]
import matplotlib.pyplot as plt

import seaborn as sns
#Türkiye 2019 

print(data_2019[data_2019['Country']=='Turkey'])
data_2019.columns
#2019'da mutluluk puanı en yüksek ülke Finlandiya çıkmıştır.

data_2019.sort_values(by='Score',ascending=False)
#Mutluluk puanı ve diğer sütunların ilişkilerine baktığımızda GDP, Social Support, Healthy sütunları ön plana çıkmaktadır.

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data_2019.corr(),annot=True,linewidth=.5,fmt='.1f')

plt.show()
X=data_2019.drop('Score',axis=1)

X.columns
#Regresyon doğrusyla da inceledğimizde GDP, Scocial Support ve Healty alanları ile Score arasında doğrusal bir ilişki görürüz. Regressyon doğrusunun eğimi yüksektir.

g = sns.pairplot(data_2019, y_vars=["Score"], x_vars=[ "GDP", "Social_support", "Healthy",

       "Freedom", "Generosity", "Perceptions"],kind="reg")

plt.show()