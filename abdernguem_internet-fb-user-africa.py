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
pathafric = "../input/africa-population-and-internet-users-statistics/openafrica-fb290751-bfd4-487a-92c2-7e1f56150602/data/sheetjs.csv"
africinfo = pd.read_csv(pathafric)
#let see the first rows of the dataset
africinfo.head()
africinfo.columns
#let see the few last rows of the dataset
africinfo.tail()

#remove row by index 58 to 60
afric = africinfo.drop([58,59,60])
afric.tail()
#confirm the types of object in the dataset
afric.info()

xcountry = africinfo.africa[:-3]
ypop = africinfo.population_2018_est[:-3]
ypopinternet = africinfo.internetusers_31_dec_2017[:-3]

x = np.arange(len(xcountry))  # label location
width = 0.35  # width of the bars

#plt.figure(figsize=(13, 7))
fig, ax = plt.subplots()

ax.bar(x - width/2, ypop, width, label='Population')
ax.bar(x + width/2, ypopinternet, width, label='Internet User')

ax.set_title('Population versus Internet user')
plt.xticks(rotation=60)
ax.set_xlabel('Country', fontsize = 'medium')
ax.set_ylabel('Population')

ax.set_xticks(x)
ax.set_xticklabels(xcountry)
ax.legend()