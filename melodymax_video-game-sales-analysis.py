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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
sales = pd.read_csv('/kaggle/input/vgsales/vgsales.csv')
sales.head(10)
sales.describe()
sales.groupby('Genre').size()
colors = ['skyblue', 'gold', 'coral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna', 'khaki', 'gold', 'violet', 'yellowgreen']

sales.groupby('Genre').size().plot.pie(autopct="%1.1f%%", colors=colors, explode=(0.2,0,0,0,0,0,0,0,0,0,0.2,0), radius=2, startangle=350, shadow=True)
sales.groupby('Genre')['Global_Sales'].sum().plot(kind='bar', legend='Global Sales')
genre = ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role Playing',  'Shooter', 'Simulation',  'Sports', 'Strategy']



genre_NA = sales.groupby('Genre')['NA_Sales'].sum().values

genre_EU = sales.groupby('Genre')['EU_Sales'].sum().values

genre_JP = sales.groupby('Genre')['JP_Sales'].sum().values

genre_Other = sales.groupby('Genre')['Other_Sales'].sum().values

x = range(12)
JP_bottom = np.add(genre_NA, genre_EU)

Other_bottom = np.add(JP_bottom, genre_JP)



plt.figure(figsize=(10, 8))

ax = plt.subplot()

plt.bar(x, genre_NA)

plt.bar(x, genre_EU,bottom=genre_NA)

plt.bar(x, genre_JP, bottom=JP_bottom)

plt.bar(x, genre_Other,bottom=Other_bottom)





ax.set_xticks(range(len(genre)))

ax.set_xticklabels(genre, rotation=30)

plt.title('Sales per Source')

plt.xlabel('Genre')

plt.ylabel('Sales')

plt.legend(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

plt.show()