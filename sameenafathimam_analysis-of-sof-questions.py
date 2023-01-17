# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_rows = 9999

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/stackindex/MLTollsStackOverflow.csv")

df.head()
df.describe()
df.columns
df.shape
df.isnull().sum().any()
df.isnull().sum()
df.Tableau.mode
df.dropna(how = 'any',inplace = True)
df.isnull().sum().any()
fig , ax = plt.subplots(figsize=(20,15))

plt.plot(df['month'], df['nltk'])

plt.show()
df.set_index('month').plot()
labels = 'Python' , 'Numpy', 'Pandas', 'Seaborn'

sizes = [df.loc[106, 'python'], df.loc[106, 'numpy'], df.loc[106, 'pandas'], df.loc[106, 'seaborn']]

colors = ['green', 'blue', 'red', 'lightcoral']



plt.pie(sizes,labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True)

#draw a circle at the center of pie to make it look like a donut

centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.title('SOF on 17-Nov for Python and Python Packages')

plt.show()  

df.python.value_counts()
plt.bar(df['month'].head(15),df['python'].head(15))

plt.show()