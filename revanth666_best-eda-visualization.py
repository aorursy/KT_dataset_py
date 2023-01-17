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

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

from matplotlib import rcParams

from collections import Counter

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('/kaggle/input/toy-dataset/toy_dataset.csv')

data.head()
data.shape
data.isnull().sum()
data.info()
data.describe()
data['City'].value_counts()
data['Gender'].value_counts()
data['Illness'].value_counts()
df = pd.DataFrame(data)

df.head(2)
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['City'].value_counts().values, y = df['City'].value_counts().index)

plt.title('City wise')

plt.xlabel('Counts')

plt.ylabel('Cities')

plt.show()
rcParams['figure.figsize'] = 10,7

values = data['Gender'].value_counts().values

counts = data['Gender'].value_counts().index

colors = ['yellow','red']

plt.pie(values,labels = counts,colors = colors,autopct='%1.1f')

plt.title('Comparision of Gender')

plt.legend()

plt.show()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['Illness'].value_counts().index, y = df['Illness'].value_counts().values)

plt.title('Ilness')

plt.xlabel('Counts')

plt.ylabel('Type')

plt.show()
rcParams['figure.figsize'] = 10,7

values = data['Illness'].value_counts().values

counts = data['Illness'].value_counts().index

colors = ['yellow','red']

explode = [0,0.1]

plt.pie(values,labels = counts,colors = colors,explode = explode,autopct='%1.1f')

plt.title('Illness')

plt.legend()

plt.show()
rcParams['figure.figsize'] = 10,5

ax = df['Age'].hist(bins = 15,alpha = 0.9, color = 'cyan')

ax.set(xlabel = 'Age',ylabel = 'Count',title = 'Visualization of Ages')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.heatmap(df.corr(),annot = True,square = True,linewidths = 2,linecolor = 'black',cmap="YlGnBu")
rcParams['figure.figsize'] = 10,5

ax = df['Income'].hist(bins = 15,alpha = 0.9, color = 'cyan')

ax.set(xlabel = 'Incomes',ylabel = 'Count',title = 'Visualization of Incomes')

plt.show()
rcParams['figure.figsize'] = 15,5

sb.countplot(x = 'City',hue = 'Gender',data = df,palette=['c','b'])

plt.title('Gender Distribution by city')

plt.show()
rcParams['figure.figsize'] = 15,5

sb.countplot(x = 'Illness',hue = 'Gender',data = df,palette=['c','b'])

plt.title('Gender Classification by illness')

plt.show()
rcParams['figure.figsize'] = 15,5

sb.boxenplot(x = df['Income'], y = df['City'])

plt.title('Distribution of income per city')

plt.show()
m = df[df['Gender'] == 'Male']

f = df[df['Gender'] == 'Female']

x = pd.Series(m['Income'])

y = pd.Series(f['Income'])

rcParams['figure.figsize'] = 15,5

plt.hist(x,alpha = 0.7,label = 'Male')

plt.hist(y,alpha = 0.4,label = 'Female')

plt.title('Income Distribution by Gender')

plt.xlabel('Income')

plt.ylabel('Count')

plt.legend()

plt.show()