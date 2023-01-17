# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"));



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); 
df = pd.read_csv('../input/zomato.csv', encoding='latin1')
df['Country Code'].value_counts().head(5)
df['City'].value_counts().head(5)
df.groupby('Cuisines')['Aggregate rating'].mean()
df.groupby('Aggregate rating')['Average Cost for two'].mean().plot(kind='bar')

plt.ylabel('Average Cost for two') 

plt.show();
df.groupby('Has Table booking')['Aggregate rating'].mean().plot(kind='bar')

plt.ylabel('Aggregate rating') 

plt.show();
df.groupby('Rating text')['Longitude', 'Latitude'].mean().plot(kind='bar')

plt.ylabel('Longitude and Latitude') 

plt.show();
df.groupby('Has Online delivery')['Aggregate rating'].mean().plot(kind='bar')

plt.ylabel('Aggregate rating') 

plt.show();
df.groupby('Aggregate rating')['Votes'].mean().plot(kind='bar')

plt.ylabel('Votes') 

plt.show();