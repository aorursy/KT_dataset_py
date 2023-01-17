import numpy as np

import pandas as pd

import  matplotlib.pyplot as plt
df = pd.DataFrame(columns=['id', 'longitude', 'latitude'])
import random

random.uniform(10,100)
x =range(1,1000)

y =[]

for n in x:

    a = random.uniform(0,100)

    y =np.append(y, a)
z =[]

for n in x:

    b = random.uniform(0,100)

    z =np.append(z, b)
df['longitude'] = y

df['latitude'] = z
id = np.arange(1,1000)
df['id'] = id
x_coords = df['longitude']

y_coords = df['latitude']



plt.figure(figsize=(20,20))



for i,type in enumerate(df['id']):

    x = x_coords[i]

    y = y_coords[i]

    plt.scatter(x, y, color='red')

    plt.text(x+0.5, y+0.5, type, fontsize=9)

plt.show()
df.head()
x = 485

x = int(x)

x= x-1

dist =[]



import math



for row in range(len(df)):

    eu_dist = math.sqrt(((df.iloc[x,1]-df.iloc[row,1])**2)+(df.iloc[x,2]-df.iloc[row,2])**2)

    dist.append(eu_dist)
new_df = df.copy()

new_df['eu_dist'] = dist
new_df = new_df.sort_values(by=['eu_dist'])
new_df['close_x'] = 0
new_df.iloc[0:15, 4]=1
x_coords = new_df['longitude']

y_coords = new_df['latitude']



plt.figure(figsize=(20,20))



categories = np.array(new_df['close_x'])

colormap = np.array(['r', 'g'])



for i,type in enumerate(new_df['id']):

    x = x_coords[i]

    y = y_coords[i]

    if new_df.close_x[i]==1:

        plt.scatter(x, y, color='r')

        plt.text(x+0.5, y+0.5, type, fontsize=9)

    else:

        plt.scatter(x, y, color='g')

        plt.text(x+0.5, y+0.5, type, fontsize=9)

plt.show()