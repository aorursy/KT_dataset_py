import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use("ggplot")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
data.head()
data.drop(['date'] , axis=1 , inplace=True)
sns.heatmap(data.isnull() , yticklabels = False , cbar = False , cmap='viridis')
data.info()
df = data.groupby('state')['number'].sum().sort_values(ascending=False)



plt.figure(figsize = (15,7))

df.plot.bar()

plt.title("Number of the Forest Fires Related To States")

plt.ylabel('Avg of Fires')

plt.xlabel('States')
df = data.groupby('month')['number'].sum().sort_values(ascending = False)



plt.figure(figsize = (15,7))

df.plot.bar(color='purple')

plt.title("Number of the Forest Fires Related To Months")

plt.ylabel('Avg of Fires')

plt.xlabel('Months')
df = data.groupby('year')['number'].sum().reset_index()



plt.figure(figsize=(18,6))

gr = sns.lineplot( x = 'year', y = 'number',data = df, color = 'blue', lw = 3)

gr.xaxis.set_major_locator(plt.MaxNLocator(19)) # 19 values between 1998 - 2017

gr.set_xlim(1998, 2017) # set to x

sns.set()



plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.title('Number of Fires per Year in Brazil',fontsize=18)

plt.xlabel('Year', fontsize = 18)

plt.ylabel('Number of Fires', fontsize = 18)
df = pd.DataFrame(data[data['state'] == 'Amazonas'])

new_df = df.groupby('year')['number'].sum().reset_index()



plt.figure(figsize=(17,8))

gr = sns.lineplot( x = 'year', y = 'number',data = new_df, color = 'red', lw = 2.5)

gr.xaxis.set_major_locator(plt.MaxNLocator(19)) # 19 values between 1998 - 2017

gr.set_xlim(1998, 2017) # set to x

sns.set()



plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.title('Number of Fires per Year In Amazon ',fontsize=18)

plt.xlabel('Year', fontsize = 18)

plt.ylabel('Number of Fires', fontsize = 18)