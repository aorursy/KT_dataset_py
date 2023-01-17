# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/buildingdatagenomeproject2/solar_cleaned.csv')

df.head()
labels = []

values = []

for col in df.columns:

    labels.append(col)

    values.append(df[col].isnull().sum())

    print(col, values[-1])
ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,50))

rects = ax.barh(ind, np.array(values), color='y')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

#autolabel(rects)

plt.show()
cols_to_use = ['Bobcat_education_Dylan', 'Bobcat_education_Alissa', 'Bobcat_education_Coleman', 'Bobcat_other_Timothy']

fig = plt.figure(figsize=(8, 20))

plot_count = 0

for col in cols_to_use:

    plot_count += 1

    plt.subplot(4, 1, plot_count)

    plt.scatter(range(df.shape[0]), df[col].values)

    plt.title("Distribution of "+col)

plt.show()
temp_df = df.groupby('Bobcat_education_Dylan')['Bobcat_other_Timothy'].agg('mean').reset_index().sort_values(by='Bobcat_other_Timothy')

temp_df.head()
id_to_use = [0.20, 9.32, 0.03, 0.04, 0.07]

fig = plt.figure(figsize=(8, 25))

plot_count = 0

for id_val in id_to_use:

    plot_count += 1

    plt.subplot(5, 1, plot_count)

    temp_df = df.loc[df['Bobcat_education_Dylan']==id_val,:]

    plt.plot(temp_df.timestamp.values, temp_df.Bobcat_other_Timothy.values)

    plt.plot(temp_df.timestamp.values, temp_df.Bobcat_other_Timothy.cumsum())

    plt.title("Asset ID : "+str(id_val))
temp_df = df.groupby('Bobcat_education_Dylan')['Bobcat_other_Timothy'].agg('mean').reset_index().sort_values(by='Bobcat_other_Timothy')

temp_df.tail()
id_to_use = [26.79, 27.07, 27.18, 27.54, 27.59]

fig = plt.figure(figsize=(8, 25))

plot_count = 0

for id_val in id_to_use:

    plot_count += 1

    plt.subplot(5, 1, plot_count)

    temp_df = df.loc[df['Bobcat_education_Dylan']==id_val,:]

    plt.plot(temp_df.timestamp.values, temp_df.Bobcat_other_Timothy.values)

    plt.plot(temp_df.timestamp.values, temp_df.Bobcat_other_Timothy.cumsum())

    plt.title("Asset ID : "+str(id_val))
temp_df = df.groupby('Bobcat_education_Dylan')['Bobcat_other_Timothy'].agg('count').reset_index().sort_values(by='Bobcat_other_Timothy')

temp_df.tail()
id_to_use = [0.07, 0.04, 0.06, 0.03, 0.00]

fig = plt.figure(figsize=(8, 25))

plot_count = 0

for id_val in id_to_use:

    plot_count += 1

    plt.subplot(5, 1, plot_count)

    temp_df = df.loc[df['Bobcat_education_Dylan']==id_val,:]

    plt.plot(temp_df.timestamp.values, temp_df.Bobcat_other_Timothy.values)

    plt.plot(temp_df.timestamp.values, temp_df.Bobcat_other_Timothy.cumsum())

    plt.title("Asset ID : "+str(id_val))
import plotly.express as px

fig = px.line(df, x="timestamp", y="Bobcat_education_Dylan", color_discrete_sequence=['darksalmon'], 

              title="Bobcat Education Dylan")

fig.show()
fig = px.scatter(df, x="timestamp", y="Bobcat_education_Dylan",color_discrete_sequence=['#4257f5'], title="Bobcat Education Dylan" )

fig.show()