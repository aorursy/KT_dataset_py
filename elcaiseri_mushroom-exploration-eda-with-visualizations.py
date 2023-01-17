import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_path = os.path.join(dirname, filename)

data = pd.read_csv(data_path)

data.head()
data.shape
data.isnull().sum()
data.describe()
from sklearn.preprocessing import LabelEncoder



data_encoded = data.copy()

le = LabelEncoder()

for col in data_encoded.columns:

    data_encoded[col] = le.fit_transform(data_encoded[col]) 

    

data_encoded.head()
data_encoded.describe()
data.columns
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',

         'axes.labelsize': 'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)
def plot_col(col, hue=None, color=['red', 'lightgreen'], labels=None):

    fig, ax = plt.subplots(figsize=(15, 7))

    sns.countplot(col, hue=hue, palette=color, saturation=0.6, data=data, dodge=True, ax=ax)

    ax.set(title = f"Mushroom {col.title()} Quantity", xlabel=f"{col.title()}", ylabel="Quantity")

    if labels!=None:

        ax.set_xticklabels(labels)

    if hue!=None:

        ax.legend(('Poisonous', 'Edible'), loc=0)
class_dict = ('Poisonous', 'Edible')

plot_col(col='class', labels=class_dict)
shape_dict = {"bell":"b","conical":"c","convex":"x","flat":"f", "knobbed":"k","sunken":"s"}

labels = ('convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical')

plot_col(col='cap-shape', hue='class', labels=labels)
color_dict = {"brown":"n","yellow":"y", "blue":"w", "gray":"g", "red":"e","pink":"p",

              "orange":"b", "purple":"u", "black":"c", "green":"r"}

plot_col(col='cap-color', color=color_dict.keys(), labels=color_dict)
plot_col(col='cap-color', hue='class', labels=color_dict)
surface_dict = {"smooth":"s", "scaly":"y", "fibrous":"f","grooves":"g"}

plot_col(col='cap-surface', hue='class', labels=surface_dict)
def get_labels(order, a_dict):    

    labels = []

    for values in order:

        for key, value in a_dict.items():

            if values == value:

                labels.append(key)

    return labels
odor_dict = {"almond":"a","anise":"l","creosote":"c","fishy":"y",

             "foul":"f","musty":"m","none":"n","pungent":"p","spicy":"s"}

order = ['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm']

labels = get_labels(order, odor_dict)      

plot_col(col='odor', color=color_dict.keys(), labels=labels)
stalk_cats = ['class', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 

              'stalk-color-above-ring', 'stalk-color-below-ring']

data_cats = data_encoded[stalk_cats]

sns.pairplot(data_cats, hue='class', kind='reg')
fig, ax = plt.subplots(3, 2, figsize=(20, 15))

for i, axis in enumerate(ax.flat):

    sns.distplot(data_cats.iloc[:, i], ax=axis)
pop_dict = {"abundant":"a","clustered":"c","numerous":"n","scattered":"s","several":"v","solitary":"y"}

hab_dict = {"grasses":"g","leaves":"l","meadows":"m","paths":"p","urban":"u","waste":"w","woods":"d"}
f, ax = plt.subplots(figsize=(15, 10))

order = list(data['population'].value_counts().index)

pop_labels = get_labels(order, pop_dict)

explode = (0.0,0.01,0.02,0.03,0.04,0.05)

data['population'].value_counts().plot.pie(explode=explode , autopct='%1.1f%%', labels=pop_labels, shadow=True, ax=ax)

ax.set_title('Mushroom Population Type Percentange');
f, ax = plt.subplots(figsize=(15, 10))

order = list(data['habitat'].value_counts().index)

hab_labels = get_labels(order, hab_dict)

explode = (0.0,0.01,0.02,0.03,0.04,0.05, 0.06)

data['habitat'].value_counts().plot.pie(explode=explode, autopct='%1.1f%%', labels=hab_labels, shadow=True, ax=ax)

ax.set_title('Mushroom Habitat Type Percentange');