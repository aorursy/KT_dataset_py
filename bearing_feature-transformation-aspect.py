import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



train = pd.read_csv('/kaggle/input/learn-together/train.csv')

train.shape
type_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']

type_ids = sorted(train['Cover_Type'].unique())



current_palette = sns.color_palette()

n = 3

fig, ax = plt.subplots(3, n, figsize=(25,15))

for t in type_ids:

    x, y = (t-1)//n, (t-1)%n

    ax[x,y].set_title(str(t) + ': ' + type_names[t-1])

    data = train['Aspect'][train['Cover_Type']==t]

    sns.distplot(data, ax=ax[x, y], color=current_palette[t-1]);

    
aspect_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

degree = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360])

def get_aspect_name(aspect):

    d = degree - aspect

    indx = np.where(d > 0, d, np.inf).argmin()

    return aspect_names[indx]



train['Aspect_Name'] = train['Aspect'].apply(get_aspect_name).astype('category')

train['Aspect_Name'].cat.set_categories(aspect_names[:-1])
fig, ax = plt.subplots(3, 3, figsize=(25,15))

for t in type_ids:

    x, y = (t-1)//3, (t-1)%3

    ax[x,y].set_title(str(t) + ': ' + type_names[t-1])

    data = train['Aspect_Name'][train['Cover_Type']==t]

    sns.countplot(x=data, ax=ax[x, y], order=aspect_names[:-1]);
fig, ax = plt.subplots(3, 3, figsize=(20,15))

for t in type_ids:

    x, y = (t-1)//3, (t-1)%3

    ax[x,y].set_title(str(t) + ': ' + type_names[t-1])

    val = train['Aspect_Name'][train['Cover_Type']==t].value_counts().to_dict()

    val_sorted = dict(sorted(val.items(), key=lambda x: aspect_names[:-1].index(x[0])))    

    labels = [key + ' (' + str(val) + ')' for key,val in val_sorted.items()]

    ax[x, y].pie(val_sorted.values(), labels=labels)

    hole = plt.Circle((0,0), 0.5, color='white')

    ax[x, y].add_artist(hole)



plt.show();