import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
data_dir = '../input/cat-in-the-dat/'

os.listdir(data_dir)
train = pd.read_csv(data_dir + 'train.csv')



print(train.shape)

train.head()
desc = pd.DataFrame(train.nunique())

desc.columns = ['cardinality']



desc['dtype'] = train.dtypes

desc['nulls'] = train.isnull().sum()



desc
# plot display settings

image_columns = 2

image_layer_index = 0

use_figsize = (15, 4)

use_wspace = 0.5



# initialize figure (1st layer)

fig = plt.figure(figsize=use_figsize)

plt.subplots_adjust(wspace=use_wspace)

for i, col in enumerate([x for x in train.columns if 'bin' in x]):

    

    if (i != 0) and (i % image_columns == 0):

        # end of layer, show layer and reset layer_index

        image_layer_index = 0

        plt.show()

        

        # initialize figure (next layer)

        fig = plt.figure(figsize=use_figsize)

        plt.subplots_adjust(wspace=use_wspace)

    

    image_layer_index += 1

    

    # add subplot

    ax = fig.add_subplot(int('1{}{}'.format(image_columns, image_layer_index)))

    

    # specity order

    use_order = sorted(train[col].unique())

    

    sns.countplot(x=col, data=train, order=use_order)



    # add target rate plot

    target_rate = train.groupby([col]).agg({'target': 'mean'})

    

    ax2 = ax.twinx()

    ax2.plot(range(target_rate.shape[0]), target_rate.loc[use_order, 'target'], '--o', color='black', markersize=8)

    ax2.set_ylabel('target rate')

    ax2.grid(False)
# plot display settings

image_columns = 2

image_layer_index = 0

use_figsize = (15, 4)

use_wspace = 0.5



# initialize figure (1st layer)

fig = plt.figure(figsize=use_figsize)

plt.subplots_adjust(wspace=use_wspace)

for i, col in enumerate([x for x in train.columns if ('nom' in x) and (train[x].nunique() < 30)]):

    

    if (i != 0) and (i % image_columns == 0):

        # end of layer, show layer and reset layer_index

        image_layer_index = 0

        plt.show()

        

        # initialize figure (next layer)

        fig = plt.figure(figsize=use_figsize)

        plt.subplots_adjust(wspace=use_wspace)

    

    image_layer_index += 1

    

    # add subplot

    ax = fig.add_subplot(int('1{}{}'.format(image_columns, image_layer_index)))

    

    # order by count

    use_order = train[col].value_counts().index

    

    sns.countplot(x=col, data=train, order=use_order)

    

    # add target rate plot

    target_rate = train.groupby([col]).agg({'target': 'mean'})

    

    ax2 = ax.twinx()

    ax2.plot(range(target_rate.shape[0]), target_rate.loc[use_order, 'target'], '--o', color='black', markersize=8)

    ax2.set_ylabel('target rate')

    ax2.grid(False)
# check if all Trapezoids are Lions



print('..check if all Trapezoids are Lions..')

print('Trapezoid target rate:', train.loc[train['nom_1'] == 'Trapezoid']['target'].mean())



print('\nTrapezoid nom_2 counts:')

train.loc[train['nom_1'] == 'Trapezoid']['nom_2'].value_counts()
# check if all Squares are from Canada



print('..check if all Squares are from Canada..')

print('Square target rate:', train.loc[train['nom_1'] == 'Trapezoid']['target'].mean())



print('\nSquare nom_3 counts:')



print(train.loc[train['nom_1'] == 'Square']['target'].mean())

train.loc[train['nom_1'] == 'Square']['nom_3'].value_counts()
meme_url = plt.imread("https://i.kym-cdn.com/entries/icons/original/000/027/475/Screen_Shot_2018-10-25_at_11.02.15_AM.png")

plt.imshow(meme_url, interpolation='nearest', aspect='equal')

             

# plt.grid(False)

plt.xticks([])

plt.yticks([])



plt.xlabel('...', size=20)



plt.show()
ord_1_manual_order = ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']

ord_2_manual_order = ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']



# plot display settings

image_columns = 2

image_layer_index = 0

use_figsize = (15, 4)

use_wspace = 0.5



# initialize figure (1st layer)

fig = plt.figure(figsize=use_figsize)

plt.subplots_adjust(wspace=use_wspace)

for i, col in enumerate([x for x in train.columns if ('ord' in x) and (train[x].nunique() < 30)]):

    

    if (i != 0) and (i % image_columns == 0):

        # end of layer, show layer and reset layer_index

        image_layer_index = 0

        plt.show()

        

        # initialize figure (next layer)

        fig = plt.figure(figsize=use_figsize)

        plt.subplots_adjust(wspace=use_wspace)

    

    image_layer_index += 1

    

    # add subplot

    ax = fig.add_subplot(int('1{}{}'.format(image_columns, image_layer_index)))

    

    # specify order

    if col == 'ord_1':

        use_order = ord_1_manual_order

    elif col == 'ord_2':

        use_order = ord_2_manual_order

    else:

        use_order = sorted(train[col].unique())

    

    sns.countplot(x=col, data=train, order=use_order)

    

    # add target rate plot

    target_rate = train.groupby([col]).agg({'target': 'mean'})

    

    ax2 = ax.twinx()

    ax2.plot(range(target_rate.shape[0]), target_rate.loc[use_order, 'target'], '--o', color='black', markersize=8)

    ax2.set_ylabel('target rate')

    ax2.grid(False)
feels_good_url = plt.imread("https://i.kym-cdn.com/photos/images/original/000/591/928/94f.png")

plt.imshow(feels_good_url, interpolation='nearest', aspect='equal')

             

# plt.grid(False)

plt.xticks([])

plt.yticks([])



plt.xlabel('how i felt after seeing the plots')



plt.show()
fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(111)



col = 'ord_5'



use_order = sorted(train[col].unique())

sns.countplot(x=col, data=train, order=use_order)



# add target rate plot

target_rate = train.groupby([col]).agg({'target': 'mean'})



ax2 = ax.twinx()

ax2.plot(range(target_rate.shape[0]), target_rate.loc[use_order, 'target'], color='black')

ax2.set_ylabel('target rate')

ax2.grid(False)



plt.xticks([])



print('first 10 in order:', use_order[:10])

plt.show()
fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(111)



train['ord_5_first'] = train['ord_5'].apply(lambda x: x[0])

col = 'ord_5_first'



use_order = sorted(train[col].unique())

sns.countplot(x=col, data=train, order=use_order)



# add target rate plot

target_rate = train.groupby([col]).agg({'target': 'mean'})



ax2 = ax.twinx()

ax2.plot(range(target_rate.shape[0]), target_rate.loc[use_order, 'target'], color='black')

ax2.set_ylabel('target rate')

ax2.grid(False)



print('first 10 in order:', use_order[:10])

plt.show()
# plot display settings

image_columns = 2

image_layer_index = 0

use_figsize = (15, 4)

use_wspace = 0.5



# initialize figure (1st layer)

fig = plt.figure(figsize=use_figsize)

plt.subplots_adjust(wspace=use_wspace)

for i, col in enumerate(['day', 'month']):

    

    if (i != 0) and (i % image_columns == 0):

        # end of layer, show layer and reset layer_index

        image_layer_index = 0

        plt.show()

        

        # initialize figure (next layer)

        fig = plt.figure(figsize=use_figsize)

        plt.subplots_adjust(wspace=use_wspace)

    

    image_layer_index += 1

    

    # add subplot

    ax = fig.add_subplot(int('1{}{}'.format(image_columns, image_layer_index)))

    

    # specity order

    use_order = sorted(train[col].unique())

    

    sns.countplot(x=col, data=train.sort_values(by=col), order=use_order)

    

    # add target rate plot

    target_rate = train.groupby([col]).agg({'target': 'mean'})

    

    ax2 = ax.twinx()

    ax2.plot(range(target_rate.shape[0]), target_rate.loc[use_order, 'target'], '--o', color='black', markersize=8)

    ax2.set_ylabel('target rate')

    ax2.grid(False)