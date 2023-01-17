import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import operator



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time

# Read in the data CSV files

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

ss = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
assess_list = ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter']

world_list = ['MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES']
for a in assess_list:

    world = train.loc[train['title'].str.contains(a), 'world'].unique()

    print('{} Assessment in {} World'.format(a, world))
print('Find_course functions')

def world_in_label(df, world):

    if world == 'MAGMAPEAK':

        df = df.loc[df['title'].str.contains('Cauldron Filler')]

    elif world == 'TREETOPCITY':

        df = df.loc[(df['title'].str.contains('Bird Measurer')) | (df['title'].str.contains('Mushroom Sorter'))]

    elif world == 'CRYSTALCAVES':

        df = df.loc[(df['title'].str.contains('Cauldron Filler')) | (df['title'].str.contains('Chest Sorter'))]

    

    return df['installation_id'].unique()



def find_course(world):

    label_ids = world_in_label(train_labels, world)

    world_df = train.query('world=="{}"'.format(world))[['installation_id', 'world', 'title']]

    world_df = world_df.loc[world_df['installation_id'].isin(label_ids)]

    ids = world_df['installation_id'].unique()

    num_course = world_df['title'].nunique()

    

    course_d = dict()

    

    print('{} ids'.format(len(ids)))

    for i in ids:

        if i not in label_ids:

            continue

        else:

            id_df = world_df.query('installation_id=="{}"'.format(i))

            if id_df['title'].nunique() == num_course:

                try:

                    course_d[str(id_df['title'].unique())] += 1

                except:

                    course_d[str(id_df['title'].unique())] = 1    

    return course_d
w_dict = dict()

for w in world_list:

    print('World : {}'.format(w))

    w_dict[w] = find_course(w)

    print(sorted(w_dict[w].items(), key=operator.itemgetter(1), reverse=True)[:5])

    print('='*100)
fig, ax = plt.subplots(1,3, figsize=(18,6))

for num, w in enumerate(world_list):

#with sns.axes_style('Set1'):

    plot = pd.DataFrame(sorted(w_dict[w].items(), key=operator.itemgetter(1), reverse=True)[:5]).plot(kind='bar', cmap='summer', ax=ax[num])

    plot.set_title(w, fontsize=20)

    plot.patches[0].set_color('orange')

    plot.set_xticklabels(['Path {}'.format(i+1)  for i in range(5)], rotation=45)

    plot.legend().remove()
temp = train.query('installation_id=="002db7e3"')

temp = temp.loc[temp['world'] == 'MAGMAPEAK']

print('"Magma Peak - Level 1" type is {}'.format(train.loc[train['title'].str.contains('Magma Peak - Level 1')]['type'].unique()))

display(temp.loc[temp['title'].str.contains('Level 1')])

print('"Magma Peak - Level 2" type is {}'.format(train.loc[train['title'].str.contains('Magma Peak - Level 2')]['type'].unique()))

display(temp.loc[temp['title'].str.contains('Level 2')])
#Save dict

pd.DataFrame(w_dict).to_csv('course_dict.csv')
%%time

train.loc[train['installation_id'] == '0006a69f']['world'].unique()
%%time

train.query('installation_id=="0006a69f"')['world'].unique()