# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib

import matplotlib.pyplot as plt
df = pd.read_csv('../input/all_parts_sets.csv')

df['rgb'] = '#'+df['rgb']

df.shape
# exclude Minitalia and other themes

df = df[~df.theme_name.isin(['Minitalia','Supplemental','NXT','Food & Drink','HO 1:87 Vehicles','Power Functions', 'Key Chain', 'Clikits',

       'Control Lab','Service Packs','Brickheadz','Game','Books','Juniors'])]

themelist = [279,444,445,457,531,20,426,121,480,125,459]

df = df[~df.theme_id.isin(themelist)]
columnlist = ['quantity','theme_name','theme_id']

themes = df[columnlist]

group_themes = themes.groupby('theme_name')

group_themes.describe()
df.theme_name.unique()
df.part_category_name.unique()
columnlist = ['part_num','quantity','part_name','part_category_id','part_category_name','color_name','rgb','set_name','debut_year','part_count','theme_name']

df = df[columnlist]

df.head()
group_partcat = df.groupby('part_category_name')

group_partcat.describe()
columnlist = ['quantity','part_name','part_category_name','color_name','rgb','set_name','debut_year','theme_name']

df2 = df[columnlist]

group_color = df2.groupby(['color_name','rgb'])

group_color.describe()
bricks = df.groupby('part_category_name').get_group('Bricks')

bricks.describe()
columnlist = ['part_name','quantity','color_name','debut_year','theme_name','rgb']

bricks = bricks[columnlist]

group_partname = bricks.groupby('part_name')

group_partname.describe()
group_partname.head()
# exclude technic pins

df = df[(df['part_category_id'] != 53)]

df.head()
coloryearmin = pd.DataFrame(df.groupby(['color_name', 'rgb'], as_index=False)['debut_year'].min())

coloryearmin.reindex()

coloryearmin.head()
coloryearmax = pd.DataFrame(df.groupby(['color_name', 'rgb'], as_index=False)['debut_year'].max())

coloryearmax.reindex()

coloryearmax = coloryearmax.rename(index=str, columns={'debut_year': 'final_year'})

coloryearmax.head()
coloryears = pd.merge(coloryearmin,coloryearmax)

coloryears.head()
colors = pd.DataFrame(df.groupby(['color_name', 'rgb'], as_index=False)['quantity'].count())

#colors = colors[(colors['quantity'] > 40) & (colors['quantity'] <= 100)]

#colors = colors[(colors['quantity'] > 100) & (colors['quantity'] <= 200)]

#colors = colors[(colors['quantity'] > 200) & (colors['quantity'] <= 400)]

#colors = colors[(colors['quantity'] > 400) & (colors['quantity'] <= 800)]

#colors = colors[(colors['quantity'] > 800) & (colors['quantity'] <= 1200)]

#colors = colors[(colors['quantity'] > 1200) & (colors['quantity'] <= 2000)]

#colors = colors[(colors['quantity'] > 2000) & (colors['quantity'] <= 10000)]

colors = colors[(colors['quantity'] > 10000)]

colors.reindex()

colors.head()
barlist=plt.bar(colors.color_name, colors.quantity, color=colors['rgb'])

plt.rcParams["figure.figsize"] = [16, 9]

plt.show()
vintagesets = df[(df['debut_year'] <= 1980)]

vintagesets.head()
vintagecolors = pd.DataFrame(vintagesets.groupby(['color_name', 'rgb'], as_index=False)['quantity'].count())

vintagecolors = vintagecolors[(vintagecolors['quantity'] > 10) & (vintagecolors['quantity'] <= 500)]

vintagecolors.reindex()

vintagecolors.head()
barlist=plt.bar(vintagecolors.color_name, vintagecolors.quantity, color=vintagecolors['rgb'])

plt.rcParams["figure.figsize"] = [16, 9]

plt.show()
brickcolors = pd.DataFrame(bricks.groupby(['color_name', 'rgb'], as_index=False)['quantity'].count())

brickcolors = brickcolors[(brickcolors['quantity'] > 50) & (brickcolors['quantity'] <= 1000)]

brickcolors.reindex()

brickcolors.head()
barlist=plt.bar(brickcolors.color_name, brickcolors.quantity, color=brickcolors['rgb'])

plt.rcParams["figure.figsize"] = [16, 9]

plt.show()