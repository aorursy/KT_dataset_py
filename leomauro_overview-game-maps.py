# Pandas

import numpy as np

import pandas as pd

# Plot

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('seaborn-whitegrid')

%matplotlib inline
dataset_folder = '../input'

plot_folder = '../plot'



courses = pd.read_csv('%s/%s.csv' % (dataset_folder, 'courses'), sep='\t', encoding='utf-8')

players = pd.read_csv('%s/%s.csv' % (dataset_folder, 'players'), sep='\t', encoding='utf-8')
courses.head(3)
# palette of colors

palette = sns.color_palette('cubehelix', 4)

sns.palplot(palette)
# autopct pie plot - function

def func(pct, allvals):

    absolute = float(pct/100.*np.sum(allvals))/1000.0

    return "{:.1f}%\n({:.1f}k)".format(pct, absolute)



# plot

fontsize = 14
# values

labels = courses['difficulty'].unique().tolist()

values = [sum(courses['difficulty'] == label) for label in labels]

print(list(zip(labels, values)))

explode = [0.03] * len(values)



# plot

fig, ax = plt.subplots()

ax.pie(values, autopct=lambda pct: func(pct, values), pctdistance=0.45,

       colors=palette, explode=explode, labels=labels,

       textprops={'fontsize':fontsize,'weight':'bold'})

centre_circle = plt.Circle((0,0),0.75,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# figure

ax.axis('equal')

plt.tight_layout()

# plt.savefig('%s/%s.pdf' % (plot_folder, 'difficulty'), dpi=300, bbox_inches='tight')

plt.show()
# values

labels = courses['gameStyle'].unique().tolist()

values = [sum(courses['gameStyle'] == label) for label in labels]

print(list(zip(labels, values)))

explode = [0.03] * len(values)



# plot

fig, ax = plt.subplots()

ax.pie(values, autopct=lambda pct: func(pct, values), pctdistance=0.45,

       colors=palette, explode=explode, labels=labels,

       textprops={'fontsize':fontsize,'weight':'bold'})

centre_circle = plt.Circle((0,0),0.75,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# figure

ax.axis('equal')

plt.tight_layout()

# plt.savefig('%s/%s.pdf' % (plot_folder, 'gameStyle'), dpi=300, bbox_inches='tight')

plt.show()
makers = courses['maker'].value_counts().to_dict()

print('number of makers: %d' % (len(makers)))
# values

top = 25

labels = list(makers.keys())[0:top]

x_axis = range(len(labels))

y_axis = list(makers.values())[0:top]



# plot

fig, ax = plt.subplots()

plt.bar(x_axis, y_axis, align='center', color=palette[0])

plt.xticks(x_axis, labels, rotation=90)

plt.show()
players = players.set_index('id')

players.head()
# players

df = pd.DataFrame(makers, index=['courses']).transpose()

df = df.rename(columns={'index':'id'})
df2 = pd.concat([df, players], sort=True, axis=1)

df2 = df2.dropna(subset=['courses']).sort_values(by=['courses'], ascending=False)

df2.head()
countries = {flag:0 for flag in df2['flag'].unique().tolist()}
for maker, row in df2.iterrows():

    countries[row['flag']] += int(row['courses'])
# values

labels = list(countries.keys())

values = [countries[label] for label in labels]

print(countries)

explode = [0.03] * len(labels)



# plot

fig, ax = plt.subplots()

ax.pie(values, autopct=lambda pct: func(pct, values), pctdistance=0.45,

       colors=palette, explode=explode, labels=labels,

       textprops={'fontsize':fontsize,'weight':'bold'})

centre_circle = plt.Circle((0,0),0.75,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# figure

ax.axis('equal')  

plt.tight_layout()

# plt.savefig('%s/%s.pdf' % (plot_folder, 'countries'), dpi=300, bbox_inches='tight')

plt.show()