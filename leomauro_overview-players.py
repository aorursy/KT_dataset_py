# Pandas

import pandas as pd

# Plot

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('seaborn-whitegrid')

%matplotlib inline
file = 'players.csv'

dataset_folder = '../input'



df = pd.read_csv('%s/%s' % (dataset_folder, file), sep='\t', encoding='utf-8')
print('number of players: %d' % (len(df)))

df.head(3)
flags = df['flag'].value_counts().to_dict()

print('number of nationalities: %d' % (len(flags)))
# values

top = 10

labels = list(flags.keys())[0:top]

x_axis = range(len(labels))

y_axis = list(flags.values())[0:top]

print(list(zip(labels, y_axis)))



# plot

fig, ax = plt.subplots()

plt.bar(x_axis, y_axis, align='center')

plt.xticks(x_axis, labels, rotation=90)

plt.show()
# plot

fig, ax = plt.subplots()

ax.pie(y_axis, labels=labels, autopct='%1.1f%%', shadow=True)

ax.axis('equal')

plt.show()