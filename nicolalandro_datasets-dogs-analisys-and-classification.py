import pandas as pd
import matplotlib.pyplot as plt
#all_path = '/kaggle/input/cats-and-dogs-breeds-classification-oxford-dataset/annotations/annotations/list.txt'
test_path = '/kaggle/input/cats-and-dogs-breeds-classification-oxford-dataset/annotations/annotations/test.txt'
train_path = '/kaggle/input/cats-and-dogs-breeds-classification-oxford-dataset/annotations/annotations/trainval.txt'
#all_df = pd.read_csv(train_path, sep=' ', skiprows=range(0, 6), names=['img', 'id', 'species', 'breed_id'])
test_df = pd.read_csv(test_path, sep=' ', skiprows=None, names=['img', 'id', 'species', 'breed_id'])
train_df = pd.read_csv(train_path, sep=' ', skiprows=None, names=['img', 'id', 'species', 'breed_id'])
train_dogs = train_df[train_df['species'] == 2]
species = dict()

for i, row in train_dogs.iterrows():
    name = row['img']
    dog_class = '_'.join(name.split('_')[:-1])
    if dog_class in species.keys():
        species[dog_class] += 1
    else:
        species[dog_class] = 1
plt.rcParams["figure.figsize"] = (20,5)
plt.bar(range(len(species)), list(species.values()), align='center')
plt.xticks(range(len(species)), [ f'({i}) {s}' for i, s in enumerate(species.keys())])
plt.xticks(rotation=70)
plt.show()
test_dogs = test_df[test_df['species'] == 2]
species = dict()

for i, row in test_dogs.iterrows():
    name = row['img']
    dog_class = '_'.join(name.split('_')[:-1])
    if dog_class in species.keys():
        species[dog_class] += 1
    else:
        species[dog_class] = 1
plt.rcParams["figure.figsize"] = (20,5)
plt.bar(range(len(species)), list(species.values()), align='center')
plt.xticks(range(len(species)), [ f'({i}) {s}' for i, s in enumerate(species.keys())])
plt.xticks(rotation=70)
plt.show()
