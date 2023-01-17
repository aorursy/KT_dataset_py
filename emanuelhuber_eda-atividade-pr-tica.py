import os

from collections import defaultdict

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image # biblioteca para o processamento de imagens

from tqdm import tqdm_notebook as tqdm # biblioteca para exibição de barra de progresso
!ls -lh ../input/vehicle/test/testset
!ls -lh ../input/vehicle/train/train
!ls -lh ../input/vehicle/test/testset | head -5
root = '../input/vehicle/train/train'

data = []

for category in sorted(os.listdir(root)):

    for file in sorted(os.listdir(os.path.join(root, category))):

        data.append((category, os.path.join(root, category,  file)))



df = pd.DataFrame(data, columns=['class', 'file_path'])
df.head()
# Cuidado ao usar o método count(), ele retorna a contagem de atributos não nulos. Logo, se houver um atributo com valor nulo, ele não entrará na contagem.

df.count()
root = '../input/vehicle/train/train'

for root, directories, files in os.walk(root):

    if len(files)==0:

        pass

    else:

        print(root)#exibir dretório

        print(len(files))#exibir quantidade de arquivos por pastas
df.shape[0]
root = '../input/vehicle/test/testset'

data = []

for file in sorted(os.listdir(root)):

    data.append(file)



dft = pd.DataFrame(data, columns=['file_path'])

dft.count()
test_data = '../input/vehicle/test/testset'

os.listdir(test_data)

for root_test, directories_test, files_test in os.walk(test_data):

    print(root_test)

    print(len(files_test))
dft.shape[0]
df.groupby('class').size()
pics_by_class = df.groupby(['class']).count()

pics_by_class.rename(columns = {'file_path':'quantity'}, inplace = True)

pics_by_class_order = pics_by_class.sort_values(by='quantity')

pics_by_class_order
df['class'].value_counts(ascending=True)
df['class'].value_counts().plot(kind='bar')

plt.title('Class counts')
max_value = pics_by_class_order.max().quantity



unbalanced = [] 

mean = pics_by_class_order.mean()[0]

tolerance = mean*0.15#desvio padrao, abaixo do 1ºquartil ou acima 4º quartil



for pics in pics_by_class_order.iloc[:, -1]:

    if pics<(mean-tolerance) or pics>(mean+tolerance):

        unbalanced.append(pics)



print(unbalanced)
df['class'].value_counts().mean()
fig = plt.figure(figsize=(25, 16))

for num, category in enumerate(sorted(df['class'].unique())):

    for i, (idx, row) in enumerate(df.loc[df['class'] == category].sample(4).iterrows()):

        ax = fig.add_subplot(17, 4, num * 4 + i + 1, xticks=[], yticks=[])

        im = Image.open(row['file_path'])

        plt.imshow(im)

        ax.set_title(f'Class: {category}')

fig.tight_layout()

plt.show()
from PIL import Image 

root = "../input/vehicle/train/train/Ambulance"

for file in sorted(os.listdir(os.path.join(root))):

    data = Image.open(os.path.join(root,file))

    print(data.size)
unique_sizes = set()

root = "../input/vehicle/train/train/Ambulance"

for file in sorted(os.listdir(os.path.join(root))):

    data = Image.open(os.path.join(root,file))

    unique_sizes.add(data.size)

print(f"Tamanhos únicos encontrados {len(unique_sizes)} para uma única classe")
data = defaultdict(lambda: defaultdict(list))

modified_df = df

width = []

height = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    image = Image.open(row[1])

    data[row[0]]['width'].append(image.size[0])

    width.append(image.size[0])

    data[row[0]]['height'].append(image.size[1])

    height.append(image.size[1])

def plot_dist(category):

    '''plot height/width dist curves for given category'''

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.distplot(data[category]['height'], color='darkorange', ax=ax).set_title(category, fontsize=16)

    sns.distplot(data[category]['width'], color='purple', ax=ax).set_title(category, fontsize=16)

    plt.xlabel('size', fontsize=15)

    plt.legend(['height', 'width'])

    plt.show()

for category in df['class'].unique():

    plot_dist(category)
modified_df['width'] = width

modified_df['height'] = height

div = {v:i+1 for i, v in enumerate(modified_df['class'].unique()) }

modified_df['mod_class'] = [div[x] for x in modified_df.iloc[:, 0]]

modified_df.corr()