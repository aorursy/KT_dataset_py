import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import seaborn as sns

sns.set(style="darkgrid")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_zoo = pd.read_csv("../input/zoo-animal-classification/zoo.csv")

df_zoo.head()
df_zoo.shape
df_class = pd.read_csv("../input/zoo-animal-classification/class.csv")

df_class
df_class.shape
len(df_zoo[df_zoo['class_type'] == 1].index)
list(df_zoo[df_zoo['class_type'] == 1]['animal_name'])
df_zoo2 = pd.read_csv("../input/zoo-animals-extended-dataset/zoo2.csv")

df_zoo3 = pd.read_csv("../input/zoo-animals-extended-dataset/zoo3.csv")
df_zoo2.shape
df_zoo3.shape
df_zoo['id_zoo'] = 'zoo1'

df_zoo2['id_zoo'] = 'zoo2'

df_zoo3['id_zoo'] = 'zoo3'
zoo_complet = pd.concat([df_zoo, df_zoo2, df_zoo3], axis=0)
zoo_complet.dtypes
for col in ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'tail', 'domestic', 'catsize']:

    zoo_complet[col] = zoo_complet[col].astype('bool')

    

zoo_complet['legs'] = zoo_complet['legs'].astype('category')
zoo_complet.dtypes
print ("Nombre d'animaux dans zoo.csv : %d" % df_zoo.shape[0])

print ("Nombre d'animaux dans zoo2.csv : %d" % df_zoo2.shape[0])

print ("Nombre d'animaux dans zoo3.csv : %d" % df_zoo3.shape[0])

print ("Nombre d'animaux dans la fusion : %d" % zoo_complet.shape[0])
ax = sns.countplot(x="class_type", data=zoo_complet) 
ax = sns.countplot(x="class_type", hue="id_zoo", data=zoo_complet)         
descriptors_columns = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'tail', 'legs', 'domestic', 'catsize']



zoo_complet.groupby(descriptors_columns).filter(lambda g: (g['class_type'].nunique() > 1)).sort_values(by=descriptors_columns)
classes = ["Mammifère", "Oiseau", "Reptile", "Poisson", "Amphibia", "Insecte", "Invertébré"]
zoo_complet['class_type'].value_counts().sort_index()
ax = sns.countplot(x="class_type", data=zoo_complet)

ax.set_xticklabels(classes, rotation=45)

plt.show()
ax = sns.countplot(x="id_zoo", hue="class_type", data=zoo_complet)

ax.set_xticklabels(["Zoo #1", "Zoo #2", "Zoo #3"], rotation=45)

plt.legend(title='Légende', ncol=2, labels=classes)

plt.show()
ax = sns.countplot(x="class_type", hue="id_zoo", data=zoo_complet)

ax.set_xticklabels(classes, rotation=45)

plt.legend(title='Légende', labels=["Zoo #1", "Zoo #2", "Zoo #3"])

plt.show()
plt.subplots_adjust(left=0,

                    bottom=0,

                    right=2,

                    top=2,

                    wspace=0.35, 

                    hspace=0.35)

    

    

cpt = 0

for col in descriptors_columns:

    cpt += 1

    fig = plt.subplot(4,4,cpt)

    ax = sns.countplot(x=col, hue="class_type", data=zoo_complet)

    plt.legend(title='Légende', ncol=2, labels=classes)

plt.show()


