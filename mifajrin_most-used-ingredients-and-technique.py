import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ay = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-ayam.csv')

ik = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-ikan.csv')

ka = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-kambing.csv')

sa = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-sapi.csv')

ta = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-tahu.csv')

te = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-telur.csv')

tem = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-tempe.csv')

ud = pd.read_csv('/kaggle/input/indonesian-food-recipes/dataset-udang.csv')
ayam = pd.DataFrame(ay)

ikan = pd.DataFrame(ik)

kambing = pd.DataFrame(ka)

sapi = pd.DataFrame(sa)

tahu = pd.DataFrame(ta)

telur = pd.DataFrame(te)

tempe = pd.DataFrame(tem)

udang = pd.DataFrame(ud)
indofood = pd.concat([ayam, ikan, kambing, sapi, tahu, telur, tempe, udang])

indofood = pd.DataFrame(indofood)
indofood['Ingredients'] = indofood['Ingredients'].astype(str)

indofood['Ingredients'] = indofood['Ingredients'].str.replace('[^a-zA-Z]', ' ')

indofood['Steps'] = indofood['Steps'].astype(str)

indofood['Steps'] = indofood['Steps'].str.replace('[^a-zA-z]',' ')
countingredients = indofood['Ingredients'].str.split(expand=True).stack().value_counts()

ingredients = pd.DataFrame({'ingredients':countingredients.index, 'value':countingredients.values})

countsteps = indofood['Steps'].str.split(expand=True).stack().value_counts()

steps = pd.DataFrame({'steps':countsteps.index,'value':countsteps.values})
ingredients
steps