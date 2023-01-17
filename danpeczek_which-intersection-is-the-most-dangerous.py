import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



caracteristics = pd.read_csv('../input/accidents-in-france-from-2005-to-2016/caracteristics.csv', encoding='latin1')
caracteristics.head()
caracteristics.columns[caracteristics.isna().sum() != 0]
int_dict = {

    '1': 'Out of intersection',

    '2': 'X intersection',

    '3': 'T intersection',

    '4': 'Y intersection',

    '5': 'More than 4 branches intersection',

    '6': 'Giratory',

    '7': 'Place',

    '8': 'Level crossing',

    '9': 'Other'



}

caracteristics['int'] = caracteristics['int'].astype(str) 

caracteristics['int'] = caracteristics['int'].replace(int_dict)

caracteristics['int'] = pd.Categorical(caracteristics['int'], list(int_dict.values()))

caracteristics.head()
plt.clf()

plt.figure(figsize=(10,10))

ax = sns.countplot(y = 'int', data=caracteristics)

ax.set_title('Number of accidents based on the intersection type')

ax.set_xlabel('Number of accidents')

ax.set_ylabel('Intersection')

plt.show()