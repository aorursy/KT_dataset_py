# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import spacy
from collections import Counter
import matplotlib.pyplot as plt

import seaborn as sns
nlp = spacy.load('en_core_web_sm')

raw = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
raw.rename(columns={'target1':'Target','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive', 'natlty1_txt':'Nationality'},inplace=True)
raw=raw[['Target','Target_type','Weapon_type','Motive', 'Nationality']]
plt.subplots(figsize=(15,6))
sns.countplot('Weapon_type', data=raw, palette='inferno', order=raw['Weapon_type'].value_counts()[:7].index, edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Weapon Type of Global Terrorism Attacks')
plt.show()
plt.subplots(figsize=(20,6))
sns.barplot(raw['Nationality'].value_counts()[:20].index, raw['Nationality'].value_counts()[:20], palette='PuBuGn_d')
plt.title('Affected Countries of the Terrorist Attacks')
plt.show()