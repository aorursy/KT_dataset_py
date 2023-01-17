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
!pip install plotly
import plotly.express as px

df = pd.read_csv(r"/kaggle/input/bengaliai-cv19/train.csv")
fig = px.histogram(df, x="grapheme_root", histnorm='probability', title="Grapheme Occurance Count in order of Grapheme Root")

fig.show()
# Grapheme Count: 

dfb = df.groupby("grapheme_root").count()

fig = px.bar(dfb["grapheme"].sort_values(), y="grapheme", title="Grapheme Occurance Count Sorted from Low to High")

fig.show()
#%%

# Vowel Diacritic Count

fig = px.histogram(df, x="vowel_diacritic", title="Vowel Diacritic Occurance Count")

fig.show()
#%%

dfb = df.groupby("vowel_diacritic").count()

fig = px.bar(dfb["grapheme"].sort_values(), y="grapheme", title="Vowel Diacritic Occurance Count Sorted from Low to High")

fig.show()
#%%

fig = px.histogram(df, x="consonant_diacritic", title="Vowel Diacritic Occurance Count")

fig.show()

#%%
# Grapheme Count: 

dfb = df.groupby("consonant_diacritic").count()

fig = px.bar(dfb["grapheme"].sort_values(), y="grapheme", title="Consonant Diacritic Occurance Count Sorted from Low to High")

fig.show()

#%%
