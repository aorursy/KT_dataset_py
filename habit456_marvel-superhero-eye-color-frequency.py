import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



df_marvel = pd.read_csv('../input/marvel-wikia-data.csv', index_col=1)



# cleaning up dataframes

df_marvel['src'] = 'marvel'

df_marvel = df_marvel.drop(columns=["GSM", "urlslug", "page_id"])



df_marvel.head()

df_marvel.EYE = df_marvel.EYE.str.split().str.get(0).astype('category')

df_eye = df_marvel[["EYE"]]

df_eye = df_eye.dropna(thresh=1)



df_eye.head()
eye_colors = []



for label, color in df_eye.iterrows():

    if color['EYE'] not in eye_colors:

        eye_colors.append(color['EYE'])



print(eye_colors)
df_eye[df_eye['EYE'] == 'Pink']
df_eye.apply(pd.value_counts).plot.bar(title='Marvel Superhero Eye Color Frequency', figsize=(20, 9), fontsize=16, stacked=True, color='g')

plt.xlabel("Eye Color")

plt.ylabel("Frequency")

plt.rcParams.update({'font.size': 16})

plt.legend(['Color'])



plt.show()