# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

desa=pd.read_csv("../input/desa.csv")

kecamatan=pd.read_csv("../input/kecamatan.csv")

kabupaten=pd.read_csv("../input/kabupaten.csv")

provinsi=pd.read_csv("../input/provinsi.csv")

# Any results you write to the current directory are saved as output.
print(desa.info())

desa.columns=['desa_id','kecamatan_id','desa_name']

desa.head()
print(kecamatan.info())

kecamatan.columns=['kecamatan_id','kabupaten_id','kecamatan_name']

kecamatan.head()
print(kabupaten.info())

kabupaten.columns=['kabupaten_id','provinsi_id','kabupaten_name']

kabupaten.head()
print(provinsi.info())

provinsi.columns=['provinsi_id','parent_code','provinsi_name']

provinsi
df=pd.merge(desa, kecamatan, left_on='kecamatan_id', right_on='kecamatan_id')

df.head()
df=pd.merge(df, kabupaten, left_on='kabupaten_id', right_on='kabupaten_id')

df.head()
df=pd.merge(df, provinsi, left_on='provinsi_id', right_on='provinsi_id')

df.head()
common_word=pd.Series(' '.join(df['desa_name']).split()).value_counts()[:100]

common_word
df[df['desa_name'].str.contains(" I ")]
df[df['desa_name'].str.contains(" II ")]
df[df['desa_name'].str.contains(" III ")]
common_word=pd.Series(' '.join(df['desa_name']).split()).value_counts()[:32]

common_word=common_word.drop(labels=['I','II'])

common_word.plot(kind='barh',figsize=(30,20), fontsize=20)
# Start with one review:

text = " ".join(review for review in df.desa_name)



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
meaning=['Glorious','New','Cape','River','Rock','West','East','Essence','Mountain','River Mouth'

         ,'Majestic','Field','Coral','Field (Acehnese)','South','Like','North','Great','Central','Source'

         ,'Prosperous','Hole','Water','Island','Hill','Bay','Lively (Javanese)','Village','City','Intersection']

pd.DataFrame({'name':common_word.index, 'count':common_word.values, 'meaning':meaning})
kabupaten.info()
kabupaten[kabupaten['kabupaten_name'].str.startswith('KOTA')]
kabupaten[kabupaten['kabupaten_name'].str.startswith('KABUPATEN')]
pd.Series(' '.join(kabupaten['kabupaten_name']).split()).value_counts()[:100]