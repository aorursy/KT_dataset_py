import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image



import os

print(os.listdir("../input"))



%matplotlib inline

plt.style.use('ggplot')
fig = plt.figure(figsize=(8, 8), dpi=100,facecolor='w', edgecolor='k')

train_imgs = os.listdir("../input/train_images/")

for idx, img in enumerate(np.random.choice(train_imgs, 12)):

    ax = fig.add_subplot(4, 20//5, idx+1, xticks=[], yticks=[])

    im = Image.open("../input/train_images/" + img)

    plt.imshow(im)
df = pd.read_csv('../input/train/train.csv')

df.head()
df.dtypes
print("#training points: {}".format(df.shape),end='\n\n')

print("#datatype of features: \n{}".format(df.dtypes))
sns.countplot(x='Type',data=df);

#(1 = Dog, 2 = Cat)
sns.countplot(x='Type', hue='Gender',data=df);

#(1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
sns.catplot(x="Type", hue="Vaccinated",col="Gender",

              kind="count", data=df);

#(1 = Yes, 2 = No, 3 = Not Sure)
sns.catplot(x="Type", hue="Dewormed",col="Gender",

              kind="count", data=df);

#(1 = Yes, 2 = No, 3 = Not Sure)
sns.catplot(x="Type", hue="Sterilized",col="Gender",

              kind="count", data=df);

#(1 = Yes, 2 = No, 3 = Not Sure)
sns.catplot(x="Type", hue="Health",col="Gender",

              kind="count", data=df);

#(1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)


breed_label = pd.read_csv('../input/breed_labels.csv')

print(breed_label.shape)

breed_label.head()
color_label = pd.read_csv('../input/color_labels.csv')

print(color_label.shape)

color_label
state_label = pd.read_csv('../input/state_labels.csv')

print(state_label.shape)

state_label
df['AdoptionSpeed'].value_counts()
sns.countplot(x='AdoptionSpeed', data=df);

# we see that very few pets were adopted the same day as it was listed.

# here we also see that there are about 4200 pets, which are not adopted even after 100 days of being listed.
fig, ax = plt.subplots(2,1, figsize=(8, 8))

fig.tight_layout()



sns.countplot(x='AdoptionSpeed', hue='Type', data=df, ax=ax[0]);

sns.countplot(x='Type', hue='AdoptionSpeed', data=df, ax=ax[1]);

fig.show()
import json

from pprint import pprint



with open('../input/train_metadata/0008c5398-1.json') as f:

    data = json.load(f)



pprint(data)
with open('../input/train_sentiment/0008c5398.json') as f:

    data = json.load(f)

pprint(data)
# wordcloud



from wordcloud import WordCloud



fig, ax = plt.subplots(figsize=(16,12))

plt.subplot(1, 2, 1)

text_cat = ' '.join(df.loc[df['Type'] == 2, 'Name'].fillna('').values)  # create the string of the words

wordcloud = WordCloud(max_font_size=None, background_color='white',

                      width=1200, height=1000).generate(text_cat)

plt.imshow(wordcloud)

plt.title('Top cat names')

plt.axis("off")





plt.subplot(1, 2, 2)

text_dog = ' '.join(df.loc[df['Type'] == 1, 'Name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white',

                      width=1200, height=1000).generate(text_dog)

plt.imshow(wordcloud)

plt.title('Top dog names')

plt.axis("off")



plt.show()