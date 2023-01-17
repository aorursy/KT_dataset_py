from IPython.display import HTML

HTML('<center><iframe width="720" height="540" src="https://www.youtube.com/embed/4-CjXCKwj2I" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import pandas as pd

from pandas.io.json import json_normalize

import PIL                      #read images

import matplotlib.pyplot as plt #show images

import seaborn as sns

sns.set(style="whitegrid")
cardsImgs = pd.read_json('/kaggle/input/tarot-json/tarot-images.json', orient='records')

cardsImgs = json_normalize(cardsImgs['cards'])

cardsImgs
cardsImgs.info()
ax = sns.countplot(x="arcana", hue="suit", data=cardsImgs)

ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()+0.5))

ax
import datetime

print("Your reading: Past, Present, Future on", datetime.datetime.now().date())



cards3 = cardsImgs.sample(n = 3).reset_index(drop=True)



plt.figure(figsize=(10,10))

c = 0

for i in cards3['img']:

    img = PIL.Image.open(f'/kaggle/input/tarot-json/cards/{i}')

    plt.subplot(1,3,c+1)

    plt.imshow(img)

    plt.axis('off')

    c+=1



plt.show()
t = ['PAST', 'PRESENT', 'FUTURE']



for index, row in cards3.iterrows():

    print('\x1b[31m{}\x1B[37m: {} \n   {}, {}'.format(t[index], row['name'], row['arcana'], row['suit']))

    for r in row['fortune_telling']:

        print("\x1B[37m\t --->\x1b[31m", r.upper())

    for r in row['Questions to Ask']:

        print("\x1B[37m\t    ?", r)

    print(10*'-')