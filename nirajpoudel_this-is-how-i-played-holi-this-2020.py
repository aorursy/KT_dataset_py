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
import matplotlib.pyplot as plt



labels = ['Avoiding people', 'Actively playing Holi', 'Cleaning Myself After']

sizes = [60, 20, 20]

explode = (0,0,0.1)

colors = ['yellowgreen', 'gold', 'lightskyblue']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
import matplotlib.pyplot as plt



labels = ['Before playing when colurs are neatly laid out', 'While playing holi', 'After a generous helping of bhang']

sizes = [40, 30, 42]

explode = (0,0,0)

colors = ['blue', 'yellow', 'maroon']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ['Laughs at ramdom things', 'Im still Laughing', 'Why am i still laughing','Why am i still laughing']

sizes = [25,25,25,25]

explode = (0,0,0,0)

colors = ['orange', 'black', 'skyblue','purple']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ['I am tired of all the physical activity', 'I need to clean up', 'i miss my phone']

sizes = [15,15,70]

explode = (0,0,0)

colors = ['pink', 'Beige', 'skyblue']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ['Water ballon did not hit the targeted person.', 'Ballon brust out during filling','Ballon or rotten egg didnot break after hitting person']

sizes = [40,20,40]

explode = (0,0,0)

colors = ['Coral', 'gray', 'brown']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ['Who hit water ballon', 'How to remove red colors form nails.']

sizes = [25,75]

explode = (0,0)

colors = ['gold', 'red']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ['Why are you wasting so much water', 'Bath properly, Everything must go.']

sizes = [50,50]

explode = (0,0)

colors = ['silver', 'blue']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ['Location is convinient', 'Where my male friends are going','My crush will be there but not friends']

sizes = [10,30,60]

explode = (0,0,0)

colors = ['yellow', 'blue','pink']

patches, texts = plt.pie(sizes, explode = explode, colors=colors, shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()