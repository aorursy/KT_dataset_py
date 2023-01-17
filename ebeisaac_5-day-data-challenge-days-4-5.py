# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Health_AnimalBites.csv')
data.columns
data.head()
set(data.SpeciesIDDesc)
import matplotlib.pyplot as plt

import seaborn as sns



sns.countplot(x='SpeciesIDDesc', data=data)



plt.title('Species of animals which bit humans') # Sorry, couldn't think of a better name

plt.xlabel('Species')

plt.ylabel('No. of bite cases reported')

plt.xticks(rotation=-45)



plt.show()
import numpy as np

import matplotlib.pyplot as plt



def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):

    ax = ax or plt.subplot(111, projection='polar')



    if level == 0 and len(nodes) == 1:

        label, value, subnodes = nodes[0]

        ax.bar([0], [0.5], [np.pi * 2])

        ax.text(0, 0, label, ha='center', va='center')

        sunburst(subnodes, total=value, level=level + 1, ax=ax)

    elif nodes:

        d = np.pi * 2 / total

        labels = []

        widths = []

        local_offset = offset

        for label, value, subnodes in nodes:

            labels.append(label)

            widths.append(value * d)

            sunburst(subnodes, total=total, offset=local_offset,

                     level=level + 1, ax=ax)

            local_offset += value

        values = np.cumsum([offset * d] + widths[:-1])

        heights = [1] * len(nodes)

        bottoms = np.zeros(len(nodes)) + level - 0.5

        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,

                       edgecolor='white', align='edge')

        for rect, label in zip(rects, labels):

            x = rect.get_x() + rect.get_width() / 2

            y = rect.get_y() + rect.get_height() / 2

            rotation = (90 + (360 - np.degrees(x) % 180)) % 360

            ax.text(x, y, label, rotation=rotation, ha='center', va='center') 



    if level == 0:

        ax.set_theta_direction(-1)

        ax.set_theta_zero_location('N')

        ax.set_axis_off()
otherBytes = data[(data.SpeciesIDDesc!='DOG') & (data.SpeciesIDDesc!='CAT')]



bites = [

    ('Bites', len(data), [

        ('Dog', len(data[data.SpeciesIDDesc=='DOG']),[

            ('Male', len(data[(data.SpeciesIDDesc=='DOG') & (data.GenderIDDesc=='MALE')]), []),

            ('Female', len(data[(data.SpeciesIDDesc=='DOG') & (data.GenderIDDesc=='FEMALE')]), [])

        ]),

        ('Cat', len(data[data.SpeciesIDDesc=='CAT']),[

            ('Male', len(data[(data.SpeciesIDDesc=='CAT') & (data.GenderIDDesc=='MALE')]), []),

            ('Female', len(data[(data.SpeciesIDDesc=='CAT') & (data.GenderIDDesc=='FEMALE')]), [])

        ]),

        ('Others', len(otherBytes),[])

    ])

]

sunburst(bites)

plt.title('Gender of Animals which Bit Humans')

plt.show()
import scipy.stats as stats 



data = pd.read_csv('../input/Health_AnimalBites.csv') # Yes, it was done before
data.head()
dogs = data[data.SpeciesIDDesc == 'DOG']

rctable = pd.crosstab(dogs.GenderIDDesc, dogs.WhereBittenIDDesc)

rctable
stats.chisquare(rctable.values[0], rctable.values[1])
