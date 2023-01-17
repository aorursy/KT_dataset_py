# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

women_degrees = pd.read_csv('../input/percent-bachelors-degrees-women-usa.csv')

women_degrees.head(3)

# Any results you write to the current directory are saved as output.
women_degrees.describe()
import pandas as pd

import matplotlib.pyplot as plt



major_cats = ['Biology', 'Computer Science', 'Engineering', 'Math and Statistics']

fig = plt.figure(figsize=(12, 12))



for sp in range(0,4):

    ax = fig.add_subplot(2,2,sp+1)

    ax.plot(women_degrees['Year'], women_degrees[major_cats[sp]], c='blue', label='Women')

    ax.plot(women_degrees['Year'], 100-women_degrees[major_cats[sp]], c='green', label='Men')

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.set_xlim(1968, 2011)

    ax.set_ylim(0,100)

    ax.set_title(major_cats[sp])

    ax.tick_params(bottom="off", top="off", left="off", right="off")





# Calling pyplot.legend() here will add the legend to the last subplot that was created.

plt.legend(loc='upper right')

plt.show()