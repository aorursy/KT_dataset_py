# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt



women_degrees = pd.read_csv('/kaggle/input/bachelorsdegreewomenusa/percent-bachelors-degrees-women-usa.csv')

major_cats = ['Biology', 'Computer Science', 'Engineering', 'Math and Statistics']
%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt



cb_dark_blue = (0/255,107/255,164/255)

cb_orange = (255/255, 128/255, 14/255)

stem_cats = ['Engineering', 'Computer Science', 'Psychology', 'Biology', 'Physical Sciences', 'Math and Statistics']



fig = plt.figure(figsize=(18, 3))



for sp in range(0,6):

    ax = fig.add_subplot(1,6,sp+1)

    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)

    ax.spines["right"].set_visible(False)    

    ax.spines["left"].set_visible(False)

    ax.spines["top"].set_visible(False)    

    ax.spines["bottom"].set_visible(False)

    ax.set_xlim(1968, 2011)

    ax.set_ylim(0,100)

    ax.set_title(stem_cats[sp])

    ax.tick_params(bottom="off", top="off", left="off", right="off")

    

    if sp == 0:

        ax.text(2005, 87, 'Men')

        ax.text(2002, 8, 'Women')

    elif sp == 5:

        ax.text(2005, 62, 'Men')

        ax.text(2001, 35, 'Women')

plt.show()
%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt



cb_dark_blue = (0/255,107/255,164/255)

cb_orange = (255/255, 128/255, 14/255)



stem_cats = ['Engineering', 'Computer Science', 'Psychology', 'Biology', 'Physical Sciences', 'Math and Statistics']

lib_arts_cats = ['Foreign Languages', 'English', 'Communications and Journalism', 'Art and Performance', 'Social Sciences and History']

other_cats = ['Health Professions', 'Public Administration', 'Education', 'Agriculture','Business', 'Architecture']



fig = plt.figure(figsize=(16, 20))



for sp in range(0,18,3):

    ax = fig.add_subplot(6,3,sp+1)

    index = int(sp/3)

    ax.plot(women_degrees['Year'], women_degrees[stem_cats[index]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[index]], c=cb_orange, label='Men', linewidth=3)

    for i,j in ax.spines.items():

        ax.spines[i].set_visible(False)

    ax.set_ylim(0,100)

    ax.set_title(stem_cats[index])

    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False)

    ax.set_yticks([0,100])

    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)

    

    if sp == 0:

        ax.text(2005, 87, 'Men')

        ax.text(2002, 8, 'Women')

    elif sp == 15:

        ax.text(2005, 62, 'Men')

        ax.text(2001, 35, 'Women')

        ax.tick_params(labelbottom=True)

 

for sp in range(1,15,3):

    ax = fig.add_subplot(6,3,sp+1)

    index = int(sp/3)

    ax.plot(women_degrees['Year'], women_degrees[lib_arts_cats[index]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[lib_arts_cats[index]], c=cb_orange, label='Men', linewidth=3)

    for i,j in ax.spines.items():

        ax.spines[i].set_visible(False)

    ax.set_ylim(0,100)

    ax.set_title(lib_arts_cats[index])

    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False)

    ax.set_yticks([0,100])

    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)



    

    if sp == 1:

        ax.text(2005, 75, 'Men')

        ax.text(2002, 20, 'Women')

    elif sp == 13:

        ax.tick_params(labelbottom=True)



        

for sp in range(2,18,3):

    ax = fig.add_subplot(6,3,sp+1)

    index = int(sp/3)

    ax.plot(women_degrees['Year'], women_degrees[other_cats[index]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[other_cats[index]], c=cb_orange, label='Men', linewidth=3)

    for i,j in ax.spines.items():

        ax.spines[i].set_visible(False)

    ax.set_ylim(0,100)

    ax.set_title(other_cats[index])

    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False)

    ax.set_yticks([0,100])

    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)



    if sp == 0:

        ax.text(2005, 87, 'Men')

        ax.text(2002, 8, 'Women')

    elif sp == 17:

        ax.text(2005, 62, 'Men')

        ax.text(2001, 35, 'Women')

        ax.tick_params(labelbottom=True)



plt.savefig('gender_degrees.png')

plt.show()