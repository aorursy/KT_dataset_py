%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
women_degrees = pd.read_csv('/kaggle/input/bachelorsdegreewomenusa/percent-bachelors-degrees-women-usa.csv')

women_degrees.head(7)
# global colors for graphs



cb_dark_blue = (0/255,107/255,164/255)

cb_orange = (255/255, 128/255, 14/255)
stem_cats = ['Engineering', 'Computer Science', 'Psychology', 'Biology', 'Physical Sciences', 'Math and Statistics']



fig1 = plt.figure(figsize=(25, 4))



for sp in range(0,6):

    ax = fig1.add_subplot(1,6,sp+1)

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
stem_cats = ['Psychology', 'Biology', 'Math and Statistics', 'Physical Sciences', 'Computer Science', 'Engineering']

lib_arts_cats = ['Foreign Languages', 'English', 'Communications and Journalism', 'Art and Performance', 'Social Sciences and History']

other_cats = ['Health Professions', 'Public Administration', 'Education', 'Agriculture','Business', 'Architecture']



fig2 = plt.figure(figsize=(25, 25))



## Generate first column of line charts. STEM degrees.

for sp in range(0,18,3):

    cat_index = int(sp/3)

    ax = fig2.add_subplot(6,3,sp+1)

    ax.plot(women_degrees['Year'], women_degrees[stem_cats[cat_index]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[cat_index]], c=cb_orange, label='Men', linewidth=3)

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.set_xlim(1968, 2011)

    ax.set_ylim(0,100)

    ax.set_title(stem_cats[cat_index])

    

    ax.set_yticks([0,25,50,75,100])

    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')

    for key,spine in ax.spines.items():

        spine.set_visible(False)

        

    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)

    

    if cat_index == 0:

        ax.text(2003, 85, 'Women')

        ax.text(2005, 10, 'Men')

    elif cat_index == 5:

        ax.text(2005, 87, 'Men')

        ax.text(2003, 7, 'Women')

        ax.tick_params(labelbottom='on')



## Generate second column of line charts. Liberal arts degrees.

for sp in range(1,16,3):

    cat_index = int((sp-1)/3)

    ax = fig2.add_subplot(6,3,sp+1)

    ax.plot(women_degrees['Year'], women_degrees[lib_arts_cats[cat_index]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[lib_arts_cats[cat_index]], c=cb_orange, label='Men', linewidth=3)

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.set_xlim(1968, 2011)

    ax.set_ylim(0,100)

    ax.set_title(lib_arts_cats[cat_index])

    

    ax.set_yticks([])

    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')

    for key,spine in ax.spines.items():

        spine.set_visible(False)

        

    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)

    

    if cat_index == 0:

        ax.text(2003, 78, 'Women')

        ax.text(2005, 18, 'Men')

        

    elif cat_index == 4:

        ax.tick_params(labelbottom='on')



## Generate third column of line charts. Other degrees.

for sp in range(2,20,3):

    cat_index = int((sp-2)/3)

    ax = fig2.add_subplot(6,3,sp+1)

    ax.plot(women_degrees['Year'], women_degrees[other_cats[cat_index]], c=cb_dark_blue, label='Women', linewidth=3)

    ax.plot(women_degrees['Year'], 100-women_degrees[other_cats[cat_index]], c=cb_orange, label='Men', linewidth=3)

    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='on')



    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.set_xlim(1968, 2011)

    ax.set_ylim(0,100)

    ax.set_title(other_cats[cat_index])

    

    ax.set_yticks([])

    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')

    for key,spine in ax.spines.items():

        spine.set_visible(False)

        

    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)

    

    if cat_index == 0:

        ax.text(2003, 90, 'Women')

        ax.text(2005, 5, 'Men')

    elif cat_index == 5:

        ax.text(2005, 62, 'Men')

        ax.text(2003, 30, 'Women')

        ax.tick_params(labelbottom='on')

        

plt.savefig('/kaggle/input/biology_degrees.png')        

plt.show()
matplotlib.get_backend()