%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
women_degrees = pd.read_csv('../input/percent-bachelors-degrees-women-usa.csv')
women_degrees.head()
women_degrees.tail()
women_degrees.describe()
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255, 128/255, 14/255)

stem_cats = ['Psychology', 'Biology', 'Math and Statistics', 'Physical Sciences', 'Computer Science', 'Engineering']
lib_arts_cats = ['Foreign Languages', 'English', 'Communications and Journalism', 'Art and Performance', 'Social Sciences and History']
other_cats = ['Health Professions', 'Public Administration', 'Education', 'Agriculture','Business', 'Architecture']

fig = plt.figure(figsize=(15, 20))

    
for sp in range(0,18,3):

    index = int(sp / 3)
    ax = fig.add_subplot(6,3,sp+1)  
    
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[index]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[index]], c=cb_orange, label='Men', linewidth=3)
    
    for i in ax.spines.keys():
        ax.spines[i].set_visible(False)
        
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_yticks([0,100])
    ax.set_title(stem_cats[index])
    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)
    
    if index == 0:
        ax.text(2004, 87, 'Men')
        ax.text(2002, 8, 'Women')
    elif index == 5:
        ax.text(2002, 90, 'Women')
        ax.text(2004, 5, 'Men')
        ax.tick_params(labelbottom='on')
        
        
        
for sp in range(0,15,3):

    index = int(sp / 3)
    ax = fig.add_subplot(6,3,sp+2)  
    
    ax.plot(women_degrees['Year'], women_degrees[lib_arts_cats[index]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[lib_arts_cats[index]], c=cb_orange, label='Men', linewidth=3)
    
    for i in ax.spines.keys():
        ax.spines[i].set_visible(False)
        
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_yticks([0,100])
    ax.set_title(lib_arts_cats[index])
    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)
    
    if index == 0:
        ax.text(2003, 80, 'Men')
        ax.text(2002, 18, 'Women')
    elif index == 4:
        ax.tick_params(labelbottom='on')
        
        
for sp in range(0,18,3):

    index = int(sp / 3)
    ax = fig.add_subplot(6,3,sp+3)  
    
    ax.plot(women_degrees['Year'], women_degrees[other_cats[index]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[other_cats[index]], c=cb_orange, label='Men', linewidth=3)
    
    for i in ax.spines.keys():
        ax.spines[i].set_visible(False)
        
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_yticks([0,100])
    ax.set_title(other_cats[index])
    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)
    
    if index == 0:
        ax.text(2005, 93, 'Men')
        ax.text(2004, 3, 'Women')
    elif index == 5:
        ax.text(2002, 65, 'Women')
        ax.text(2003, 30, 'Men')
        ax.tick_params(labelbottom='on')

plt.title('Gender Gap in College Degrees')
plt.show();
matplotlib.get_backend()