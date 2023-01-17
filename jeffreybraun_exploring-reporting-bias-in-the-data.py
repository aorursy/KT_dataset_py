import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
from IPython.display import display, HTML

df_weapons = pd.read_csv('/kaggle/input/police-shootings-weapon-type/weapon_types.csv')
display(HTML(df_weapons.to_html()))

slices_bc_on = [0, 0, 0, 0, 0, 0, 0, 0]
slices_bc_off = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(df.shape[0]):
    weapon = df.loc[i, 'armed']
    if isinstance(weapon, float):
        weapon_id = 5
    else:
        weapon_id = int(df_weapons.loc[df_weapons['weapon'] == weapon, 'weapon_type'])
    bc = df.loc[i, 'body_camera']
    if bc == True:
        slices_bc_on[weapon_id] = slices_bc_on[weapon_id] + 1
    elif bc == False:
        slices_bc_off[weapon_id] = slices_bc_off[weapon_id] + 1

bc_on_count = df[df['body_camera'] == True].shape[0]
bc_off_count = df[df['body_camera'] == False].shape[0]    

labeling = ['Gun', 'LR High-Threat', 'SR High-Threat', 'Medium-Threat', 'Low-Threat', 'Unknown', 'Faux', 'Unarmed']




def compare_bars(slices_bc_on, slices_bc_off, labeling, ylabel, title, plot_width, plot_height):
    x = np.arange(len(labeling))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, np.around(slices_bc_on,decimals=3), width, label='Body Cam On')
    rects2 = ax.bar(x + width/2, np.around(slices_bc_off,decimals=3), width, label='Body Cam Off')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labeling)
    ax.legend()
    autolabel(rects1)
    autolabel(rects2)
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom') 
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom') 
    fig = plt.gcf()
    fig.set_size_inches(plot_width,plot_height)
    plt.show()
    
print("There are %i instances where the body camera was reported to be on" %(bc_on_count))
print("There are %i instances where the body camera was reported to be off" %(bc_off_count))

slices_bc_on = np.array(slices_bc_on) * (1/sum(slices_bc_on))
slices_bc_off = np.array(slices_bc_off) * (1/sum(slices_bc_off))

ylabel = 'Percent Reported Weapon Type'
title = 'Reported Weapon Type Grouped by Body Camera Status'

compare_bars(slices_bc_on, slices_bc_off, labeling, ylabel, title, 15, 10)
slices_bc_on_black = [0, 0, 0, 0, 0, 0, 0, 0]
slices_bc_off_black = [0, 0, 0, 0, 0, 0, 0, 0]

df_black = df[df.race == 'B']
df_black = df_black.reset_index()

for i in range(df_black.shape[0]):
    weapon = df_black.loc[i, 'armed']
    if isinstance(weapon, float):
        weapon_id = 5
    else:
        weapon_id = int(df_weapons.loc[df_weapons['weapon'] == weapon, 'weapon_type'])
    bc = df_black.loc[i, 'body_camera']
    if bc == True:
        slices_bc_on_black[weapon_id] = slices_bc_on_black[weapon_id] + 1
    elif bc == False:
        slices_bc_off_black[weapon_id] = slices_bc_off_black[weapon_id] + 1

bc_on_count_black = df_black[df_black['body_camera'] == True].shape[0]
bc_off_count_black = df_black[df_black['body_camera'] == False].shape[0]  


slices_bc_on_white = [0, 0, 0, 0, 0, 0, 0, 0]
slices_bc_off_white = [0, 0, 0, 0, 0, 0, 0, 0]

df_white = df[df.race == 'W']
df_white = df_white.reset_index()

for i in range(df_white.shape[0]):
    weapon = df_white.loc[i, 'armed']
    if isinstance(weapon, float):
        weapon_id = 5
    else:
        weapon_id = int(df_weapons.loc[df_weapons['weapon'] == weapon, 'weapon_type'])
    bc = df_white.loc[i, 'body_camera']
    if bc == True:
        slices_bc_on_white[weapon_id] = slices_bc_on_white[weapon_id] + 1
    elif bc == False:
        slices_bc_off_white[weapon_id] = slices_bc_off_white[weapon_id] + 1

bc_on_count_white = df_white[df_white['body_camera'] == True].shape[0]
bc_off_count_white = df_white[df_white['body_camera'] == False].shape[0]  

print("There are %i instances where the body camera was reported to be on for black individuals" %(bc_on_count_black))
print("There are %i instances where the body camera was reported to be off for black individuals" %(bc_off_count_black))
print("Probability body camera was on for black individuals: %.0f%%" %(100 * bc_on_count_black / (bc_on_count_black + bc_off_count_black) ))

slices_bc_on_black = np.array(slices_bc_on_black) * (1/sum(slices_bc_on_black))
slices_bc_off_black = np.array(slices_bc_off_black) * (1/sum(slices_bc_off_black))

ylabel = 'Percent Reported Weapon Type'
title = 'Reported Weapon Type Grouped by Body Camera Status for Black Individuals'

compare_bars(slices_bc_on_black, slices_bc_off_black, labeling, ylabel, title, 15, 10)

print("There are %i instances where the body camera was reported to be on for white individuals" %(bc_on_count_white))
print("There are %i instances where the body camera was reported to be off for white individuals" %(bc_off_count_white))
print("Probability body camera was on for white individuals: %.0f%%" %(100 * bc_on_count_white / (bc_on_count_white + bc_off_count_white) ))

slices_bc_on_white = np.array(slices_bc_on_white) * (1/sum(slices_bc_on_white))
slices_bc_off_white = np.array(slices_bc_off_white) * (1/sum(slices_bc_off_white))

ylabel = 'Percent Reported Weapon Type'
title = 'Reported Weapon Type Grouped by Body Camera Status for White Individuals'

compare_bars(slices_bc_on_white, slices_bc_off_white, labeling, ylabel, title, 15, 10)
