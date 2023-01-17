# import useful library

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

color = sns.color_palette()

# read the data

data = pd.read_csv('../input/No-show-Issue-Comma-300k.csv', parse_dates=[2,3])
data.head()
data_size = data.shape

print("Our dataset has {} rows and {} columns".format(data_size[0], data_size[1]))

print()

print(data.info())
for i in ['Diabetes', 'Alcoolism','DayOfTheWeek', 'Gender','HiperTension','Handcap', 'Smokes', 'Scholarship', 'Tuberculosis', 'Sms_Reminder']:

    print(i)

    print(data[i].value_counts())

    print()
# replace all the value greater than 1 by one in variables: Handcap and Sms_reminder

for item in ['Handcap', 'Sms_Reminder']:

    data[item] = data[item].apply(lambda x: 1 if x > 1 else x)

    

for i in ['Handcap', 'Sms_Reminder']:

    print(i)

    print(data[i].value_counts())

    print()
# occurrences of show vs no show

status = data['Status'].value_counts()

plt.figure(figsize=(8,4))

sns.barplot(status.index, status.values, alpha=0.8, color=color[5])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Status', fontsize=12)

plt.title('Show-Up VS No-Show')

plt.show()
# let's visualize the proportion of show vs no-show

status_proportion = (status/sum(status.values))*100

pd_show = pd.DataFrame({'Show-Up': [status_proportion['Show-Up']], 'No-Show': [status_proportion['No-Show']]})

pd_show.plot(kind='bar', stacked=True)

plt.xlabel('Status')

plt.title('Proportion of show vs no-show')

plt.show()
conditions = data[['Diabetes', 'Alcoolism', 'HiperTension', 'Handcap', 'Smokes', 'Tuberculosis', 'Status' ]]
conditions = pd.melt(conditions, 

               id_vars=['Status'], 

               var_name='condition', 

               value_name='result')



conditions_1 = conditions[conditions.result == 1]

conditions_1.result = conditions_1.result.astype(int)
conditions_1.head()
 

conditions_1.result.value_counts()
plt.figure(figsize=(8,6))

sns.countplot(y='Status', hue='condition', data=conditions_1)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Status', fontsize=12)

plt.show()
grouped_conditions = conditions_1.groupby(['condition', 'Status'])['result'].sum().unstack().reset_index()

grouped_conditions
# proportion within a conditions

# Create a figure with a single subplot

f, ax = plt.subplots(1, figsize=(10,5))



# Set bar width at 1

bar_width = 1

bar_l = [i for i in range(len(grouped_conditions['No-Show']))]

tick_pos = [i+(bar_width/2) for i in bar_l]

totals = [i+j for i,j in zip(grouped_conditions['No-Show'], grouped_conditions['Show-Up'])]

pre_rel = [i / j * 100 for  i,j in zip(grouped_conditions['No-Show'], totals)]

right_rel = [i / j * 100 for  i,j in zip(grouped_conditions['Show-Up'], totals)]



# Create a bar chart in position 1

ax.bar(bar_l,

       pre_rel,

       label='No-Show',

       alpha=0.9,

       color='#019600',

       width=bar_width,

       edgecolor='white'

       )



# Create a bar chart in position 2

ax.bar(bar_l,

       right_rel,

       bottom=pre_rel,

       label='Show Up',

       alpha=0.9,

       color='#3C5F5A',

       width=bar_width,

       edgecolor='white'

       )



# Set the ticks to be conditons

plt.xticks(tick_pos, grouped_conditions['condition'])

ax.set_ylabel("Proportion")

ax.set_xlabel("")



# Let the borders of the graphic

plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

plt.xlabel('Condition')

plt.legend()

# rotate axis labels

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')



# shot plot

plt.show()