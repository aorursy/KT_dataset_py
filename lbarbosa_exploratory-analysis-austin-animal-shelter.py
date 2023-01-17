import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
% matplotlib inline
from math import pi
from palettable.colorbrewer.qualitative import Pastel1_7
pd.options.mode.chained_assignment = None
df = pd.read_csv('../input/aac_intakes_outcomes.csv')
print(df.shape)
df.head()
print(df.max())
df.describe()
# Plot bar plot (animal type, age)
plt.figure(figsize=(8,5))
sns.barplot(x=df['animal_type'], y=df['age_upon_intake_(years)'])
plt.title('Animal Type Average Age Upon Intake', fontsize=20);
plt.xlabel('Animal Type', fontsize=18)
plt.ylabel('Age (Years)', fontsize=18);
plt.figure(figsize=(10, 8))
sns.boxplot(x='animal_type', y='age_upon_intake_(years)', data=df, orient='v')
plt.title('Age Distributions for Diferent Animal Type', fontsize=20);
plt.xlabel('Animal Type', fontsize=18)
plt.ylabel('Age (Years)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18);
# Plot bar plot (intake_condition, age)
plt.figure(figsize=(10,5))
sns.barplot(x=df['intake_condition'], y=df['age_upon_intake_(years)'])
plt.title('Average Age by Intake Condition', fontsize=20);
plt.xlabel('Intake Condition', fontsize=18)
plt.ylabel('Age (Years)', fontsize=18);
plt.figure(figsize=(12, 8))
sns.boxplot(x='intake_condition', y='age_upon_intake_(years)', data=df, orient='v')
plt.title('Age Distributions for Diferent Conditions', fontsize=20)
plt.xlabel('Intake Condition', fontsize=18)
plt.ylabel('Age (Years)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18);
# grouping dataset by intake condition, counting, and sorting.
by_condition = df.groupby(['intake_condition'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
# grouping conditions with low count into 'Other' category for illustration purposes
for i, row in by_condition.iterrows():
    if (by_condition.intake_condition[i]=='Aged') or (by_condition.intake_condition[i]=='Feral') or (by_condition.intake_condition[i] == 'Pregnant'):
        by_condition.intake_condition[i]='Other'       
by_condition = by_condition.groupby(['intake_condition'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
# Donut Plot
plt.figure(figsize=(8,8))
names = by_condition['intake_condition']
size_of_groups = by_condition['count']
plt.pie(size_of_groups, labels=names, colors=['skyblue','green','red','blue','orange'], 
        autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)  # Create a pieplot
my_circle = plt.Circle( (0,0), 0.7, color='white') # add a circle at the center
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Percentage of Intake Condition - ALL', fontsize=20);
plt.show()
# pd.Categorical to order the weekdays starting from Monday
df['intake_weekday'] = pd.Categorical(df['intake_weekday'], ordered=True,
                                      categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.figure(figsize=(12, 8))
sns.countplot(x='intake_weekday', data=df, orient='v')
plt.title('Count of Intakes by Day of Week - All years', fontsize=18)
plt.xlabel('Weekday', fontsize=18)
plt.ylabel('Count of Intakes', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16);
plt.figure(figsize=(12, 8))
sns.countplot(x='intake_month', data=df, orient='v')
plt.title('Count of Intakes by Month of Year - All years', fontsize=18)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Count of Intakes', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16);
plt.figure(figsize=(10, 8))
sns.countplot(x='intake_type', data=df, orient='v')
plt.title('Count of Intakes by Type - All', fontsize=18)
plt.xlabel('Intake Type', fontsize=16)
plt.ylabel('Count of Intakes', fontsize=16)
plt.xticks(fontsize=16, rotation='vertical')
plt.yticks(fontsize=16);
plt.figure(figsize=(12, 8))
sns.countplot(x='outcome_type', data=df, orient='v')
plt.title('Count of Outcomes by Type - All', fontsize=18)
plt.xlabel('Outcome Type', fontsize=16)
plt.ylabel('Count of Outcomes', fontsize=16)
plt.xticks(fontsize=12, rotation='vertical')
plt.yticks(fontsize=12);
# Creating columns for day of year
df['intake_month_day'] = pd.DatetimeIndex(df['intake_datetime']).strftime('%m-%d')
df['outcome_month_day'] = pd.DatetimeIndex(df['outcome_datetime']).strftime('%m-%d')
by_intake_md = df.groupby('intake_month_day')[['count']].sum()
by_outcome_md = df.groupby('outcome_month_day')[['count']].sum()
by_intake_md['md'] = by_intake_md.index
by_intake_md = by_intake_md.rename(columns={'count':'intake_count'})
by_outcome_md['md'] = by_outcome_md.index
by_outcome_md = by_outcome_md.rename(columns={'count':'outcome_count'})
df_md = pd.merge(by_intake_md, by_outcome_md, on='md')
df_md.sort_values('outcome_count').tail() # August 19th is the day with most outcomes.
# Creating columns for day of month
df['outcome_day'] = pd.DatetimeIndex(df['outcome_datetime']).day
# Manipulating data to feed pivot table to then feed to seaborn heatmap
by_outcome_md = df.groupby(['outcome_day','outcome_month'])[['count']].sum()
by_outcome_md = by_outcome_md.reset_index(level=[0,1]) # to go from multiIndex to singleIndex
pivoted_table = by_outcome_md.pivot(index='outcome_month', columns='outcome_day', values='count')
pivoted_table.fillna(0, inplace=True)
plt.figure(figsize=(18, 8))
sns.heatmap(pivoted_table, cmap='Blues', annot=True, fmt='g', annot_kws={'size': 8})
plt.title('Count of Outcomes by Month and Day of Year - All Years', fontsize=18);
plt.xlabel('Outcome Day', fontsize=16)
plt.ylabel('Outcome Month', fontsize=16);
# grouping dataset by anymal type and counting
by_at = df.groupby(['animal_type'])[['count']].sum()
# Donut Plot
plt.figure(figsize=(8,8))
names = by_at.index
size_of_groups = by_at['count']
plt.pie(size_of_groups, labels=names, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'], 
        autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)  # Create a pieplot
my_circle = plt.Circle( (0,0), 0.7, color='white') # add a circle at the center
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Anymal Type Distirbution- ALL', fontsize=20);
plt.show()
Other = df[df['animal_type']=='Other']
# preping data for 1st donut plot - left
Other1 = Other.groupby(['breed'])[['count']].sum().reset_index().sort_values(by='count', ascending=False).head(12)
names1 = Other1['breed']
size_of_groups1 = Other1['count']
#preping data for 2nd donut plot - right
Other2 = Other.groupby(['outcome_type'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
names2 = Other2['outcome_type']
size_of_groups2 = Other2['count']
fig = plt.figure(figsize=(16,8))
plt.subplot(121)
ax = plt.pie(size_of_groups1, labels=names1, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)  # Create a pieplot
my_circle = plt.Circle( (0,0), 0.7, color='white') # add a circle at the center
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Top 12 - Other Animal Type', fontsize=20);
plt.subplot(122)
ax = plt.pie(size_of_groups2, labels=names2, autopct='%1.1f%%', colors=Pastel1_7.hex_colors, pctdistance=1.1, labeldistance=1.2)
my_circle = plt.Circle( (0,0), 0.7, color='white') # add a circle at the center
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Outcome - Other Animal Type', fontsize=20);
fig.set_facecolor('w')
by_ao = df.groupby(['animal_type','outcome_type'])[['count']].sum()
by_ao = by_ao.reset_index(level=[0,1]) # to go from multiIndex to singleIndex
by_ao.animal_type.unique()
# Values of each group
Adoption = by_ao[by_ao['outcome_type']=='Adoption']['count']
A=Adoption.values[1:4]
Died = by_ao[by_ao['outcome_type']=='Died']['count']
D=Died.values[1:4]
Disposal = by_ao[by_ao['outcome_type']=='Disposal']['count']
Di=Disposal.values[1:4]
Euthanasia = by_ao[by_ao['outcome_type']=='Euthanasia']['count']
E=Euthanasia.values[1:4]
#Relocate = by_ao[by_ao['outcome_type']=='Relocate']['count']
#R=Relocate.values[1:4]
Return = by_ao[by_ao['outcome_type']=='Return to Owner']['count']
Re=Return.values[1:4]
Transfer = by_ao[by_ao['outcome_type']=='Transfer']['count']
T=Transfer.values[1:4]
legend=['Adoption','Died','Disposal','Euthanasia','Return','Transfer']
rc('font', weight='bold') # y-axis in bold
r = [1,2,3] # The position of the bars on the x-axis
names = ['Cat', 'Dog', 'Other'] # Names of group and bar width
barWidth = 1
# Create stacked bar plot
plt.figure(figsize=(8,8))
plt.bar(r, A, color='green', edgecolor='white', width=barWidth)
plt.bar(r, D, bottom=A, color='purple', edgecolor='white', width=barWidth)
plt.bar(r, Di, bottom=A+D, color='orange', edgecolor='white', width=barWidth)
plt.bar(r, E, bottom=A+D+Di, color='red', edgecolor='white', width=barWidth)
plt.bar(r, Re, bottom=A+D+Di+E, color='blue', edgecolor='white', width=barWidth)
plt.bar(r, T, bottom=A+D+Di+E+Re, color='pink', edgecolor='white', width=barWidth)
plt.title('Count of Outcomes by Animal Type - Birds Excluded', fontsize=18);
plt.xticks(r, names, fontweight='bold')
plt.xlabel('Animal Type', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend(legend,loc='upper right');
plt.show()
adoption = df[df['outcome_type']=='Adoption']
to_owner = df[df['outcome_type']=='Return to Owner']
possitive = adoption.append(to_owner, ignore_index=True) # df of all animals with an outcome that indicates they are "preferred"
# Plot bar plot (animal type, time in shelter)
plt.figure(figsize=(8,5))
sns.barplot(x=possitive['animal_type'], y=possitive['time_in_shelter_days']);
plt.title('Avg # Days by Animal Type - Adoption or Return to Owner', fontsize=18)
plt.xlabel('Animal Type', fontsize=16)
plt.ylabel('Avg Time in Shelter (Days)', fontsize=16);
stray = df[df['intake_type']=='Stray']
surrender = df[df['intake_type']=='Surrender']
negative = stray.append(surrender, ignore_index=True) # df of all animals with an intake type that indicates they aren't wanted
# Plot count plot (animal type, intake type)
plt.figure(figsize=(8, 5))
sns.countplot(x='animal_type', data=negative, orient='v')
plt.title('Count of Intake by Animal Type - Stray or Surrender', fontsize=18)
plt.xlabel('Animal Type', fontsize=16)
plt.ylabel('Count of Intakes', fontsize=16);
plt.xticks(fontsize=12, rotation='vertical')
plt.yticks(fontsize=12);
# getting Avg time in shelter to possitive outcome for dogs and manipulating so it can me merged to other df
avg_time = possitive.groupby(['animal_type'])[['time_in_shelter_days']].mean().reset_index().sort_values(by='time_in_shelter_days',ascending=False)
avg_time_dog = avg_time[avg_time['animal_type']=='Dog']
avg_time_dog['animal_type']='Avg # Days in Shelter to Possitive Outcome'
avg_time_dog = avg_time_dog.rename(columns = {'animal_type':'criteria','time_in_shelter_days':'value'})
# getting % metrics for dogs from df
dogs = df[df['animal_type']=='Dog']
dogs_by_ot = dogs.groupby(['outcome_type'])[['count']].sum().reset_index().sort_values(by='count',ascending=False)
dogs_count = dogs.shape[0]
dogs_by_ot['value'] = 100*dogs_by_ot['count']/dogs_count # this is percent
dogs_by_ot = dogs_by_ot.rename(columns={'outcome_type':'criteria'})
top2_dbot = dogs_by_ot.head(2) # adoption and return to owner happen to be the top 2
dogs_by_it = dogs.groupby(['intake_type'])[['count']].sum().reset_index().sort_values(by='count',ascending=False)
dogs_by_it['value'] = 100*dogs_by_it['count']/dogs_count # this is percent
dogs_by_it = dogs_by_it.rename(columns={'intake_type':'criteria'})
top2_dbit = dogs_by_it.head(2) # stray and owner surrender happen to be the top 2
# combining top2_dbit & top2_dbot into one df
dog_spi_0 = top2_dbot.append(top2_dbit, ignore_index=True)
dog_spi_0 = dog_spi_0.drop(['count'],axis=1)
dog_spi_0['criteria'] = '% ' + dog_spi_0['criteria']
# combining dog_spi_0 with avg_time_dog
dog_spi = dog_spi_0.append(avg_time_dog, ignore_index=True)
dog_spi
# grouping dataset by anymal type and counting
avg_time_cat = avg_time[avg_time['animal_type']=='Cat']
avg_time_cat['animal_type'] = 'Avg # Days in Shelter to Possitive Outcome'
avg_time_cat = avg_time_cat.rename(columns = {'animal_type':'criteria', 'time_in_shelter_days':'value'})
# getting % metrics for dogs from df
cats = df[df['animal_type']=='Cat']
cats_by_ot = cats.groupby(['outcome_type'])[['count']].sum().reset_index().sort_values(by='count',ascending=False)
cats_count = cats.shape[0]
cats_by_ot['value'] = 100*cats_by_ot['count']/cats_count # this is percent
cats_by_ot = cats_by_ot.rename(columns = {'outcome_type':'criteria'})
cat_adopt = cats_by_ot[1:2]
cat_return = cats_by_ot[3:4]
top2_dbot = cat_adopt.append(cat_return, ignore_index=True)
cats_by_it = cats.groupby(['intake_type'])[['count']].sum().reset_index().sort_values(by='count',ascending=False)
cats_by_it['value'] = 100*cats_by_it['count']/cats_count # this is percent
cats_by_it = cats_by_it.rename(columns = {'intake_type':'criteria'})
top2_dbit = cats_by_it.head(2) # stray and owner surrender happen to be the top 2
# combining top2_dbit & top2_dbot into one df
cat_spi_0 = top2_dbot.append(top2_dbit, ignore_index=True)
cat_spi_0 = cat_spi_0.drop(['count'], axis=1)
cat_spi_0['criteria'] = '% '+ cat_spi_0['criteria']
# combining dog_spi_0 with avg_time_dog
cat_spi = cat_spi_0.append(avg_time_cat, ignore_index=True)
cat_spi
categories = dog_spi['criteria'].tolist()
N = len(categories)
dog_values = dog_spi['value'].values.tolist()
dog_values += dog_values[:1] # need to repeat first value to close loop
cat_values = cat_spi['value'].values.tolist()
cat_values += cat_values[:1] # need to repeat first value to close loop
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(8,8))
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
plt.title('Dogs vs. Cats', fontsize=18)
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='blue', size=10)
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30,40,50,60,70], ['10','20','30','40','50','60','70'], color='grey', size=8)
plt.ylim(0,80)
# Plot data
ax.plot(angles, dog_values, linewidth=1, linestyle='solid', label='Dogs')
ax.fill(angles, dog_values, 'b', alpha=0.1)
ax.plot(angles, cat_values, linewidth=1, linestyle='solid', label='Cats')
ax.fill(angles, cat_values, 'r', alpha=0.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=12);
breed=dogs.groupby(['breed'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
breed_40=breed.head(40)
# Bar plot
fig = plt.figure(figsize=(8,10))
sns.barplot('count','breed',
                 data=breed_40, palette='husl', linewidth=0.7, edgecolor='k')  # hls/husl chooses the palette based on evenly spaced colors taken out from a circular color space. husl controls for color intensity.
plt.ylabel('Dog Breed', fontsize=14)
plt.xlabel('Count', fontsize=14)
plt.title('TOP 40 DOG BREEDS', fontsize=15)
for i,j in enumerate(breed_40['breed']):
    ax.text(.9,i,j,weight='bold',fontsize=14)
plt.grid(True,alpha= .3);
# Creating a column to indicate if breed is pit bull or pit bull mix (Y) or other (N)
dogs['Pit Bull'] = np.where(dogs['breed'].str.contains('Pit Bull'), 'Y', 'N')
adoption = dogs[dogs['outcome_type']=='Adoption']
avg_time_adopt = adoption.groupby(['Pit Bull'])[['time_in_shelter_days']].mean().reset_index().sort_values(by='time_in_shelter_days', ascending=False)
avg_time_adopt_pit = avg_time_adopt[avg_time_adopt['Pit Bull']=='Y']
avg_time_adopt_pit['Pit Bull'] = 'Avg # Days to Adoption'
avg_time_adopt_pit = avg_time_adopt_pit.rename(columns = {'Pit Bull':'criteria', 'time_in_shelter_days':'value'})
to_owner = dogs[dogs['outcome_type']=='Return to Owner']
avg_time_return = to_owner.groupby(['Pit Bull'])[['time_in_shelter_days']].mean().reset_index().sort_values(by='time_in_shelter_days', ascending=False)
avg_time_return_pit = avg_time_return[avg_time_return['Pit Bull']=='Y']
avg_time_return_pit['Pit Bull'] = 'Avg # Days to Return to Owner'
avg_time_return_pit = avg_time_return_pit.rename(columns = {'Pit Bull':'criteria', 'time_in_shelter_days':'value'})
# combining avg time adoption and return
avg_time_pit = avg_time_adopt_pit.append(avg_time_return_pit, ignore_index=True)
# getting % metrics for pits from dogs
pits = dogs[dogs['Pit Bull']=='Y']
pits_by_ot = pits.groupby(['outcome_type'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
pits_count = pits.shape[0]
pits_by_ot['value'] = 100*pits_by_ot['count']/pits_count # this is percent
pits_by_ot = pits_by_ot.rename(columns = {'outcome_type':'criteria'})
top2_dbot = pits_by_ot.head(2) # adoption and return to owner happen to be the top 2
pits_by_it = pits.groupby(['intake_type'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
pits_by_it['value'] = 100*pits_by_it['count']/pits_count # this is percent
pits_by_it = pits_by_it.rename(columns = {'intake_type':'criteria'})
top2_dbit = pits_by_it.head(2) # stray and owner surrender happen to be the top 2
# combining top2_dbit & top2_dbot into one df
pit_spi_0 = top2_dbot.append(top2_dbit, ignore_index=True)
pit_spi_0 = pit_spi_0.drop(['count'],axis=1)
pit_spi_0['criteria'] = '% '+ pit_spi_0['criteria']
# combining dog_spi_0 with avg_time_dog
pit_spi = pit_spi_0.append(avg_time_pit, ignore_index=True)
pit_spi
# grouping dataset by anymal type and counting
avg_time_adopt_other = avg_time_adopt[avg_time_adopt['Pit Bull']=='N']
avg_time_adopt_other['Pit Bull'] = 'Avg # Days to Adoption'
avg_time_adopt_other = avg_time_adopt_other.rename(columns = {'Pit Bull':'criteria', 'time_in_shelter_days':'value'})
avg_time_return_other = avg_time_return[avg_time_return['Pit Bull']=='N']
avg_time_return_other['Pit Bull'] = 'Avg # Days to Return to Owner'
avg_time_return_other = avg_time_return_other.rename(columns = {'Pit Bull':'criteria', 'time_in_shelter_days':'value'})
# combining avg time adoption and return
avg_time_other = avg_time_adopt_other.append(avg_time_return_other, ignore_index=True)
# getting % metrics for non pitbulls from dogs
others = dogs[dogs['Pit Bull']=='N']
others_by_ot = others.groupby(['outcome_type'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
others_count = others.shape[0]
others_by_ot['value'] = 100*others_by_ot['count']/others_count # this is percent
others_by_ot = others_by_ot.rename(columns = {'outcome_type':'criteria'})
top2_dbot = others_by_ot.head(2) # adoption and return to owner happen to be the top 2
others_by_it = others.groupby(['intake_type'])[['count']].sum().reset_index().sort_values(by='count', ascending=False)
others_by_it['value'] = 100*others_by_it['count']/others_count # this is percent
others_by_it = others_by_it.rename(columns = {'intake_type':'criteria'})
top2_dbit = others_by_it.head(2) # stray and owner surrender happen to be the top 2
# combining top2_dbit & top2_dbot into one df
other_spi_0 = top2_dbot.append(top2_dbit, ignore_index=True)
other_spi_0 = other_spi_0.drop(['count'],axis=1)
other_spi_0['criteria'] = '% '+ other_spi_0['criteria']
# combining dog_spi_0 with avg_time_dog
other_spi = other_spi_0.append(avg_time_other, ignore_index=True)
other_spi
categories = pit_spi['criteria'].tolist()
N = len(categories)
pit_values = pit_spi['value'].values.tolist()
pit_values += pit_values[:1] # need to repeat first value to close loop
other_values = other_spi['value'].values.tolist()
other_values += other_values[:1] # need to repeat first value to close loop
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(8,8))
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
plt.title('Pit Bull vs. Other Breeds', fontsize=18)
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='blue', size=10)
# Draw ylabels
ax.set_rlabel_position(1.5)
plt.yticks([10,20,30,40,50,60,70], ['10','20','30','40','50','60','70'], color='grey', size=8)
plt.ylim(0,80)
# Plot data
ax.plot(angles, pit_values, color='green', linewidth=1, linestyle='solid', label='Pit Bull or Pit Bull Mix')
ax.fill(angles, pit_values, color='green', alpha=0.1)
ax.plot(angles, other_values, color='red', linewidth=1, linestyle='solid', label='Other Breed')
ax.fill(angles, other_values, color='red', alpha=0.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, .4), fontsize=12);