# Imports and Basic Data Cleaning (Drop Duplicates and change 97, 98, and 99 to np.nan)
import numpy as np
import pandas as pd 
import os
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
# Data Cleaning
df = pd.read_csv('/kaggle/input/covid19-patient-precondition-dataset/covid.csv')
df = df.drop_duplicates().reset_index(drop=True)
df = df.replace(97, np.nan)
df = df.replace(98, np.nan)
df = df.replace(99, np.nan)

# Fix date data
df.entry_date = pd.to_datetime(df.entry_date, dayfirst=True, errors='coerce')
df.date_symptoms = pd.to_datetime(df.date_symptoms, dayfirst=True, errors='coerce')
df.date_died = pd.to_datetime(df.date_died, dayfirst=True, errors='coerce')

# Docs says that sex == 1 means male, but then some males are pregnant and no females are pregnant. 
# Also, all pregnancy data for sex == 2 is NaN, so it seems that sex should be switched, as opposed to pregnant column.
df.sex = df.sex.replace([1,2], ['Female', 'Male'])

# Change patient_type column to inpatient, change to bool
df = df.rename(columns = {'patient_type':'outpatient'})

# Change 1 to True and 2 to False (bool Type) and 3 to 'Awaiting' (for covid_res colum)
df.loc[:, df.columns != 'age'] = df.loc[:, df.columns != 'age'].replace([1,2,3], [True, False, 'Awaiting'])
# Data Augmentation

# Just because date_died == NaT does not necesarrily mean the patient is not deceasd
# However, if date_died != NaT, I consider that patient to be confirmed deceased
df['confirmed_deceased'] = np.where(df['date_died'].isnull(), False, True)

# Get time difference between different events
df['entry_symptoms_delta'] = df.entry_date - df.date_symptoms
df['died_entry_delta'] = df.date_died - df.entry_date
df['died_symptoms_delta'] = df.date_died - df.date_symptoms
fig, ax = plt.subplots(figsize=(10,10))

plt.style.use('fivethirtyeight')

positive_alive = df[df['covid_res'] == True].confirmed_deceased.value_counts()[False]
positive_deceased = df[df['covid_res'] == True].confirmed_deceased.value_counts()[True]

negative_alive = df[df['covid_res'] == False].confirmed_deceased.value_counts()[False]
negative_deceased = df[df['covid_res'] == False].confirmed_deceased.value_counts()[True]

awaiting_alive = df[df['covid_res'] == 'Awaiting'].confirmed_deceased.value_counts()[False]
awaiting_deceased = df[df['covid_res'] == 'Awaiting'].confirmed_deceased.value_counts()[True]

outer_labels = ['Positive', 'Negative', 'Awaiting']
outer_sizes = [positive_alive + positive_deceased, negative_alive + negative_deceased, awaiting_alive + awaiting_deceased]
inner_labels = ['Alive with Covid-19', 'Deceased with Covid-19', 'Alive without Covid-19', 'Deceased without Covid-19', 'Alive Awaiting Result','Deceased Awaiting Result']
inner_sizes = [positive_alive, positive_deceased, negative_alive, negative_deceased, awaiting_alive, awaiting_deceased]
#outer_colors = ['#ff6666', '#0366fc', '#99ff99']
#inner_colors = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([2, 1, 6, 5, 10, 9]))
outer_explode = (0,0,0)
inner_explode = (0,0,0,0,0,0)

wedges, texts = plt.pie(outer_sizes, labels=outer_labels, colors=outer_colors, startangle=90,frame=True, explode=outer_explode,radius=3)

pos = (positive_alive + positive_deceased) / (positive_alive + positive_deceased + negative_alive + negative_deceased + awaiting_alive + awaiting_deceased)
neg = (negative_alive + negative_deceased) / (positive_alive + positive_deceased + negative_alive + negative_deceased + awaiting_alive + awaiting_deceased)
awa = (awaiting_alive + awaiting_deceased) / (positive_alive + positive_deceased + negative_alive + negative_deceased + awaiting_alive + awaiting_deceased)

leg1 = ax.legend(wedges, ['Positive: ' +  "{:.2%}".format(pos), 'Negative: ' + "{:.2%}".format(neg), 'Awaiting: ' + "{:.2%}".format(awa)],
          title='Covid-19 Diagnosis',
          loc="center left",
          bbox_to_anchor=(-0.4, 0, 0.5, 1))

wedges, texts = plt.pie(inner_sizes,colors=inner_colors,startangle=90, explode=inner_explode,radius=2 )

pos = positive_deceased / (positive_alive + positive_deceased)
neg = negative_deceased / (negative_alive + negative_deceased)
awa = awaiting_deceased / (awaiting_alive + awaiting_deceased)


label_wedges = [ wedges[index] for index in [1,3,5] ]
leg2 = ax.legend(label_wedges, ['Positive: ' + "{:.2%}".format(pos), 'Negative: ' + "{:.2%}".format(neg), 'Awaiting: ' + "{:.2%}".format(awa)],
          title='Mortality Rate by Type',
          loc="center right",
          bbox_to_anchor=(0.85, 0, 0.5, 1))

centre_circle = plt.Circle((0,0),1.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
 
ax.add_artist(leg1)
plt.axis('equal')
plt.title('Patient Outcomes by Covid-19 Diagnosis')

plt.show()

def assign_age_bin(row):
    age = row
    if age <= 2:
        return "2 and Under"
    elif age >= 3 and age <= 7:
        return "3 to 7"
    elif age >= 8 and age <= 12:
        return "8 to 12"
    elif age >= 13 and age <= 17:
        return "13 to 17"
    elif age >= 18 and age <= 24:
        return "18 to 24"
    elif age >= 25 and age <= 34:
        return "25 to 34"
    elif age >= 35 and age <= 44:
        return "35 to 44"
    elif age >= 45 and age <= 54:
        return "45 to 54"
    elif age >= 55 and age <= 64:
        return "55 to 64"
    elif age >= 65 and age <= 74:
        return "65 to 74"
    elif age >= 75 and age <= 84:
        return "75 to 84"
    elif age >= 85:
        return "85+"
    
df['age_bin'] = df.age.apply(lambda row: assign_age_bin(row))
df_covid = df[df.covid_res == True]

plt.figure(figsize=(20,10))
sns.distplot(df_covid.age, kde=False)
plt.title('Age Distribution')
plt.ylabel('Number of Patients')
plt.show()

col_order = ['2 and Under', '3 to 7', '8 to 12', '13 to 17', '18 to 24', '25 to 34', '35 to 44', '45 to 54', '55 to 64', '65 to 74', '75 to 84', '85+']

sns.catplot(x='confirmed_deceased', y=None, col="age_bin", data=df_covid, saturation=.5, col_order=col_order, kind='count', ci=None, aspect=.6, height=5)
plt.show()

prob = []
for age_bin_lab in col_order:
    vc = df_covid[df_covid.age_bin == age_bin_lab]['confirmed_deceased'].value_counts()
    prob.append(vc[True]/(vc[False] + vc[True]))

plt.figure(figsize=(20,10))
sns.lineplot(x = col_order, y = prob, sort=False)
ax = sns.scatterplot(x = col_order, y = prob, s = 200)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.title('Age Group vs. Fatality Rate')
plt.xlabel('')
plt.ylabel('')
plt.show()
ax = sns.catplot(x='confirmed_deceased', y=None, col="sex", data=df_covid,saturation=.5,col_order=['Male','Female'],kind='count', ci=None, aspect=.6)
plt.show()

cols = ['Male', 'Female']
prob = []
for sex in cols:
    vc = df_covid[df_covid.sex == sex]['confirmed_deceased'].value_counts()
    prob.append(vc[1]/(vc[0] + vc[1]))
    
plt.figure(figsize=(6,6))
ax = sns.barplot(x = cols, y = prob)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
plt.title('Sex vs. Fatality Rate')
plt.ylim((0, 0.20))
plt.xlabel('Sex')
plt.ylabel('Fatality Rate')
plt.show()

df_covid_male = df_covid[df_covid.sex == 'Male']
df_covid_female = df_covid[df_covid.sex == 'Female']

col_order = ['2 and Under', '3 to 7', '8 to 12', '13 to 17', '18 to 24', '25 to 34', '35 to 44', '45 to 54', '55 to 64', '65 to 74', '75 to 84', '85+']

ax = sns.catplot(x='confirmed_deceased', y=None, col="age_bin", data=df_covid_male, saturation=.5, col_order=col_order, kind='count', ci=None, aspect=.6, height=5)
ax.fig.suptitle('Male Patients', y =1.1)
plt.show()

ax = sns.catplot(x='confirmed_deceased', y=None, col="age_bin", data=df_covid_male, saturation=.5, col_order=col_order, kind='count', ci=None, aspect=.6, height=5)
ax.fig.suptitle('Female Patients', y =1.1)
plt.show()

prob_male = []
prob_female = []
for age_bin_lab in col_order:
    vc_male = df_covid_male[df_covid_male.age_bin == age_bin_lab]['confirmed_deceased'].value_counts()
    prob_male.append(vc_male[True]/(vc_male[False] + vc_male[True]))
    vc_female = df_covid_female[df_covid_female.age_bin == age_bin_lab]['confirmed_deceased'].value_counts()
    prob_female.append(vc_female[True]/(vc_female[False] + vc_female[True]))

plt.figure(figsize=(20,10))
sns.lineplot(x = col_order, y = prob_male, color='g', sort=False)
ax = sns.scatterplot(x = col_order, y = prob_male, color='g', s = 200, label='Male')
sns.lineplot(x = col_order, y = prob_female, color='m', sort=False)
ax = sns.scatterplot(x = col_order, y = prob_female, color='m', s = 200, label = 'Female')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.title('Age Group vs. Fatality Rate by Sex')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
