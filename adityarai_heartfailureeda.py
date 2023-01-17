import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pathLink = None
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        pathlink = os.path.join(dirname,filename)
dataset = pd.read_csv(pathlink)
dataset.tail()
dataset.columns.values
dataset.isnull().sum(axis = 0)
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(2, 4)
f.set_figheight(15)
f.set_figwidth(15)

plot_labels = dataset.columns
plot_labels = plot_labels.values.tolist()
plot_labels = [item for item in plot_labels if item not in ['anaemia','diabetes','high_blood_pressure','smoking', 'sex']]
plot_labels = np.array(plot_labels).reshape(2,4)
for i in range(2):
    for j in range (4):
        sns.distplot(dataset[plot_labels[i][j]], ax = axes[i][j])


f, axes = plt.subplots(2, 3)
f.set_figheight(15)
f.set_figwidth(15)
smoker, non_smoker = len([item for item in dataset.smoking if item == 1]), len([item for item in dataset.smoking if item == 0])

anaemic, non_anaemic =len([item for item in dataset.anaemia if item == 1]), len([item for item in dataset.anaemia if item == 0])

high_bp, non_high_bp = len([item for item in dataset.high_blood_pressure if item == 1]), len([item for item in dataset.high_blood_pressure if item == 0])

diabetic, non_diabetic = len([item for item in dataset.diabetes if item == 1]), len([item for item in dataset.diabetes if item == 0])

male, female = len([item for item in dataset.sex if item == 1]), len([item for item in dataset.sex if item == 0])

death, no_death = len([item for item in dataset.DEATH_EVENT if item == 1]), len([item for item in dataset.DEATH_EVENT if item == 0])

axes[0,0].pie([smoker,non_smoker], labels = ['smoker', 'non smoker'], colors = ['aliceblue','palevioletred'])
axes[0,1].pie([anaemic,non_anaemic], labels = ['anaemic', 'non anaemic'], colors = ['aliceblue','palevioletred'])
axes[0,2].pie([high_bp,non_high_bp], labels = ['High BP', 'Non- High BP'], colors = ['aliceblue','palevioletred'])
axes[1,0].pie([diabetic,non_diabetic], labels = ['Diabetic', 'Non Diabetic'], colors = ['aliceblue','palevioletred'])
axes[1,1].pie([male,female], labels = ['Male', 'Female'], colors = ['aliceblue','palevioletred'])
axes[1,2].pie([death,no_death], labels = ['Dead', 'Alive'], colors = ['aliceblue','palevioletred'])
plt.show()

fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(dataset.corr(), annot = True, fmt='.1g')
smoke_dead = dataset.loc[(dataset['smoking'] == 1) & (dataset['DEATH_EVENT'] == 1)]
smoke_alive = dataset.loc[(dataset['smoking'] == 1) & (dataset['DEATH_EVENT'] == 0)]
plt.title('Understanding effect of smoking on Death Event')
plt.pie([len(smoke_dead),len(smoke_alive)], labels = ['Smoking and Dead', 'Smoking and Alive'], colors = ['aliceblue','palevioletred'])
male = dataset.loc[(dataset['sex'] == 1 )  & dataset['DEATH_EVENT'] == 0]
male_dead = dataset.loc[(dataset['sex'] == 1 ) & dataset['DEATH_EVENT'] == 1]
female = dataset.loc[(dataset['sex'] == 0)  & dataset['DEATH_EVENT'] == 0 ]
female_dead = dataset.loc[(dataset['sex'] == 0 ) & dataset['DEATH_EVENT'] == 1]

plt.title('Analysing cases of Death with respect to Gender')
#plt.pie([len(male_alive), len(male_dead), len(female_alive), len(female_dead)], labels = ['Male - Alive', 'Male - Dead', 'Female - Alive', 'Female - Dead'])
life_status = ['Alive','Dead'] 
bar_width = 0.2
plt.bar(['Male','Female'], [len(male), len(female)],bar_width, color='aliceblue')
plt.bar(['Male','Female'], [len(male_dead), len(female_dead)], bar_width, bottom = [len(male), len(female)], color='palevioletred')
plt.legend(life_status, loc = 3)
plt.show()
male_non_diabetic_alive = dataset.loc[(dataset['sex'] == 1 )  & (dataset['diabetes'] == 0 )  & dataset['DEATH_EVENT'] == 0]
male_non_diabetic_dead = dataset.loc[(dataset['sex'] == 1 )  & (dataset['diabetes'] == 0 ) & dataset['DEATH_EVENT'] == 1]
male_diabetic_alive = dataset.loc[(dataset['sex'] == 1 )  & (dataset['diabetes'] == 1 ) & dataset['DEATH_EVENT'] == 0]
male_diabetic_dead = dataset.loc[(dataset['sex'] == 1 )  & (dataset['diabetes'] == 1 ) & dataset['DEATH_EVENT'] == 1]
female_non_diabetic_alive = dataset.loc[(dataset['sex'] == 0 )  & (dataset['diabetes'] == 0 )  & dataset['DEATH_EVENT'] == 0]
female_non_diabetic_dead = dataset.loc[(dataset['sex'] == 0 )  & (dataset['diabetes'] == 0 ) & dataset['DEATH_EVENT'] == 1]
female_diabetic_alive = dataset.loc[(dataset['sex'] == 0 )  & (dataset['diabetes'] == 1 ) & dataset['DEATH_EVENT'] == 0]
female_diabetic_dead = dataset.loc[(dataset['sex'] == 0 )  & (dataset['diabetes'] == 1 ) & dataset['DEATH_EVENT'] == 1]
diabetic = dataset.loc[dataset['diabetes'] == 1]
non_diabetic = dataset.loc[dataset['diabetes'] == 0]
diabetic_alive = dataset.loc[(dataset['diabetes'] == 1) & (dataset['DEATH_EVENT'] == 0)]
diabetic_dead = dataset.loc[(dataset['diabetes'] == 1) & (dataset['DEATH_EVENT'] == 1)]
non_diabetic_alive = dataset.loc[(dataset['diabetes'] == 0) & (dataset['DEATH_EVENT'] == 0)]
non_diabetic_dead = dataset.loc[(dataset['diabetes'] == 0) & (dataset['DEATH_EVENT'] == 1)]

f, axes = plt.subplots(2,3)
f.set_figheight(20)
f.set_figwidth(20)
axes[0,0].title.set_text('Diabetes v/s Non Diabetic')
axes[0,0].pie([len(diabetic),len(non_diabetic)], labels = ['Diabetic', 'Non Diabetic'])
axes[0,1].title.set_text('Comparing Death ratio in Male Non Diabetic Patients')
axes[0,1].pie([len(male_non_diabetic_alive),len(male_non_diabetic_dead)], labels = ['Alive', 'Dead'])
axes[0,2].title.set_text('Comparing Death ratio in Male Diabetic Patients')
axes[0,2].pie([len(male_diabetic_alive), len(male_diabetic_dead)], labels = ['Alive','Dead'])
axes[1,0].title.set_text('Comparing Death ratio in Female Non-Diabetic Patients')
axes[1,0].pie([len(female_non_diabetic_alive), len(female_non_diabetic_dead)], labels = ['Alive', 'Dead'])
axes[1,1].title.set_text('Comparing Death ratio in Female Diabetic Patients')
axes[1,1].pie([len(female_diabetic_alive), len(female_diabetic_dead)], labels = ['Alive', 'Dead'])
axes[1,2].title.set_text('Comparing Dibetes and Non Diabetes related Mortality')
axes[1,2].pie([len(diabetic_alive), len(diabetic_dead), len(non_diabetic_alive), len(non_diabetic_dead)], labels = ['Diabetic Alive', 'Diabetic Dead','Non Diabetic Alive', 'Non Diabetic Dead'])
plt.show()
high_blood_pressure = dataset[dataset['high_blood_pressure'] == 1 ]
non_high_blood_pressure = dataset[dataset['high_blood_pressure'] == 0]
high_blood_pressure_alive = dataset[(dataset['high_blood_pressure']==1) & (dataset['DEATH_EVENT'] == 0)]
high_blood_pressure_dead = dataset[(dataset['high_blood_pressure']==1) & (dataset['DEATH_EVENT'] == 1)]
non_high_blood_pressure_alive = dataset[(dataset['high_blood_pressure']==0) & (dataset['DEATH_EVENT'] == 0)]
non_high_blood_pressure_dead = dataset[(dataset['high_blood_pressure']==0) & (dataset['DEATH_EVENT'] == 1)]

male_bp_alive = dataset.loc[(dataset['sex'] == 1 )  & (dataset['high_blood_pressure'] == 1 )  & dataset['DEATH_EVENT'] == 0]
male_bp_dead = dataset.loc[(dataset['sex'] == 1 )  & (dataset['high_blood_pressure'] == 1 ) & dataset['DEATH_EVENT'] == 1]
male_non_bp_alive = dataset.loc[(dataset['sex'] == 1 )  & (dataset['high_blood_pressure'] == 0 ) & dataset['DEATH_EVENT'] == 0]
male_non_bp_dead = dataset.loc[(dataset['sex'] == 1 )  & (dataset['high_blood_pressure'] == 0 ) & dataset['DEATH_EVENT'] == 1]
female_bp_alive = dataset.loc[(dataset['sex'] == 0 )  & (dataset['high_blood_pressure'] == 1 )  & dataset['DEATH_EVENT'] == 0]
female_bp_dead = dataset.loc[(dataset['sex'] == 0 )  & (dataset['high_blood_pressure'] == 1 ) & dataset['DEATH_EVENT'] == 1]
female_non_bp_alive = dataset.loc[(dataset['sex'] == 0 )  & (dataset['high_blood_pressure'] == 0 ) & dataset['DEATH_EVENT'] == 0]
female_non_bp_dead = dataset.loc[(dataset['sex'] == 0 )  & (dataset['high_blood_pressure'] == 0 ) & dataset['DEATH_EVENT'] == 1]




f, axes = plt.subplots(2,3)
f.set_figheight(20)
f.set_figwidth(20)
axes[0,0].title.set_text('High Blood Pressure v/s Non High Blood Pressue')
axes[0,0].pie([len(high_blood_pressure),len(non_high_blood_pressure)], labels = ['High Blood Pressue', 'Non High Blood Pressue'])
axes[0,1].title.set_text('Comparing Death ratio in Male Non High BP Patients')
axes[0,1].pie([len(male_non_bp_alive),len(male_non_bp_dead)], labels = ['Alive', 'Dead'])
axes[0,2].title.set_text('Comparing Death ratio in Male High BP Patients')
axes[0,2].pie([len(male_bp_alive), len(male_bp_dead)], labels = ['Alive','Dead'])
axes[1,0].title.set_text('Comparing Death ratio in Female Non-High BP Patients')
axes[1,0].pie([len(female_non_bp_alive), len(female_non_bp_dead)], labels = ['Alive', 'Dead'])
axes[1,1].title.set_text('Comparing Death ratio in Female High BP Patients')
axes[1,1].pie([len(female_bp_alive), len(female_bp_dead)], labels = ['Alive', 'Dead'])
axes[1,2].title.set_text('Comparing High BP and Non High BP related Mortality')
axes[1,2].pie([len(high_blood_pressure_alive), len(high_blood_pressure_dead), len(non_high_blood_pressure_alive), len(non_high_blood_pressure_dead)], labels = ['High BP Alive', 'High BP Dead','Non High BP Alive', 'Non High BP Dead'])
plt.show()
anaemic = dataset[dataset['anaemia'] == 1 ]
non_anaemic = dataset[dataset['anaemia'] == 0]
anaemic_alive = dataset[(dataset['anaemia']==1) & (dataset['DEATH_EVENT'] == 0)]
anaemic_dead = dataset[(dataset['anaemia']==1) & (dataset['DEATH_EVENT'] == 1)]
non_anaemic_alive = dataset[(dataset['anaemia']==0) & (dataset['DEATH_EVENT'] == 0)]
non_anaemic_dead = dataset[(dataset['anaemia']==0) & (dataset['DEATH_EVENT'] == 1)]

male_anaemic_alive = dataset.loc[(dataset['sex'] == 1 )  & (dataset['anaemia'] == 1 )  & dataset['DEATH_EVENT'] == 0]
male_anaemic_dead = dataset.loc[(dataset['sex'] == 1 )  & (dataset['anaemia'] == 1 ) & dataset['DEATH_EVENT'] == 1]
male_non_anaemic_alive = dataset.loc[(dataset['sex'] == 1 )  & (dataset['anaemia'] == 0 ) & dataset['DEATH_EVENT'] == 0]
male_non_anaemic_dead = dataset.loc[(dataset['sex'] == 1 )  & (dataset['anaemia'] == 0 ) & dataset['DEATH_EVENT'] == 1]
female_anaemic_alive = dataset.loc[(dataset['sex'] == 0 )  & (dataset['anaemia'] == 1 )  & dataset['DEATH_EVENT'] == 0]
female_anaemic_dead = dataset.loc[(dataset['sex'] == 0 )  & (dataset['anaemia'] == 1 ) & dataset['DEATH_EVENT'] == 1]
female_non_anaemic_alive = dataset.loc[(dataset['sex'] == 0 )  & (dataset['anaemia'] == 0 ) & dataset['DEATH_EVENT'] == 0]
female_non_anaemic_dead = dataset.loc[(dataset['sex'] == 0 )  & (dataset['anaemia'] == 0 ) & dataset['DEATH_EVENT'] == 1]




f, axes = plt.subplots(2,3)
f.set_figheight(20)
f.set_figwidth(20)
axes[0,0].title.set_text('Anaemic v/s Non Anaemic')
axes[0,0].pie([len(anaemic),len(non_anaemic)], labels = ['Anaemic', 'Non Anaemic'])
axes[0,1].title.set_text('Comparing Death ratio in Male Non Anaemic Patients')
axes[0,1].pie([len(male_non_anaemic_alive),len(male_non_anaemic_dead)], labels = ['Alive', 'Dead'])
axes[0,2].title.set_text('Comparing Death ratio in Male Anaemic Patients')
axes[0,2].pie([len(male_anaemic_alive), len(male_anaemic_dead)], labels = ['Alive','Dead'])
axes[1,0].title.set_text('Comparing Death ratio in Female Non-Anaemic Patients')
axes[1,0].pie([len(female_non_anaemic_alive), len(female_non_anaemic_dead)], labels = ['Alive', 'Dead'])
axes[1,1].title.set_text('Comparing Death ratio in Female High Anaemic Patients')
axes[1,1].pie([len(female_anaemic_alive), len(female_anaemic_dead)], labels = ['Alive', 'Dead'])
axes[1,2].title.set_text('Comparing Anaemic and Non Anaemic related Mortality')
axes[1,2].pie([len(anaemic_alive), len(anaemic_dead), len(non_anaemic_alive), len(non_anaemic_dead)], labels = ['Anaemic Alive', 'Anaemic Dead','Non Anaemic Alive', 'Non Anaemic Dead'])
plt.show()
