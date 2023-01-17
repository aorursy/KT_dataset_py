import os



import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import numpy as np



ENV = 'kaggle'



if ENV == 'kaggle':

    DATA_ROOT_PATH = '../input/osic-pulmonary-fibrosis-progression'

    TRAIN_CSV_PATH = os.path.join(DATA_ROOT_PATH, 'train.csv')

    TEST_CSV_PATH = os.path.join(DATA_ROOT_PATH, 'test.csv')

    SUBMISSION_CSV_PATH = os.path.join(DATA_ROOT_PATH, 'sample_submission.csv')

else:

    raise UserWarning("Please choose a valid ENV")

    

font = {'family' : 'normal',

        'size'   : 16}

plt.rc('font', **font)
train_df = pd.read_csv(TRAIN_CSV_PATH).sort_values(['Patient', 'Weeks'])

train_df.head()
print(f"{len(train_df)} entries in training data")

print(f"{len(train_df['Patient'].unique())} unique patients")



fig, axes = plt.subplots(3, 2, figsize=(20,14))



axes = axes.flatten()



train_df[['Patient', 'SmokingStatus']].drop_duplicates().groupby(

    ['SmokingStatus']).count().plot.bar(title='Patients segmented by smoking status', ax=axes[0]);



train_df[['Patient', 'Sex']].drop_duplicates().groupby(

    ['Sex']).count().plot.bar(title='Patients segmented by sex', ax=axes[1]);



bin_size = 10

age_bins = pd.cut(train_df[['Patient', 'Age']].drop_duplicates()['Age'],

                  range(min(train_df['Age'] // bin_size)*bin_size, max(train_df['Age'] // bin_size)*bin_size + bin_size*2, bin_size))

train_df[['Patient', 'Age']].drop_duplicates().groupby(

    age_bins)['Age'].count().plot.bar(title='Patients segmented by age', ax=axes[2]);



bin_size = 500

first_fvc_df = train_df[['Patient', 'FVC', 'Percent']].drop_duplicates(subset=['Patient'])

fvc_bins = pd.cut(first_fvc_df['FVC'],

                  range(min(train_df['FVC'] // bin_size)*bin_size, max(train_df['FVC'] // bin_size)*bin_size + bin_size*2, bin_size))

first_fvc_df.groupby(fvc_bins)['FVC'].count().plot.bar(title='Patients segmented by first FVC reading (mL)', ax=axes[3]);



bin_size = 10

pct_bins = pd.cut(first_fvc_df['Percent'],

                  range(int(min(train_df['Percent'] // bin_size)*bin_size), int(max(train_df['Percent'] // bin_size)*bin_size + bin_size*2), bin_size))

first_fvc_df.groupby(pct_bins)['Percent'].count().plot.bar(title='Patients segmented by first FVC reading (percent)', ax=axes[4]);



train_df.groupby('Patient').count().groupby('FVC').count()['Sex'].plot.bar(title='Patients segmented by number of FVC measurements', ax=axes[5]);

axes[4].set_xlabel('# FVC measurements')



plt.tight_layout()
patients = train_df['Patient'].unique()



found_change = False

for patient in patients:

    if len(train_df[train_df['Patient'] == patient]['SmokingStatus'].unique()) > 1:

        found_change = True

        

print(found_change)



found_change = False

for patient in patients:

    if len(train_df[train_df['Patient'] == patient]['Age'].unique()) > 1:

        found_change = True

        

print(found_change)



# why not check while I'm at it...

found_change = False

for patient in patients:

    if len(train_df[train_df['Patient'] == patient]['Sex'].unique()) > 1:

        found_change = True

        

print(found_change)
n_cols = 3

n_rows = 5

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))

axes = axes.flatten()



for i, patient in enumerate(np.random.choice(patients, min(n_cols*n_rows, len(patients)), replace=False)):

    patient_df = train_df[train_df['Patient'] == patient]

    sex = patient_df['Sex'].unique()[0]

    age = patient_df['Age'].unique()[0]

    smoking_status = patient_df['SmokingStatus'].unique()[0]

    patient_df.plot('Weeks', 'FVC', ax=axes[i], title=f"{sex}, aged {age}, {smoking_status}", lw=2, style='o-')

    axes[i].grid()



plt.tight_layout()
### CHAGE FILTER PARAMS HERE ###

sex = 'Male'

age_bracket = (50, 70)

smoking_status = ['Ex-smoker', 'Currently smokes', 'Never smoked'][1]

###



train_df_filt = train_df[train_df['Sex'] == sex]

train_df_filt = train_df_filt[train_df_filt['Age'] >= age_bracket[0]]

train_df_filt = train_df_filt[train_df_filt['Age'] < age_bracket[1]]

train_df_filt = train_df_filt[train_df_filt['SmokingStatus'] == smoking_status]

patients = train_df_filt['Patient'].unique()



n_cols = 3

n_rows = min(5, -(- len(patients) // n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))

axes = axes.flatten()



for i, patient in enumerate(np.random.choice(patients, min(n_cols*n_rows, len(patients)), replace=False)):

    patient_df = train_df_filt[train_df_filt['Patient'] == patient]

    sex = patient_df['Sex'].unique()[0]

    age = patient_df['Age'].unique()[0]

    smoking_status = patient_df['SmokingStatus'].unique()[0]

    patient_df.plot('Weeks', 'FVC', ax=axes[i], title=f"{sex}, aged {age}, {smoking_status}", lw=2, style='o-')

    axes[i].grid()



plt.tight_layout()
patients = train_df['Patient'].unique()



first = []

last = []



for patient in patients:

    patient_df = train_df[train_df['Patient'] == patient]

    first.append(list(patient_df['Percent'])[0])

    last.append(list(patient_df['Percent'])[-1])



fig, ax = plt.subplots()

ax.scatter(first, last)

m, b = np.polyfit(first, last, 1)

ax.plot(first, m*np.array(first) + b, color='r')

ax.plot(first, first, color='lightgray')

ax.set_xlabel('First FVC result (%)')

ax.set_ylabel('Last FVC result (%)')

ax.set_title(f"last = {m:.2f}first + {b:.2f}")

ax.grid()

ax.legend(['Fit', 'x=x']);
### CHAGE FILTER PARAMS HERE ###

sex = 'Male'

age_bracket = (50, 70)

smoking_status = ['Ex-smoker', 'Currently smokes', 'Never smoked'][0]

###



train_df_filt = train_df[train_df['Sex'] == sex]

train_df_filt = train_df_filt[train_df_filt['Age'] >= age_bracket[0]]

train_df_filt = train_df_filt[train_df_filt['Age'] < age_bracket[1]]

train_df_filt = train_df_filt[train_df_filt['SmokingStatus'] == smoking_status]

patients = train_df_filt['Patient'].unique()



first = []

last = []



for patient in patients:

    patient_df = train_df[train_df['Patient'] == patient]

    first.append(list(patient_df['Percent'])[0])

    last.append(list(patient_df['Percent'])[-1])



fig, ax = plt.subplots()

ax.scatter(first, last)

m, b = np.polyfit(first, last, 1)

ax.plot(first, m*np.array(first) + b, color='r')

ax.plot(first, first, color='lightgray')

ax.set_xlabel('First FVC result (%)')

ax.set_ylabel('Last FVC result (%)');

ax.set_title(f"last = {m:.2f}first + {b:.2f}")

ax.grid()

ax.legend(['Fit', 'x=x'])
### CHAGE FILTER PARAMS HERE ###

sex = 'Male'

age_brackets = [(50, 60), (60, 70), (70, 80)]

smoking_status = ['Ex-smoker', 'Currently smokes', 'Never smoked'][0]

###



train_df_filt = train_df[train_df['Sex'] == sex]

train_df_filt = train_df_filt[train_df_filt['SmokingStatus'] == smoking_status]



# PLOT INDIVIDUALLY

n_ages = len(age_brackets)

n_cols = 3

n_rows = min(5, -(- n_ages // n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

axes = axes.flatten()



for i, age_bracket in enumerate(age_brackets):

    age_df = train_df_filt[train_df_filt['Age'] < age_bracket[1]]

    age_df = age_df[age_df['Age'] >= age_bracket[0]]

    if not len(age_df):

        continue

    m, b = np.polyfit(age_df['Percent'], age_df['FVC'], 1)

    age_df.plot.scatter('Percent', 'FVC', ax=axes[i], title=f"Age bracket: {age_bracket}", lw=2, style='o-')

    axes[i].plot(np.array(age_df['Percent']), m*np.array(age_df['Percent']) + b, color='r')

    axes[i].grid()

    

plt.tight_layout();



# PLOT TOGETHER

fig, ax = plt.subplots(1,1,figsize=(10,8))



# color selection

colors = np.random.choice(list(mcolors.TABLEAU_COLORS.keys()),

                          size=len(age_brackets),

                          replace=False)



for i, age_bracket in enumerate(age_brackets):

    age_df = train_df_filt[train_df_filt['Age'] < age_bracket[1]]

    age_df = age_df[age_df['Age'] >= age_bracket[0]]

    fvc = np.array(age_df['FVC'])

    if not len(fvc):

        continue

    percent = np.array(age_df['Percent'])

    ax.scatter(fvc, percent, color=colors[(age-age_bracket[0]) % len(colors)])

    m, b = np.polyfit(fvc, percent, 1)

    ax.plot(fvc, m*fvc + b, color=colors[(age-age_bracket[0]) % len(colors)])

    

ax.set_title('All plotted together for comparison')

ax.set_xlabel('FVC')

ax.set_ylabel('Percent')

ax.grid()

ax.legend(age_brackets);
fig, ax = plt.subplots(1,2, figsize=(20,5))

ax = ax.flatten()



def plot(param, ax, designations, colors):

    for color, designation in zip(colors, designations):

        train_df[train_df[param] == designation].plot.scatter('FVC', 'Percent', ax=ax, color=color, marker='.')

        fvc = train_df[train_df[param] == designation]['FVC']

        percent = train_df[train_df[param] == designation]['Percent']

        m, b = np.polyfit(fvc, percent, 1)

        ax.plot(fvc, m*fvc + b, color=color, linewidth=2)

        ax.legend(designations, loc='lower right')



plot('Sex', ax[0], ['Male', 'Female'], ['green', 'blue'])

plot('SmokingStatus', ax[1], ['Ex-smoker', 'Currently smokes', 'Never smoked'], ['green', 'blue', 'purple'])