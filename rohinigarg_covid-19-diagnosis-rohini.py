%reset -f

import numpy as np

import pandas as pd

# 1.2 For plotting

import matplotlib.pyplot as plt

#import matplotlib

#import matplotlib as mpl     # For creating colormaps

import seaborn as sns

# 1.3 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os



%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

#InteractiveShell.ast_node_interactivity = "all"

#plt.style.use('dark_background')





df = pd.read_csv("/kaggle/input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")#, parse_dates=['daterep'])

df.drop('patient_id', axis=1, inplace=True)

df.rename(columns = 

{

    'patient_age_quantile': 'age',

    'sars_cov_2_exam_result': 'res',

    'patient_addmited_to_regular_ward_1_yes_0_no': 'regW',

    'patient_addmited_to_semi_intensive_unit_1_yes_0_no': 'semW',

    'patient_addmited_to_intensive_care_unit_1_yes_0_no': 'intW'

}, inplace=True)

df.tail()
dfPositive = df.loc[(df.res == 'positive'), 'age']

sns.catplot(x='res', y='age', data=df, kind='boxen')



# Observation -----

# 50% of the people effected are in age 35 (age bin 7) and 70 (age bin 14)

# 
def get_interesting_cols():

    dfPos = df.loc[df.res == 'positive', :]

    arrColumns = df.columns.values

    arrNotRequired = np.array(['patient_id', 'age', 'res'])

    arrColumns = np.setdiff1d(arrColumns, arrNotRequired)





    # Get percentage of data for each column (Only numeric columns)

    arrInterestingColumns = []

    for c in arrColumns:

        if df[c].dtype not in [np.float64, np.int64, np.int32, np.float32]:

            continue

        naRows = df.loc[np.isnan(df[c]), c].shape[0]

        avRows = df.loc[~np.isnan(df[c]), c].shape[0]

        total = naRows + avRows

        if total == 0:

            continue

        percent = avRows / total * 100

        if percent >= 10:

            arrInterestingColumns.append((c, percent))



    return pd.DataFrame(arrInterestingColumns, columns=["Column_name", "Percent_avail"])



print(get_interesting_cols())



# Observation ---------

# Most of the numeric data is missing. There are only 14 columns who have reported > 10% data

#
dfPos = df.loc[df.res == 'positive', :]

ax = sns.boxenplot(x='regW', y='age', data=dfPos)

ax.set(xlabel='Admitted in Regular ward?', ylabel='Age bins of 5')



# Observation ------

# People with age (12.5*5=)62.5 and (17*5=)85 were shifted to regular ward

#
ax = sns.boxenplot(x='semW', y='age', data=dfPos)

ax.set(xlabel='Admitted in Semi Intensive care ward?', ylabel='Age bins of 5')



# Observation ---- 

# People with age >= 65 (13*5) were shifted to semi-intensive care wards

#
ax = sns.boxenplot(x='intW', y='age', data=dfPos)

ax.set(xlabel='Admitted in Intensive care ward?', ylabel='Age bins of 5')



# Observation ---- 

# People with age >= 65 (13*5) were shifted to intensive care wards

#
df.dtypes.unique()

dfNum = df.select_dtypes(include = ['float64', 'int64']).copy()

dfNum['res'] = df['res']

dfNum.shape



%matplotlib inline



nCols = len(dfNum.columns.values)

fig = plt.figure(figsize=(20, 100))

plt.subplots_adjust(wspace=0.5, hspace=0.5)

for colIdx in range(0, nCols-1, 1):

    ax = plt.subplot(18, 4, colIdx+1)

    x = 'res'

    y = dfNum.columns.values[colIdx]

    ax = sns.boxenplot(x=x, y=y, data=df, ax=ax)

    ax.axes.set_title("COVID result vs {}".format(y))

    ax.set_xlabel("Result")

    ax.set_ylabel(y)





# Observation ------ 

# -------------  Interesting columns (direct relationships) out of numerical columns



# platelets, leukocytes, eosinophils, monocytes, ionized_calcium, magnesium,

# urine_density, rods, segmented, ferritin, arterial_lactic_acid, lipase_dosage,

# pco2_arterial_blood_gas_analysis, ph_arterial_blood_gas_analysis, 

# total_co2_arterial_blood_gas-analysis, hco3_arterial_blood_gas_analysis,

# po2_arterial_blood_gas_analysis, cto2_arterial_blood_gas_analysis,

# hb_saturation_arterial_blood_gases, basophils
dfCat = df.select_dtypes(include = ['O'])

#

#   Category to numeric mapping

#

def mapString(x):

    if x in ['detected', 't', 'positive', 'present', 'cloudy']:

        return 1

    elif x in ['not_detected', 'f', 'negative', 'absent', 'clear', 'normal']:

        return 0

    elif x in ['altered_coloring']:

        return 2

    elif x in ['lightly_cloudy']:

        return 3

    elif x in ['not_done']:

        return np.nan

    elif x == 'light_yellow':

        return 4

    elif x == 'yellow':

        return 5

    elif x == 'orange':

        return 6

    elif x == 'citrus_yellow':

        return 7

    elif x == '<1000':

        return 1000

    return x



#

#    Try converting to Object type fields

#

for col in dfCat.columns.values:

    try:

        df[col] = dfCat[col].map(mapString).astype('float64')

    except Exception as e:

        print(col + ' : ' + str(e))
nCols = len(dfCat.columns.values)

fig = plt.figure(figsize=(20, 100))

plt.subplots_adjust(wspace=0.5, hspace=0.5)

for colIdx in range(1, nCols-1, 1):

    ax = plt.subplot(18, 4, colIdx)

    x = 'res'

    y = dfCat.columns.values[colIdx]

    if y == 'res':

        continue

    try:

        ax = sns.barplot(x=x, y=y, data=df, ax=ax, estimator=np.sum)

    except Exception as ex:

        print(ex)

        print(y)

    ax.axes.set_title("COVID result vs {}".format(y))

    ax.set_xlabel("Result")

    ax.set_ylabel(y)

    

# Observation: Intersting categorical columns

# respiratory_syncytial_virus, influenza_a, influenza_b, coronavirusn163, rhinovirus_enterovirus

# coronavirus_hku1, inf_a_h1n1_2009, parainfluenza_4, metapneurovirus, influenza_b_rapid_test,

# influenza_a_rapid_test, strepto_a, urine_hemoglobin
numCols = ['platelets', 'leukocytes', 'eosinophils', 'monocytes', 'ionized_calcium', 

           'magnesium', 'urine_density', 'rods', 'segmented', 'ferritin', 'arterial_lactic_acid',

           'lipase_dosage', 'pco2_arterial_blood_gas_analysis', 'ph_arterial_blood_gas_analysis', 

           'total_co2_arterial_blood_gas_analysis', 'hco3_arterial_blood_gas_analysis',

           'po2_arterial_blood_gas_analysis', 'cto2_arterial_blood_gas_analysis', 

           'hb_saturation_arterial_blood_gases', 'basophils']



catCols = ['respiratory_syncytial_virus', 'influenza_a', 'influenza_b', 'coronavirusnl63',

           'rhinovirus_enterovirus', 'coronavirus_hku1', 'inf_a_h1n1_2009', 'parainfluenza_4',

           'metapneumovirus', 'influenza_b_rapid_test', 'influenza_a_rapid_test',

           'strepto_a', 'urine_hemoglobin', 'res']



dfCopy = df.copy()



cols = numCols + catCols



ss= StandardScaler()

newDf = ss.fit_transform(dfCopy.loc[:,cols])

newDf = pd.DataFrame(newDf, columns = cols)



for catCol in catCols:

    newDf[catCol] = df[catCol]
%matplotlib inline

fig1 = plt.figure(figsize=(20, 20))

pd.plotting.parallel_coordinates(newDf, 'res', colormap='winter')

plt.xticks(rotation=90)

plt.title("Parallel curves for COVID")



# Observation -----

# --------  19 columns can separate the data. Most of them are categorical columns

# platlets, leukocytes, eosinophils, monocytes, ionized_calcium, magnesium, rods, segmented,

# urine_density



# Categorical -------

# -------  'influenza_a', 'influenza_b', 'coronavirusnl63', 'rhinovirus_enterovirus', 'coronavirus_hku1', 

# 'inf_a_h1n1_2009', 'parainfluenza_4', 'metapneumovirus', 'influenza_b_rapid_test', 'influenza_a_rapid_test',

#  'strepto_a', 'urine_hemoglobin'

#