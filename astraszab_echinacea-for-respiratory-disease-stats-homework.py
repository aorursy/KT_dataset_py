import pandas as pd

import numpy as np

import scipy.stats as sps

import statsmodels.stats.multitest as multitest

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="darkgrid")
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/echinacea.csv')

data
echinacea_data = data.loc[data['Group'] == 'echinacea']

placebo_data = data.loc[data['Group'] == 'placebo']



echinacea_group_cases = echinacea_data['Count'].sum()

placebo_group_cases = placebo_data['Count'].sum()

print(f'Treatment group illness cases: {echinacea_group_cases}.')

print(f'Control group illness cases: {placebo_group_cases}.')
echinacea_data['Percentage'] = (100 * echinacea_data['Count'] 

                            / echinacea_group_cases)

placebo_data['Percentage'] = 100 * placebo_data['Count'] / placebo_group_cases

data = pd.concat([echinacea_data, placebo_data]).sort_index()



data
sns.barplot(x='Assessment', y='Percentage', hue='Group', data=data)

plt.title('Severity of disease in treatment and control groups')
observed = np.array(echinacea_data['Count'])

expected = np.array(placebo_data['Percentage']) * echinacea_group_cases / 100

_, chi_squared_pval = sps.chisquare(observed, expected)

print(f'p-value for chi-squared test is {chi_squared_pval}')
data = pd.read_csv('../input/events.csv')

data
echinacea_symptom_cases = data.at[18, 'Count']

placebo_symptom_cases = data.at[19, 'Count']

print(f'Number of cases with symptoms in treatment group: {echinacea_symptom_cases}.')

print(f'Number of cases with symptoms in control group: {placebo_symptom_cases}.')
data = data.drop([16, 17, 18, 19])
echinacea_data = data.loc[data['Group'] == 'echinacea']

placebo_data = data.loc[data['Group'] == 'placebo']



echinacea_data['Percentage'] = (100 * echinacea_data['Count'] 

                            / echinacea_symptom_cases)

placebo_data['Percentage'] = 100 * placebo_data['Count'] / placebo_symptom_cases

data = pd.concat([echinacea_data, placebo_data]).sort_index()



data
plt.figure(figsize=(12, 6))

sns.barplot(x='AdverseEvent', y='Percentage', hue='Group', data=data)

plt.title('Symptoms frequencies in treatment and control groups')
symptoms = ['itchiness', 'rash', 'hyper', 'diarrhea', 'vomiting',

            'headache', 'stomachache', 'drowsiness']

pvals = []



for symptom in symptoms:

    symptom_data = data[data['AdverseEvent'] == symptom]

    num_cases = symptom_data.iloc[0]['Count']

    prob = symptom_data.iloc[1]['Percentage'] / 100

    pvals.append(sps.binom_test(x=num_cases, n=echinacea_symptom_cases, p=prob))



pvals = np.array(pvals)

# Correct p-values with FWER powerful Holm method

h0_rejected, pvals, _, _ = multitest.multipletests(pvals, method='holm')



test_results = pd.DataFrame({'p-value': pvals, 'H0 rejected': h0_rejected}, 

                            index=symptoms)

test_results