import os

import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

from collections.abc import Iterable

import plotly.graph_objects as go

pd.options.display.max_rows = 100

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

print('train shape', train.shape, 'test shape', test.shape)





# Plot a timeseries line Weeks x FVC (or %)

def plot_timeserie(patient_id, var='FVC'):

    data = train[train.Patient==patient_id]

    sex = data.loc[data.index[0], 'Sex']

    smoke = data.loc[data.index[0], 'SmokingStatus']

    print("Week: {} to {} | Age: {} | Sex: {} | Smoke: {}".format(

        data.Weeks.min(), data.Weeks.max(), data.Age.max(), sex, smoke))



    fig, ax = plt.subplots(figsize=(12, 4))

    p1 = sns.lineplot(x='Weeks', y=var, data=data)

    for i in data.index:

        s = "w{}: {:.0f}".format(data.loc[i, 'Weeks'], data.loc[i, var])

        ax.text(data.loc[i, 'Weeks'], data.loc[i, var], s)

    



# Plot FVC for multiple patients (one line for each)

def plot_timeseries(patient_ids, var='FVC'):

    if not isinstance(patient_ids, Iterable):

        patient_ids = [patient_ids]

    

    plt.figure(figsize=(12, 6))

    data = train[train.Patient.isin(patient_ids)]

    p1 = sns.lineplot(x='Weeks', y=var, data=data, hue='Patient')

    p1.get_legend().remove()
train.head()
patients = train.Patient.unique()

print("There are", len(train), "records in the training set")

print("There are", len(patients), "unique patients in the training set")
# create curves and buttons for menu

traces = []



for i, patient_id in enumerate(patients):

    tmp = train[train.Patient == patient_id]

    traces.append(go.Scatter(x=tmp.Weeks, y=tmp.FVC, text=tmp.FVC, mode='lines+markers', name=patient_id[-5:]))

    vx = [i == j for j in range(len(patients))]



# create plot

fig = go.Figure()

fig.add_traces(traces)

fig.update_layout(title_text="FVC Curve")
outlier_patients = [

    "ID00076637202199015035026", "ID00077637202199102000916", "ID00082637202201836229724",

    "ID00117637202212360228007", "ID00119637202215426335765", "ID00126637202218610655908",

    "ID00135637202224630271439", "ID00165637202237320314458", "ID00170637202238079193844",

    "ID00172637202238316925179", "ID00197637202246865691526", "ID00218637202258156844710",

    "ID00235637202261451839085", "ID00288637202279148973731", "ID00323637202285211956970",

    "ID00337637202286839091062", "ID00355637202295106567614"

]

for patient_id in outlier_patients:

    plot_timeserie(patient_id)
metadata = train.groupby('Patient').first()
plt.figure(figsize=(10,4))

_ = sns.distplot(metadata.Age, bins=20).set_title("Age distribution")
plt.figure(figsize=(10,4))

_ = sns.countplot(metadata.SmokingStatus, hue=metadata.Sex).set_title("Smoking status and sex distribution")
print(metadata.Sex.value_counts(normalize=False))

print(metadata.SmokingStatus.value_counts(normalize=False))
plt.figure(figsize=(10,4))

_ = sns.distplot(metadata[metadata.SmokingStatus == 'Ex-smoker'].Age, hist=False).set_title("Age by Smoking Status")

_ = sns.distplot(metadata[metadata.SmokingStatus == 'Never smoked'].Age, hist=False)

_ = sns.distplot(metadata[metadata.SmokingStatus == 'Currently smokes'].Age, hist=False)
plt.figure(figsize=(10,4))

_ = sns.distplot(train.FVC, bins=20).set_title("FVC distribution")
plt.figure(figsize=(10,4))

_ = sns.distplot(train[train.Sex == 'Male'].FVC, bins=20).set_title("Male (blue) vs Female (orange) FVC distribution")

_ = sns.distplot(train[train.Sex == 'Female'].FVC, bins=20)
plt.figure(figsize=(10,4))

_ = sns.distplot(train[train.SmokingStatus == 'Ex-smoker'].FVC, hist=False).set_title("FVC by Smoking Status")

_ = sns.distplot(train[train.SmokingStatus == 'Never smoked'].FVC, hist=False)

_ = sns.distplot(train[train.SmokingStatus == 'Currently smokes'].FVC, hist=False)
plot_timeseries(patients, var='FVC')
plot_timeseries(patients, var='Percent')
smoking_patients = train[train.SmokingStatus == 'Currently smokes'].Patient.unique()

plot_timeseries(smoking_patients, var='FVC')
# get value of first FVC measure for each patient

idx = train.groupby('Patient').Weeks.idxmin()

first_fvc = train.loc[idx, ['Patient', 'FVC']]

# Divide in bins (quantiles)

bin_edges = stats.mstats.mquantiles(first_fvc.FVC, [0.1*i for i in range(1, 11)])

bin_edges
for i in range(10):

    if i == 0:

        tmp = first_fvc[(first_fvc.FVC <= bin_edges[i])]

    else:

        tmp = first_fvc[(first_fvc.FVC <= bin_edges[i]) & (first_fvc.FVC > bin_edges[i-1])]

    plot_timeseries(tmp.Patient, var='FVC')