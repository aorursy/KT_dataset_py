import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df = pd.read_csv("../input/covid19-hospitalizations-aug-22-2020/COVID-19 Laboratory-Confirmed Hospitalizations (Aug 22 2020).csv", skiprows=2, engine='python') # https://gis.cdc.gov/grasp/COVIDNet/COVID19_5.html

df = df.loc[df['Secondary Strata'] != 'Race/Ethnicity']

df['Percent'] = df['Percent'].fillna(0)

display(df.head(12))
df['Secondary Strata'].unique()
def getOne(series):

    return series.iat[0]



dfPivot = df.pivot_table(index=['Primary Strata', 'Primary Strata Name'], columns='Secondary Strata', values='Percent', aggfunc=getOne)

dfAge = dfPivot.xs('Age', level='Primary Strata')

dfAge
dfAge1 = dfAge[dfAge.columns[dfAge.mean(axis=0) > 20]].drop('Sex', axis=1)

dfAge1['In-hospital death'] = dfAge['In-hospital death']

dfAge1['Intensive care unit'] = dfAge['Intensive care unit']

dfAge1['Mechanical ventilation'] = dfAge['Mechanical ventilation']
dfAge1 = dfAge1.rename(columns={"Primary Strata Name": "Age"})

ind = dfAge1.index

dfAge2 = dfAge1.reindex([ind[0], ind[2], ind[1], ind[3], ind[4], ind[5]])

dfAge2
dfAge2.T.plot(kind='bar', figsize=(20, 5), legend=True, rot=10, xlabel="Condition", ylabel="Percent")