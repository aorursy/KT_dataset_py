import numpy as np

import pandas as pd

import matplotlib as plt

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.gridspec import GridSpec

from matplotlib.offsetbox import AnchoredText
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
meta_data = pd.read_csv("../input/IndiaAffectedWaterQualityAreas.csv",encoding='latin1')
meta_data.columns
meta_data['State Name'].unique()
#meta_data['State Name'].value_counts()
meta_data['Quality Parameter'].value_counts()
meta_data['Quality Parameter'].groupby(meta_data['State Name']).describe()
#meta_data.groupby("State Name").size()
import dateutil

meta_data['date'] = meta_data['Year'].apply(dateutil.parser.parse, dayfirst=True)

import datetime as dt

meta_data['date'] = pd.to_datetime(meta_data['date'])

meta_data['year'] = meta_data['date'].dt.year

meta_data['month'] = meta_data['date'].dt.month





State_Data = meta_data[['State Name', 'Quality Parameter', 'year','month']]

del State_Data ['year']

del State_Data ['month']
import sklearn

from sklearn.preprocessing import LabelEncoder

numbers = LabelEncoder()
State_Data['Quality'] = numbers.fit_transform(State_Data['Quality Parameter'].astype('str'))
State_Data.head(5)
Group1 = State_Data.groupby(['State Name','Quality Parameter','Quality']).count()

Group1
#pd.DataFrame({'count' : State_Data.groupby( [ "State Name", "Quality"] ).size()}).reset_index()
State_Quality_Count = pd.DataFrame({'count' : State_Data.groupby( [ "State Name", "Quality","Quality Parameter"] ).size()}).reset_index()
TAMIL_NADU   =  State_Quality_Count[State_Quality_Count["State Name"] == "TAMIL NADU"]    

ANDHRA_PRADESH = State_Quality_Count[State_Quality_Count["State Name"] == "ANDHRA PRADESH"]

KERALA = State_Quality_Count[State_Quality_Count["State Name"] == "KERALA"]

KARNATAKA = State_Quality_Count[State_Quality_Count["State Name"] == "KARNATAKA"]

GUJARAT = State_Quality_Count[State_Quality_Count["State Name"] == "GUJARAT"]

MAHARASHTRA = State_Quality_Count[State_Quality_Count["State Name"] == "MAHARASHTRA"]





TAMIL_NADU
plt.figure(figsize=(6,4))

ax = sns.barplot(x="count", y ="Quality Parameter", data = TAMIL_NADU)

ax.set(xlabel='Count')

sns.despine(left=True, bottom=True)

plt.title("Water Quality Parameter In Tamil Nadu")
KARNATAKA
plt.figure(figsize=(6,4))

ax = sns.barplot(x="count", y ="Quality Parameter", data = KARNATAKA)

ax.set(xlabel='Count')

sns.despine(left=True, bottom=True)

plt.title("Water Quality Parameter In Karnataka")
MAHARASHTRA
plt.figure(figsize=(6,4))

ax = sns.barplot(x="count", y ="Quality Parameter", data = MAHARASHTRA)

ax.set(xlabel='Count')

sns.despine(left=True, bottom=True)

plt.title("Water Quality Parameter In Mahrashtra")
GUJARAT
plt.figure(figsize=(6,4))

ax = sns.barplot(x="count", y ="Quality Parameter", data = GUJARAT)

ax.set(xlabel='Count')

sns.despine(left=True, bottom=True)

plt.title("Water Quality Parameter In Gujarat")
ANDHRA_PRADESH
plt.figure(figsize=(6,4))

ax = sns.barplot(x="count", y ="Quality Parameter", data = ANDHRA_PRADESH)

ax.set(xlabel='Count')

sns.despine(left=True, bottom=True)

plt.title("Water Quality Parameter In Andhra Pradesh")
plt.figure(figsize=(6,4))

ax = sns.barplot(x="count", y ="Quality Parameter", data = KERALA)

ax.set(xlabel='Count')

sns.despine(left=True, bottom=True)

plt.title("Water Quality Parameter In Kerala")
x = State_Quality_Count.groupby('State Name')

plt.rcParams['figure.figsize'] = (9.5, 6.0)

genre_count = sns.barplot(y='Quality Parameter', x='count', data=State_Quality_Count, palette="Blues", ci=None)

plt.show()