import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))

happiness_report = pd.read_csv('../input/world-happiness-report-2019.csv')

happiness_report = happiness_report.rename(columns = {'Country (region)':'Country','SD of Ladder':'SD_of_ladder',

                         'Positive affect':'Positive_affect','Negative affect':'Negative_affect','Social support':'Social_support','Log of GDP per capital':'log_of_GDP_per_capital', 'Healthy life expectancy':'Healthy_life_expectancy'

                         })

happiness_report.head(n=10)
happiness_report.describe()
happiness_report.info()
happiness_report.columns
happiness_report.info()
happiness_report.isnull().sum()
happiness_report=happiness_report.fillna(method = 'ffill')
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(happiness_report.corr(),ax=ax,annot=True,linewidth=0.05,fmt='.2f',cmap='magma')

plt.show()
plt.scatter(happiness_report['Positive_affect'],happiness_report.Negative_affect)

plt.title('Positive affect compare with Negative affect')

plt.xlabel('Positive_affect')

plt.ylabel('Negative_affect')

plt.show()
plt.scatter(happiness_report['Positive_affect'],happiness_report.Social_support)

plt.xlabel('Positive_affect')

plt.ylabel('Social_support')

plt.title('Positive_affect compare with Social_support')
plt.scatter(happiness_report['Positive_affect'],happiness_report.Freedom)

plt.xlabel('Positive_affect')

plt.ylabel('Freedom')

plt.title('Positive_affect compare with Freedom')
plt.scatter(happiness_report['Positive_affect'],happiness_report.Generosity)

plt.xlabel('Positive_affect')

plt.ylabel('Generosity_affect')

plt.title('Positive_affect compare with Generosity')
plt.scatter(happiness_report['Negative_affect'],happiness_report.Corruption)

plt.xlabel('Negative_affect')

plt.ylabel('Corruption')

plt.title('Negative_affect compare with Corruption')
list_of_columns = list(happiness_report.columns)

log = list_of_columns[-2]
plt.scatter(happiness_report['Negative_affect'],happiness_report[log])

plt.xlabel('Negative_affect')

plt.ylabel('Log of GDP per capita')

plt.title('Negative_affect compare with Log of GDP per capita')
freedom = happiness_report['Freedom']

features = happiness_report.drop('Freedom', axis=1)
for v in ['Positive_affect']:

    sns.regplot(happiness_report[v], freedom, marker ='+', color='red')
log_of_GDP_per_capita= happiness_report[log]

features = happiness_report.drop(log, axis=1)

for v in ['Negative_affect']:

    sns.regplot(happiness_report[v], log_of_GDP_per_capita, marker ='+', color ='red')
list_of_columns = list(happiness_report.columns)

Health = list_of_columns[-1]
log_of_GDP_per_capita= happiness_report[log]

features = happiness_report.drop(log, axis=1)

for v in [Health]:

    sns.regplot(happiness_report[v], log_of_GDP_per_capita, marker ='+', color ='red')
Ladder = happiness_report['Ladder']

features = happiness_report.drop(log, axis=1)

for v in ['Social_support']:

    sns.regplot(happiness_report[v], Ladder, marker ='+', color ='red')
happiness_report.tail(n=10)
Ladder = happiness_report['Ladder']

features = happiness_report.drop(log, axis=1)

for v in [Health]:

    sns.regplot(happiness_report[v], Ladder, marker ='+', color ='red')
Ladder = happiness_report['Ladder']

features = happiness_report.drop(log, axis=1)

for v in [log]:

    sns.regplot(happiness_report[v], Ladder, marker ='+', color ='red')