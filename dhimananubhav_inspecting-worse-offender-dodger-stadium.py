%matplotlib  inline

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import glob
import missingno as msno
from fbprophet import Prophet

from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import statsmodels.api as sm
from itertools import product
from math import sqrt
from sklearn.metrics import mean_squared_error 

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
colors = ["windows blue", "amber", "faded green", "dusty purple"]
sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })
sns.set(context='poster', font_scale=.8, style='ticks')

inspections = pd.read_csv('../input/restaurant-and-market-health-inspections.csv', 
                          parse_dates=['activity_date'])
inspections_raw = inspections.copy()
inspections.set_index('serial_number', inplace=True)

violations = pd.read_csv('../input/restaurant-and-market-health-violations.csv',
                        parse_dates=['activity_date'])
violations_raw = violations.copy()
inspections = inspections[inspections.service_description == 'ROUTINE INSPECTION']
inspections = inspections[inspections.program_status == 'ACTIVE']

cols = ['activity_date', 'employee_id', 'facility_id', 'record_id', 'pe_description',
       'score', 'grade']

inspections = inspections[cols]

inspections.head(3)
print(inspections.employee_id.nunique(), 'employees inspected', inspections.facility_id.nunique(), 
      'facilities and recorded', inspections.record_id.nunique(), 'entries from', 
      inspections.activity_date.min().date(), 'to',
     inspections.activity_date.max().date())
daily_inspections = pd.DataFrame(inspections.groupby(['activity_date'])['facility_id'].nunique())
daily_inspections.reset_index(inplace=True)
daily_inspections.head(3)
ax = sns.lineplot(x='activity_date', y='facility_id',  data=daily_inspections)
ax.set_xlim(daily_inspections.activity_date.min(), daily_inspections.activity_date.max())
ax.set_title('Number of facilities visited per day')
ax.set_xlabel('Date')
ax.set_ylabel('# Facilities')
sns.despine();
facility_type = pd.DataFrame(inspections.groupby(['pe_description'])['facility_id'].nunique())
facility_type.reset_index(inplace=True)
facility_type = facility_type.sort_values(by='pe_description')
ax = sns.barplot(y='pe_description', x='facility_id', data=facility_type, color='#081d58')
ax.set_xlabel('# Facilities')
ax.set_ylabel('Facility type')
ax.set_title('Number of facilities per Facility type')
sns.despine();
facility_score = pd.DataFrame(inspections.groupby(['pe_description'])['score'].mean())
facility_score.reset_index(inplace=True)
facility_score = facility_score.sort_values(by='pe_description')
facility_score['score'] = np.round(facility_score['score'], 1)
ax = sns.barplot(y='pe_description', x='score', data=facility_score, color='#081d58')
ax.set_xlabel('Average Score')
ax.set_ylabel('Facility type')
ax.set_xlim(facility_score.score.min()-.5, facility_score.score.max()+.5)
ax.set_title('Avg Score per Facility type')
sns.despine();
facility_grade = pd.DataFrame(inspections.groupby(['pe_description', 'grade'])['score'].mean())
facility_grade.reset_index(inplace=True)
facility_grade = facility_grade.sort_values(by='pe_description')
facility_grade['score'] = np.round(facility_grade['score'], 1)
ax = sns.barplot(y='pe_description', x='score', data=facility_grade[facility_grade.grade == 'A'], color='#225ea8')
ax.set_xlabel('Average Score')
ax.set_ylabel('Facility type')
ax.set_title('Avg Grade A Score per Facility type')
ax.set_xlim(facility_grade.score.min()-.5, facility_grade.score.max()+.5)
sns.despine();
ax = sns.barplot(y='pe_description', x='score', data=facility_grade[facility_grade.grade == 'B'], color='#7fcdbb')
ax.set_xlabel('Average Score')
ax.set_ylabel('Facility type')
ax.set_title('Avg Grade B Score per Facility type')
ax.set_xlim(facility_grade.score.min()-.5, facility_grade.score.max()+.5)
sns.despine();
ax = sns.barplot(y='pe_description', x='score', data=facility_grade[facility_grade.grade == 'C'], color='#edf8b1')
ax.set_xlabel('Average Score')
ax.set_ylabel('Facility type')
ax.set_title('Avg Grade C Score per Facility type')
ax.set_xlim(facility_grade.score.min()-.5, facility_grade.score.max()+.5)
sns.despine();
most_inspected = pd.DataFrame(inspections.groupby('facility_id')['record_id'].count())
most_inspected = most_inspected.sort_values('record_id', ascending=False)
most_inspected.head(1)
most_inspected_facility = inspections_raw[inspections_raw.facility_id == most_inspected.index[0]]
most_inspected_facility.head(3)
dodgers_score  = pd.DataFrame(most_inspected_facility.groupby('activity_date')['score'].mean())
dodgers_score.reset_index(inplace=True)
dodgers_score['score'] = np.round(dodgers_score.score, 1)
dodgers_score.head(3)
ax = sns.lineplot(x='activity_date', y='score', data=dodgers_score)
ax = sns.scatterplot(x='activity_date', y='score', data=dodgers_score)
ax.set_xlim(dodgers_score.activity_date.min(), dodgers_score.activity_date.max())
ax.set_title('Score of Dodger Stadium is decreasing every year')
ax.set_xlabel('Date')
ax.set_ylabel('Avg Score')
sns.despine();
dodger_violations = violations[violations.facility_id == most_inspected.index[0]]
dodger_violations.head(3)
dodger_violation_count = pd.DataFrame(dodger_violations.groupby(['activity_date'])['violation_code'].count())
dodger_violation_count.reset_index(inplace=True)
dodger_violation_count.tail(3)
ax = sns.lineplot(x='activity_date', y='violation_code', data=dodger_violation_count)
ax = sns.scatterplot(x='activity_date', y='violation_code', data=dodger_violation_count)
ax.set_xlim(dodger_violation_count.activity_date.min(), dodger_violation_count.activity_date.max())
ax.set_title('Violations by Dodger Stadium is increasing every year')
ax.set_xlabel('Date')
ax.set_ylabel('# Violations')
sns.despine();
dodger_violation_type = pd.DataFrame(dodger_violations.groupby(['violation_description'])['violation_code'].count())
dodger_violation_type.reset_index(inplace=True)
dodger_violation_type = dodger_violation_type.sort_values('violation_code', ascending=False)
dodger_violation_type = dodger_violation_type.head(6)
ax = sns.barplot(y='violation_description', x='violation_code', data=dodger_violation_type, color='#081d58')
ax.set_xlabel('# Violations')
ax.set_ylabel('Violation type')
ax.set_title('Top 6 Non-Compliance violation in Dodger Stadium')
sns.despine();
