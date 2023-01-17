# Import dependancies

import numpy as np

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt



!pip install lifelines

from lifelines import CoxPHFitter

from lifelines import KaplanMeierFitter



%matplotlib inline
data = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', names = ['Age','Operation_year','Nb_pos_detected','Surv'])

data.head(5)
pandas_profiling.ProfileReport(data)
T = data.Age

E = data.Surv



## create a kmf object

km = KaplanMeierFitter() 



## Fit the data into the model

km.fit(T, E,label='Kaplan Meier Estimate')



## Create an estimate

km.plot(ci_show=False) ## ci_show is meant for Confidence interval, since our data set is too tiny, not showing it.
## Instantiate the class to create an object

km_2 = KaplanMeierFitter()



## creating 2 cohorts : with at least one positive axillary detected, and one with no one detected

groups = data['Nb_pos_detected']   

i1 = (groups >= 1)   

i2 = (groups < 1)     





## fit the model for 1st cohort

km_2.fit(T[i1], E[i1], label='at least one positive axillary detected')

a1 = km_2.plot()



## fit the model for 2nd cohort

km_2.fit(T[i2], E[i2], label='no positive axillary nodes detected')

km_2.plot(ax=a1)
# Create Model

cph = CoxPHFitter()



# Fit the data to train the model

cph.fit(data, 'Age', event_col='Surv')



# Have a look at the significance of the features

cph.print_summary()
cph.plot()
## I want to see the Survival curve at the patient level.

## Random patients

patients = [4,125,211]



rows_selected = data.iloc[patients, 1:3]

rows_selected
## Lets predict the survival curve for the selected patients. 

## Patients can be identified with the help of the number mentioned against each curve.

cph.predict_survival_function(rows_selected).plot()