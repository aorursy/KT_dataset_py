import numpy as np

import pandas as pd 

from pandas import read_csv

bmi_data = read_csv("../input/eating-health-module-dataset//ehresp_2014.csv")

# remover rows where the reported BMI is less than 0

bmi_data = bmi_data[bmi_data['erbmi'] > 0]

nyc_census = read_csv("../input/new-york-city-census-data/nyc_census_tracts.csv")
bmi_data.head()
import statsmodels.api as sm



X = ['euexfreq' , 'euwgt' ,'euhgt' , 'ertpreat']

# bmi_data[X]



gamma_model = sm.GLM(bmi_data['erbmi'], bmi_data[X], family=sm.families.Gamma())

gamma_results = gamma_model.fit()

print(gamma_results.summary())
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt

erbmi_model = ols("erbmi ~ euexfreq + euwgt + euhgt + ertpreat", data=bmi_data).fit()

print(erbmi_model.summary())
fig, ax = plt.subplots(figsize=(12,8))

fig = sm.graphics.influence_plot(erbmi_model, ax=ax, criterion="cooks")
# examine our model

summary(model)
# added-variable plots for our model

avPlots(model)
# your work goes here! :)