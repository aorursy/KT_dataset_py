#install lmfit
!pip install lmfit
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import LorentzianModel
all_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
key = "Country/Region"
df = all_df[all_df[key]=='Indonesia']

# discard unnecessary column & transpose data
df = df.drop("Province/State", 1).drop("Country/Region", 1).drop('Lat', 1).drop('Long', 1).T

# delta only
df.columns = ["cases"]
df["new_cases"] = df.cases - df.cases.shift()

df = df.drop("cases", 1)
df = df[df["new_cases"]>0]
df
#plot
df.plot(kind='bar',y='new_cases')
# reset index
df.reset_index(level=0, inplace=True)
df.reset_index(level=0, inplace=True)
df.columns = ["index", "date", "new_cases"]
df
# curve fitting
model = LorentzianModel()
params = model.guess(df['new_cases'], x=df['index'])

result = model.fit(df['new_cases'], params, x=df['index'])

result.plot_fit()
plt.show()
