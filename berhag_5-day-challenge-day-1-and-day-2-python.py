import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
Energy_Census_GDP = pd.read_csv("../input/Energy Census and Economic Data US 2010-2014.csv")

Energy_Census_GDP.describe()
Energy_Census_GDP.shape
Energy_Census_GDP.describe(include = ['O'])
Energy_Census_GDP.describe(exclude = ['O'])
Energy_Census_GDP.head(10)
Energy_Census_GDP.select_dtypes(include = ["O"]).columns
def ShowMissing(df):

    num_missing = df.isnull().sum().sort_values(ascending=False)

    missing_percent = (100*df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_num_percent = pd.concat([num_missing, missing_percent], axis=1, 

                                    keys=['num_missing', 'missing_percent'])

    return missing_num_percent
missing_num = ShowMissing(Energy_Census_GDP)

missing_num.head(10)
missing_feats = ["Coast", "Region", "RDOMESTICMIG2014", "RDOMESTICMIG2013", 

                "RDOMESTICMIG2012", "RDOMESTICMIG2011", "Division", "Great Lakes"   ]
Energy_Census_GDP[missing_feats].head(10)
cat_miss_cols = ['Coast', 'Region', 'Division', 'Great Lakes']

for item in cat_miss_cols:

    Energy_Census_GDP[item] = Energy_Census_GDP[item].apply(str)
# Looking at categorical values

def MaxIndex(df, cols):

    df[cols].value_counts().max()

    return df[cols].value_counts().idxmax()

def ImputeMissing(df, cols, value):

    df.loc[df[cols].isnull(), cols] = value
# In this for loop the missing data is replaced using the most frequent data 

# of the categorical feature

for item in cat_miss_cols:

    freq_value = MaxIndex(Energy_Census_GDP, item)

    ImputeMissing(Energy_Census_GDP, item, freq_value)
num_miss_cols = [ "RDOMESTICMIG2014", "RDOMESTICMIG2013", "RDOMESTICMIG2012", "RDOMESTICMIG2011"]

# there are four numerical feature with missing data and the 

# missing values are replaced with mean value

for item in num_miss_cols:

    Energy_Census_GDP[item].fillna(Energy_Census_GDP[item].mean(), inplace = True)

missing_num = ShowMissing(Energy_Census_GDP)

missing_num.head(5)
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

Energy_Census_GDP.describe()


num_bins = 5





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(Energy_Census_GDP['TotalC2014'], num_bins, alpha=0.7, label=["TotalC2014"])

ax0.set_title(" Total energy consumption in billion BTU (2014)")



ax1.hist(Energy_Census_GDP['LPGC2014'], num_bins, alpha=0.7, label=["LPGC2014"])

ax1.set_title("Liquid natural gas total consumption in billion BTU (2014)")



ax2.hist(Energy_Census_GDP['CoalC2014'], num_bins, alpha=0.7, label=["CoalC2014"])

ax2.set_title("Coal total consumption in billion BTU (2014)")



ax3.hist(Energy_Census_GDP['HydroC2014'], num_bins, alpha=0.7, label=["HydroC2014"])

ax3.set_title("Hydro power total consumption in billion BTU (2014)")





plt.show()



num_bins = 10





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(Energy_Census_GDP['RBIRTH2014'], num_bins, alpha=0.7, label=["RBIRTH2014"])

ax0.set_title(" Birth rate in 2014")



ax1.hist(Energy_Census_GDP['RDEATH2014'], num_bins, alpha=0.7, label=["RDEATH2014"])

ax1.set_title("Death rate in 2014")



ax2.hist(Energy_Census_GDP['RINTERNATIONALMIG2014'], num_bins, alpha=0.7, label=["RINTERNATIONALMIG2014"])

ax2.set_title("Net international migration rate in 2014")



ax3.hist(Energy_Census_GDP['RDOMESTICMIG2014'], num_bins, alpha=0.7, label=["RDOMESTICMIG2014"])

ax3.set_title("Net migration rate in 2014")





plt.show()
num_bins = 4





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(Energy_Census_GDP['GDP2014Q1'], num_bins, alpha=0.7, label=["GDP2014Q1"])

ax0.set_title(" The GDP in the first quarter in 2014 (in million USD)")



ax1.hist(Energy_Census_GDP['GDP2014Q2'], num_bins, alpha=0.7, label=["GDP2014Q2"])

ax1.set_title(" The GDP in the second quarter in 2014 (in million USD)")



ax2.hist(Energy_Census_GDP['GDP2014Q3'], num_bins, alpha=0.7, label=["GDP2014Q3"])

ax2.set_title(" The GDP in the third quarter in 2014 (in million USD)")



ax3.hist(Energy_Census_GDP['GDP2014Q4'], num_bins, alpha=0.7, label=["GDP2014Q4"])

ax3.set_title(" The GDP in the fouth quarter in 2014 (in million USD)")





plt.show()
num_bins = 5





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(Energy_Census_GDP['GDP2011'], num_bins, alpha=0.7, label=["GDP2011"])

ax0.set_title(" The average GDP throughout 2011 (in million USD)")



ax1.hist(Energy_Census_GDP['GDP2012'], num_bins, alpha=0.7, label=["GDP2012"])

ax1.set_title(" The average GDP throughout 2012 (in million USD)")



ax2.hist(Energy_Census_GDP['GDP2013'], num_bins, alpha=0.7, label=["GDP2013"])

ax2.set_title(" The average GDP throughout 2013 (in million USD)")



ax3.hist(Energy_Census_GDP['GDP2014'], num_bins, alpha=0.7, label=["GDP2014"])

ax3.set_title(" The average GDP throughout 2014 (in million USD)")





plt.show()