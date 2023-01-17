import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, ttest_ind, chisquare, chi2_contingency, chi2
Energy_Census_GDP = pd.read_csv("../input/Energy Census and Economic Data US 2010-2014.csv")

Energy_Census_GDP.head()
Energy_Census_GDP.describe(exclude = ['O'])
Energy_Census_GDP.describe(include = ['O'])
def ShowMissing(df):

    num_missing = df.isnull().sum().sort_values(ascending=False)

    missing_percent = (100*df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_num_percent = pd.concat([num_missing, missing_percent], axis=1, 

                                    keys=['num_missing', 'missing_percent'])

    return missing_num_percent
missing_num = ShowMissing(Energy_Census_GDP)

missing_num.head(10)
missing_feats = ["Coast", "Region", "RDOMESTICMIG2014", "RDOMESTICMIG2013", 

                "RDOMESTICMIG2012", "RDOMESTICMIG2011", "Division", "Great Lakes"]
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

missing_num.head()
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
num_bins = 5





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(Energy_Census_GDP['TotalC2014'], num_bins, alpha=0.7, 

         label=["TotalC2014"], edgecolor = 'black')

ax0.set_title(" Total energy consumption in billion BTU (2014)")



ax1.hist(Energy_Census_GDP['LPGC2014'], num_bins, alpha=0.7, 

         label=["LPGC2014"], edgecolor = 'black')

ax1.set_title("Liquid natural gas total consumption in billion BTU (2014)")



ax2.hist(Energy_Census_GDP['CoalC2014'], num_bins, alpha=0.7, 

         label=["CoalC2014"], edgecolor = 'black')

ax2.set_title("Coal total consumption in billion BTU (2014)")



ax3.hist(Energy_Census_GDP['HydroC2014'], num_bins, alpha=0.7, 

         label=["HydroC2014"], edgecolor = 'black')

ax3.set_title("Hydro power total consumption in billion BTU (2014)")





plt.show()
num_bins = 10





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(Energy_Census_GDP['RBIRTH2014'], num_bins, alpha=0.7, 

         label=["RBIRTH2014"], edgecolor = 'black')

ax0.set_title(" Birth rate in 2014")



ax1.hist(Energy_Census_GDP['RDEATH2014'], num_bins, alpha=0.7, 

         label=["RDEATH2014"], edgecolor = 'black')

ax1.set_title("Death rate in 2014")



ax2.hist(Energy_Census_GDP['RINTERNATIONALMIG2014'], num_bins, alpha=0.7, 

         label=["RINTERNATIONALMIG2014"], edgecolor = 'black')

ax2.set_title("Net international migration rate in 2014")



ax3.hist(Energy_Census_GDP['RDOMESTICMIG2014'], num_bins, alpha=0.7, 

         label=["RDOMESTICMIG2014"], edgecolor = 'black')

ax3.set_title("Net migration rate in 2014")





plt.show()
Energy_Census_GDP['Coast'] = Energy_Census_GDP['Coast'].apply(str)

Energy_Census_GDP_ttest = Energy_Census_GDP[['Coast', 'TotalC2014', 'RINTERNATIONALMIG2014']]
Energy_Census_GDP_ttest.head()
EC_coastal = Energy_Census_GDP_ttest[Energy_Census_GDP_ttest['Coast'] == "1.0" ]['TotalC2014']

EC_ncoastal = Energy_Census_GDP_ttest[Energy_Census_GDP_ttest['Coast'] == "0.0" ]['TotalC2014']

print("Energy consumption standard deviation for coastal is {:.3f} and std for non costal is {:.3f}".

     format(EC_coastal .std(), EC_ncoastal.std()))
print("Mean of energy consuption for costal is {:.0f} billion BTU and mean for non costal {:.0f} billion BTU"

      .format(EC_coastal.mean(), EC_ncoastal.mean()))
print("Sample size of the coastal is {:.0f} and non coastal is {:.0f}".

     format(len(EC_coastal), len(EC_ncoastal)))
ttest_ind(EC_coastal, EC_ncoastal, axis=0, equal_var=False)
IER_coastal = Energy_Census_GDP_ttest[Energy_Census_GDP_ttest['Coast'] == "1.0" ]['RINTERNATIONALMIG2014']

IER_ncoastal = Energy_Census_GDP_ttest[Energy_Census_GDP_ttest['Coast'] == "0.0" ]['RINTERNATIONALMIG2014']
print("International imigration rate standard deviation for coastal states is {:.3f}".

     format(EC_coastal.std()))

print('------------------------------------------')

print("International imigration rate standard deviation for non costal states is {:.3f}".

     format(EC_ncoastal.std()))
print("International imigration rate mean for coastal states is {:.3f}".

     format(EC_coastal.mean()))

print('------------------------------------------')

print("International imigration rate mean for non costal states is {:.3f}".

     format(EC_ncoastal.mean()))
print("Sample size of the coastal is {:.0f} and non coastal is {:.0f}".

     format(len(IER_coastal), len(IER_ncoastal)))
ttest_ind(IER_coastal, IER_ncoastal, axis=0, equal_var=False)
Energy_features = ['TotalC2014',  'TotalP2014', 'TotalE2014', 'TotalPrice2014']

Energy_CPEP = Energy_Census_GDP[Energy_features]



colormap = plt.cm.viridis

plt.figure(figsize = (8,8))

plt.title('Correlation between energy features', y=1.05, size = 20)

sns.heatmap(Energy_CPEP.corr(),

            linewidths=0.1, 

            center = 0,

            vmin = -1,

            vmax = 1, 

            annot = True,

            square = True, 

            fmt ='.2f', 

            annot_kws = {'size': 10},

            cmap = colormap, 

            linecolor ='white');

Census_features = [ 'CENSUS2010POP', 'POPESTIMATE2014', 'RBIRTH2014', 'RDEATH2014',

                   'RNATURALINC2014', 'RINTERNATIONALMIG2014', 'RDOMESTICMIG2014', 'RNETMIG2014']

Census_num = Energy_Census_GDP[Census_features]

colormap = plt.cm.viridis

plt.figure(figsize = (16,16))

plt.title('Correlation between census features', y=1.05, size = 20)

sns.heatmap(Census_num.corr(),

            linewidths=0.1, 

            center = 0,

            vmin = -1,

            vmax = 1, 

            annot = True,

            square = True, 

            fmt ='.2f', 

            annot_kws = {'size': 10},

            cmap = colormap, 

            linecolor ='white');

Energy_Census_GDP[Census_features].max()


Census_feat_ex2 =  [ 'RBIRTH2014', 'RDEATH2014','RNATURALINC2014', 

                    'RINTERNATIONALMIG2014', 'RDOMESTICMIG2014', 'RNETMIG2014']

sns.set()

sns.pairplot(Energy_Census_GDP[Census_features], size = 2.5)

plt.show();
GDP_features =['GDP2014Q1', 'GDP2014Q2',

       'GDP2014Q3', 'GDP2014Q4', 'GDP2014', 'CENSUS2010POP']

GDP_num = Energy_Census_GDP[GDP_features]

colormap = plt.cm.viridis

plt.figure(figsize = (16,16))

plt.title('Correlation between GDP features', y=1.05, size = 20)

sns.heatmap(GDP_num.corr(),

            linewidths=0.1, 

            center = 0,

            vmin = -1,

            vmax = 1, 

            annot = True,

            square = True, 

            fmt ='.2f', 

            annot_kws = {'size': 10},

            cmap = colormap, 

            linecolor ='white');

num_features = ['TotalP2014',  'TotalPrice2014', 'RBIRTH2014', 'RDEATH2014',

                   'RNATURALINC2014', 'RINTERNATIONALMIG2014',  'RNETMIG2014', 'GDP2014']

ECG_num = Energy_Census_GDP[num_features]

colormap = plt.cm.viridis

plt.figure(figsize = (16,16))

plt.title('Correlation between features', y=1.05, size = 20)

sns.heatmap(ECG_num.corr(),

            linewidths=0.1, 

            center = 0,

            vmin = -1,

            vmax = 1, 

            annot = True,

            square = True, 

            fmt ='.2f', 

            annot_kws = {'size': 10},

            cmap = colormap, 

            linecolor ='white');
Energy_Census_GDP.describe(include = ['O'])
Div_states = Energy_Census_GDP['Division'].unique()

state_count = []

for stc in Div_states:

    state_count.append(Energy_Census_GDP[Energy_Census_GDP['Division'] == stc]['Division'].count())
div_list = ['New England', 'Middle Atlantic', 'East North Central', 

             'West North Central', 'South Atlantic', 'East South Central', 

             'West South Central', 'Mountain', 'Pacific']



x = np.arange(len(Div_states)-1)

plt.figure(figsize=(10,8))

plt.bar(x, state_count[0:9], align='center', alpha=0.5, edgecolor = 'black')

xticks_pos = np.arange(len(Div_states))-0.4

plt.xticks(xticks_pos, div_list, rotation = 60, fontsize=18)

plt.ylabel('Number of states in each division', fontsize=18 )

plt.title('Regional divisions of the US ', fontsize=18)

 

plt.show()
explode = (0, 0, 0, 0, 0, 0.2, 0, 0, 0)  # explode 1st slice

plt.figure(figsize=(8,8))

plt.rcParams['font.size'] = 18

plt.pie(state_count[0:9], 

        explode=explode, 

        labels=div_list, 

        autopct='%1.1f%%', 

        shadow=True, 

        startangle = 30 )

 

plt.axis('equal')

plt.tight_layout()

plt.show()
Energy_Census_GDP['GL']= Energy_Census_GDP['Great Lakes']



Census_tab1 = Energy_Census_GDP[Energy_Census_GDP['Division'] != 'nan']

Census_tab2 = Energy_Census_GDP[Energy_Census_GDP['GL'] != 'nan']
Census_tab = pd.crosstab(Census_tab1.Division, Census_tab2.GL, margins = True)

Census_tab.columns =  ['Lake', 'noLeak', 'total_rows']

Census_tab.index  = ['New England', 'Middle Atlantic', 'East North Central', 

             'West North Central', 'South Atlantic', 'East South Central', 

             'West South Central', 'Mountain', 'Pacific', 'total']

Census_tab
Census_Observed = Census_tab.iloc[0:9, 0:2]

chi2_contingency(observed = Census_Observed)
# Find the critical value for 95% confidence where  df = number of categories  - 1

# the df is calculated (9-1)*(2-1) = 8

critical = chi2.ppf(q = 0.95, df = 8)

print()

print("Critical value {:.3f}".format(critical))