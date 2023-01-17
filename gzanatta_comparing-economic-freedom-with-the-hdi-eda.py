import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
path = "../input/economic-freedom/efw_cc.csv"
data = pd.read_csv(path)
print('Dimensions:',data.shape)
data.head(8)
data.describe()
data.info()
# How many null values does the columns have?
data.isnull().sum()
data = data.loc[:, (data.isnull().sum(axis=0) <= 1242)]
# Rename the columns for a better undestanding
data.rename(columns={"year": "YEAR",
                     "ISO_code": "ISO_CODE",
                     "countries": "COUNTRY",
                     "rank" :"RANK",
                     "quartile": "QUARTILE",
                     "ECONOMIC FREEDOM": "SCORE",
                     "1a_government_consumption": "GOV_CONSUMPTION",
                     "1b_transfers": "TRANSFERS",
                     "1c_gov_enterprises": "GOV_ENTERPRISES",
                     "1d_top_marg_tax_rate": "TOP_MARG_TAX_RATE",
                     "1_size_government": "GOV_SIZE",
                     "2b_impartial_courts": "IMPARTIAL_COURTS", 
                     "2c_protection_property_rights": "PROTEC_PROP_RIGHTS",
                     "2d_military_interference": "MILITARY_INTERF",
                     "2e_integrity_legal_system": "INTEGRITY_LEGAL_SYST",
                     "2j_gender_adjustment": "GENDER_ADJUSTMENT",
                     "2_property_rights": "PROPERTY_RIGHTS",
                     "3a_money_growth": "MONEY_GROWTH",
                     "3b_std_inflation": "STD_INFLATION",
                     "3c_inflation": "INFLATION",
                     "3d_freedom_own_foreign_currency": "FOREIGN_CURRENCY",
                     "3_sound_money": "SOUND_MONEY",
                     "4a_tariffs": "TARIFFS",
                     "4c_black_market": "BLACK_MARKET",
                     "4d_control_movement_capital_ppl": "CONTROL_MOVEMENT",
                     "4_trade": "TRADE",
                     "5a_credit_market_reg": "CREDIT_MARKET_REG",
                     "5b_labor_market_reg": "LABOR_MARKET_REG",
                     "5_regulation": "REGULATION"}, inplace=True)
# First I'm going to use 'ffill' method to fill the quartile column. It has to be an integer.
data.QUARTILE = data.QUARTILE.fillna(method='ffill')

# Then separete the numeric values to fill the missing spaces.
num_names = data._get_numeric_data().columns

data[num_names] = data.groupby('ISO_CODE')[num_names].transform(lambda x: x.fillna(x.median()))
data.isnull().sum()
data.QUARTILE = data.QUARTILE.astype('object')

data[['TRANSFERS','GOV_ENTERPRISES','PROTEC_PROP_RIGHTS','INTEGRITY_LEGAL_SYST','TARIFFS','BLACK_MARKET']] = data.groupby('QUARTILE')\
    [['TRANSFERS','GOV_ENTERPRISES','PROTEC_PROP_RIGHTS','INTEGRITY_LEGAL_SYST','TARIFFS','BLACK_MARKET']].transform(lambda x: x.fillna(x.median()))
# Numeric Value
data_num = data._get_numeric_data()
data_cor = data_num.corr()

#Plot heatmap
sns.set(font_scale=1.4)

plt.figure(figsize=(13,13))
sns.heatmap(data_cor,  square=True, cmap='coolwarm_r')
# Main Features
data_num_2 = data.loc[:,['SCORE', 'GOV_SIZE', 'PROPERTY_RIGHTS', 'SOUND_MONEY', 'TRADE', 'REGULATION']]
data_cor_2 = data_num_2.corr()

sns.set(font_scale=1.4)

plt.figure(figsize=(12,12))
sns.heatmap(data_cor_2,  square=True, annot=True, cmap='coolwarm_r')
sns.set_palette(sns.dark_palette("red",15, reverse=False))
sns.set_style('whitegrid')

top_15_16_least = data[data.YEAR==2016].sort_values(by='SCORE', ascending=False).tail(15)
top_15_16_least.plot('COUNTRY', 'SCORE', kind='bar', figsize=(14,8), rot=45)

plt.xlabel('COUNTRIES')
plt.ylabel('SCORE')
plt.title('Least 15 Economically Free Countries in 2016')
sns.set_palette(sns.dark_palette("green",15, reverse=False))
sns.set_style('whitegrid')

top_15_2016 = data[data.YEAR==2016].sort_values(by='SCORE', ascending=False).head(15)
top_15_2016.plot('COUNTRY', 'SCORE', kind='bar', figsize=(14,8), rot=45)

plt.xlabel('COUNTRIES')
plt.ylabel('SCORE')
plt.title('Top 10 Most Economically Free Countries in 2016')
names = top_15_2016['COUNTRY']
top_15 = data.loc[data['COUNTRY'].isin(names)]

sns.set_palette(sns.color_palette("colorblind",15))
sns.set_style('whitegrid')

fig, ax = plt.subplots()

for key, grp in top_15.groupby(['COUNTRY']):
    ax = grp.plot(ax=ax, kind='line', x='YEAR', y='SCORE', label=key, figsize=(20,10), linewidth=2.5)
    
plt.xlim((1970, 2016))
plt.xlabel('YEAR')
plt.ylabel('SCORE')
plt.title('SCORE BETWEEN 1970 AND 2016')
briccs_names = ['Brazil', 'Russia', 'India', 'China', 'Chile', 'South Africa']
briccs = data.loc[data['COUNTRY'].isin(briccs_names)]

sns.set_palette(sns.color_palette("bright",6))
sns.set_style('whitegrid')

fig, ax = plt.subplots()

for key, grp in briccs.groupby(['COUNTRY']):
    ax = grp.plot(ax=ax, kind='line', x='YEAR', y='SCORE', label=key, figsize=(18,10), linewidth=2.5)
    
plt.xlim((1970, 2016))
plt.legend(loc='lower right')
plt.xlabel('YEAR')
plt.ylabel('SCORE')
plt.title('BRICCS SCORE BETWEEN 1970 AND 2016')
# Separate values from 1970
briccs_1970 = briccs.loc[briccs['YEAR'] == 1970]
main_feat = ['SCORE','GOV_SIZE', 'PROPERTY_RIGHTS', 'SOUND_MONEY', 'TRADE', 'REGULATION']

sns.set(font_scale=1.4)
sns.set_style('whitegrid')
briccs_1970.plot(x='COUNTRY', y=main_feat, kind='bar', rot= 0,figsize=(16,10))
plt.ylim(0,11)
plt.legend(loc='upper left')
plt.xlabel("BRIC'C'S COUNTRIES")
plt.ylabel("SCORE")
plt.title("Main Features For BRIC'C'S COUNTRIES IN 1970")

########################################################################################################################

# Separate values from 2016
briccs_2016 = briccs.loc[briccs['YEAR'] == 2016]

briccs_2016.plot(x='COUNTRY', y=main_feat, kind='bar', rot= 0,figsize=(16,10))
plt.ylim(0,11)
plt.xlabel("BRIC'C'S COUNTRIES")
plt.ylabel("SCORE")
plt.title("Main Features For BRIC'C'S COUNTRIES IN 2016")
fig = plt.gcf()
fig.set_size_inches(16, 10)
sns.set(font_scale=1.4)

data.QUARTILE = data.QUARTILE.astype('int64')

sns.scatterplot(x='PROPERTY_RIGHTS', y='SCORE', data=data, s=45,\
                hue='QUARTILE', palette=["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"])
plt.xlabel('PROPERTY RIGHTS')
plt.ylabel('SCORE')
plt.title('RELATION BETWEEN SCORE AND PROPERTY RIGHTS')
fig = plt.gcf()
fig.set_size_inches(16, 10)
sns.set(font_scale=1.4)

sns.scatterplot(x='TRADE', y='SCORE', data=data, s=45,\
                hue='QUARTILE', palette=["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"])
plt.xlabel('TRADE')
plt.ylabel('SCORE')
plt.title('RELATION BETWEEN SCORE AND TRADE')
path_2 = "../input/human-development-index/Human Development Index.csv"
hdi = pd.read_csv(path_2)
print('Dimensions:',hdi.shape)
hdi.head(10)
# We're going to reshape it in 3 columns
hdi.info()
# First I'm going to drop the Rank column, that won't be needed
hdi = hdi.drop(columns='HDI Rank')

# Then reshape the df in 3 columns
hdi = pd.melt(frame=hdi, id_vars='Country')

hdi.rename(columns={'Country': 'COUNTRY',
                   'variable': 'YEAR',
                   'value': 'INDEX'}, inplace=True)
hdi.YEAR = hdi.YEAR.astype('float')
hdi.head()
hdi[['INDEX']] = hdi.groupby('COUNTRY')[['INDEX']].transform(lambda x: x.fillna(x.median())) 

hdi.isnull().sum()
hdi['new'] = hdi['COUNTRY'].str.split(',').str[0]
hdi['COUNTRY'] = hdi['new'].str.split('(').str[0]

hdi = hdi.drop(columns='new')
hdi.COUNTRY = hdi.COUNTRY.str.strip()

hdi['COUNTRY'] = hdi['COUNTRY'].replace({'Russian Federation': 'Russia'})
sns.set_palette(sns.dark_palette("blue",15, reverse=False))
sns.set_style('whitegrid')

hdi_15_17 = hdi[hdi.YEAR==2016].sort_values(by='INDEX', ascending=False).head(15)
hdi_15_17.plot('COUNTRY', 'INDEX', kind='bar', figsize=(14,8), rot=45, legend=None)

plt.xlabel('COUNTRIES')
plt.ylabel('HDI')
plt.title('Top 15 Countries in Human Development Index in 2016')
hdi_2016 = hdi[hdi.YEAR==2016]
hdi_1990 = hdi[hdi.YEAR==1990] 

sns.set_style('whitegrid')

fig = plt.gcf()
fig.set_size_inches(16, 10)

sns.kdeplot(hdi_1990.INDEX, shade=True, color= "orange", legend= None)
sns.kdeplot(hdi_2016.INDEX, shade=True, color= "blue", legend= None)

plt.xlabel('HDI')
plt.title('HDI DISTRIBUTION OF 1990 AND 2016 ')
hdi = hdi[hdi.YEAR != 2017]

# Get only the main columns of the Economic data
econ = data[['YEAR', 'COUNTRY', 'SCORE','QUARTILE','GOV_SIZE', 'PROPERTY_RIGHTS', 'SOUND_MONEY', 'TRADE', 'REGULATION']]

# And then merge both data on Country and Year
hdi_econ = hdi.merge(econ, how='left', on=['COUNTRY', 'YEAR'])
hdi_econ.head()
print('Dimensions:',hdi_econ.shape)
hdi_econ = hdi_econ.dropna()
hdi_econ.describe()
hdi_econ_num = hdi_econ.drop(['COUNTRY', 'YEAR'], axis=1)
hdi_econ_cor = hdi_econ_num.corr()

sns.set(font_scale=1.4)

plt.figure(figsize=(12,12))
sns.heatmap(hdi_econ_cor,  square=True, annot=True, cmap='BrBG')
fig = plt.gcf()
fig.set_size_inches(16, 10)
sns.set(font_scale=1.4)

sns.scatterplot(x='SCORE', y='INDEX', hue='QUARTILE',
                data=hdi_econ, s=45, palette=["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"])

plt.ylabel('HDI')
plt.title('RELATION BETWEEN HDI AND ECONOMIC FREEDOM')
briccs_names = ['Brazil', 'Russia', 'India', 'China', 'Chile', 'South Africa']
briccs_hdi = hdi_econ.loc[hdi_econ['COUNTRY'].isin(briccs_names)]

sns.set_palette(sns.color_palette("bright",6))
sns.set_style('whitegrid')

fig, ax = plt.subplots()

for key, grp in briccs_hdi.groupby(['COUNTRY']):
    ax = grp.plot(ax=ax, kind='line', x='YEAR', y='INDEX', label=key, figsize=(16,9), linewidth=2.5)
    
plt.xlim((1990, 2016))
plt.legend(loc='lower right')
plt.xlabel('YEAR')
plt.ylabel('INDEX')
plt.title('BRICCS HDI BETWEEN 1990 AND 2016')