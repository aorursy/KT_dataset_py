# Import required modules

import matplotlib.pyplot as plt

import missingno as msno

import numpy as np

import os

import pandas as pd

import seaborn as sns

import warnings



from IPython.display import display

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from tqdm import tqdm_notebook as tqdm



sns.set()

warnings.filterwarnings('ignore')



all_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        all_files.append(os.path.join(dirname, filename))

all_files
# WDI country index

ctry = pd.read_csv('/kaggle/input/world-development-indicators/wdi-csv-zip-57-mb-/WDICountry.csv')



# Filter data

ctry = ctry[['Short Name', 'Country Code']].rename(columns={'Short Name': 'country'})



# Import train and test data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

train = train.rename(columns={'Country_Region': 'ctry'})



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

test = test.rename(columns={'Country_Region': 'ctry'})



# Standardise country name

ctry['country'] = ctry.country.str.lower()

train['ctry'] = train.ctry.str.lower()

test['ctry'] = train.ctry.str.lower()
# Prepare identifer

ctry['ctry'] = ctry.country.copy()



# Manually replace names

ctry['ctry'].loc[ctry.country.str.contains('baha')] = 'bahamas'

ctry['ctry'].loc[ctry.country.str.contains('dem. rep. congo')] = 'congo (kinshasa)'

ctry['ctry'].loc[ctry.country=='congo'] = 'congo (brazzaville)'

ctry['ctry'].loc[ctry.country.str.contains("côte d'ivoire")] = "cote d'ivoire"

ctry['ctry'].loc[ctry.country.str.contains('czech republic')] = 'czechia'

ctry['ctry'].loc[ctry.country.str.contains('the gambia')] = 'gambia'

ctry['ctry'].loc[ctry.country == 'korea'] = 'korea, south'

ctry['ctry'].loc[ctry.country.str.contains('kyrgyz republic')] = 'kyrgyzstan'

ctry['ctry'].loc[ctry.country.str.contains('lao pdr')] = 'laos'

ctry['ctry'].loc[ctry.country.str.contains('st. kitts and nevis')] = 'saint kitts and nevis'

ctry['ctry'].loc[ctry.country.str.contains('st. lucia')] = 'saint lucia'

ctry['ctry'].loc[ctry.country.str.contains('st. vincent and the grenadines')] = 'saint vincent and the grenadines'

ctry['ctry'].loc[ctry.country.str.contains('slovak republic')] = 'slovakia'

ctry['ctry'].loc[ctry.country.str.contains('syrian arab republic')] = 'syria'

ctry['ctry'].loc[ctry.country.str.contains('united states')] = 'us'



# Filter dataset

ctry = ctry[ctry.ctry.isin(train.ctry)].reset_index(drop=True)
# Load data

iter_csv = pd.read_csv('/kaggle/input/world-development-indicators/wdi-csv-zip-57-mb-/WDIData.csv', iterator=True, chunksize=10000)

df = pd.concat([chunk[chunk['Country Code'].isin(ctry['Country Code'])] for chunk in iter_csv])
# Copy data

df2 = df.copy()



# Remove rows with absolutely no data

df2['missing'] = df2[[str(x) for x in np.arange(1960, 2019)]].isnull().sum(axis=1)

df2 = df2[~(df2.missing == 59)]



# Identify features without missing rows

selected_feats = df2['Indicator Name'].value_counts().index[df2['Indicator Name'].value_counts() >= 165]
# Manual adjustment

selected_feats = [

    'Access to clean fuels and technologies for cooking (% of population)',

    'Access to electricity (% of population)',

    'Access to electricity, rural (% of rural population)',

    'Access to electricity, urban (% of urban population)',

    'Adjusted savings: carbon dioxide damage (% of GNI)',

    'Adjusted savings: consumption of fixed capital (% of GNI)',

    'Adjusted savings: education expenditure (% of GNI)',

    'Adjusted savings: energy depletion (% of GNI)',

    'Adjusted savings: mineral depletion (% of GNI)',

    'Agricultural land (% of land area)',

    'Agricultural methane emissions (% of total)',

    'Agricultural nitrous oxide emissions (% of total)',

    'Agricultural raw materials exports (% of merchandise exports)',

    'Agricultural raw materials imports (% of merchandise imports)',

    'Agriculture, forestry, and fishing, value added (% of GDP)',

    'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)',

    'Arable land (% of land area)',

    'Arable land (hectares per person)',

    'Average precipitation in depth (mm per year)',

    'Birth rate, crude (per 1,000 people)',

    'Business extent of disclosure index (0=less disclosure to 10=more disclosure)',

    'CO2 emissions (metric tons per capita)',

    'CO2 emissions from gaseous fuel consumption (% of total)',

    'CO2 emissions from liquid fuel consumption (% of total)',

    'CO2 emissions from solid fuel consumption (% of total)',

    'Coal rents (% of GDP)',

    'Cost of business start-up procedures (% of GNI per capita)',

    'Crop production index (2004-2006 = 100)',

    'Current health expenditure (% of GDP)',

    'Current health expenditure per capita (current US$)',

    'Death rate, crude (per 1,000 people)',

    'Depth of credit information index (0=low to 8=high)',

    'Diabetes prevalence (% of population ages 20 to 79)',

    'Distance to frontier score (0=lowest performance to 100=frontier)',

    'Domestic general government health expenditure (% of GDP)',

    'Domestic general government health expenditure (% of current health expenditure)',

    'Domestic general government health expenditure (% of general government expenditure)',

    'Domestic general government health expenditure per capita (current US$)',

    'Domestic private health expenditure (% of current health expenditure)',

    'Domestic private health expenditure per capita (current US$)',

    'Ease of doing business index (1=most business-friendly regulations)',

    'Energy related methane emissions (% of total)',

    'Export unit value index (2000 = 100)',

    'Export value index (2000 = 100)',

    'Export volume index (2000 = 100)',

    'Exports of goods and services (% of GDP)',

    'External balance on goods and services (% of GDP)',

    'Fertility rate, total (births per woman)',

    'Fixed broadband subscriptions (per 100 people)',

    'Fixed telephone subscriptions (per 100 people)',

    'Food exports (% of merchandise exports)',

    'Food imports (% of merchandise imports)',

    'Food production index (2004-2006 = 100)',

    'Foreign direct investment, net inflows (% of GDP)',

    'Forest area (% of land area)',

    'Forest rents (% of GDP)',

    'Fuel exports (% of merchandise exports)',

    'Fuel imports (% of merchandise imports)',

    'GDP deflator (base year varies by country)',

    'GDP growth (annual %)',

    'GDP per capita (current US$)',

    'GDP per capita growth (annual %)',

    'GNI per capita, Atlas method (current US$)',

    'Government expenditure on education, total (% of GDP)',

    'Hospital beds (per 1,000 people)',

    'Immunization, DPT (% of children ages 12-23 months)',

    'Immunization, measles (% of children ages 12-23 months)',

    'Import unit value index (2000 = 100)',

    'Import value index (2000 = 100)',

    'Import volume index (2000 = 100)',

    'Imports of goods and services (% of GDP)',

    'Incidence of tuberculosis (per 100,000 people)',

    'Individuals using the Internet (% of population)',

    'Industry (including construction), value added (% of GDP)',

    'Inflation, GDP deflator (annual %)',

    'Intentional homicides (per 100,000 people)',

    'International migrant stock (% of population)',

    'International tourism, number of arrivals',

    'International tourism, receipts (current US$)',

    'Labor force participation rate for ages 15-24, total (%) (national estimate)',

    'Labor force participation rate, total (% of total population ages 15+) (national estimate)',

    'Labor tax and contributions (% of commercial profits)',

    'Law mandates nondiscrimination based on gender in hiring (1=yes; 0=no)',

    'Law mandates paid or unpaid maternity leave (1=yes; 0=no)',

    'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',

    'Life expectancy at birth, total (years)',

    'Livestock production index (2004-2006 = 100)',

    'Low-birthweight babies (% of births)',

    'Lower secondary school starting age (years)',

    'Manufactures exports (% of merchandise exports)',

    'Manufactures imports (% of merchandise imports)',

    'Manufacturing, value added (% of GDP)',

    'Merchandise exports by the reporting economy, residual (% of total merchandise exports)',

    'Merchandise exports to economies in the Arab World (% of total merchandise exports)',

    'Merchandise exports to high-income economies (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies in Europe & Central Asia (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies in Latin America & the Caribbean (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies in Middle East & North Africa (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies in South Asia (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies in Sub-Saharan Africa (% of total merchandise exports)',

    'Merchandise exports to low- and middle-income economies outside region (% of total merchandise exports)',

    'Merchandise imports by the reporting economy, residual (% of total merchandise imports)',

    'Merchandise imports from economies in the Arab World (% of total merchandise imports)',

    'Merchandise imports from high-income economies (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies in East Asia & Pacific (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies in Europe & Central Asia (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies in Latin America & the Caribbean (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies in Middle East & North Africa (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies in South Asia (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)',

    'Merchandise imports from low- and middle-income economies outside region (% of total merchandise imports)',

    'Merchandise trade (% of GDP)',

    'Methane emissions (% change from 1990)',

    'Mineral rents (% of GDP)',

    'Mobile cellular subscriptions (per 100 people)',

    'Mortality rate, infant (per 1,000 live births)',

    'Mortality rate, neonatal (per 1,000 live births)',

    'Mortality rate, under-5 (per 1,000 live births)',

    'Mothers are guaranteed an equivalent position after maternity leave (1=yes; 0=no)',

    'Natural gas rents (% of GDP)',

    'Net barter terms of trade index (2000 = 100)',

    'Nitrous oxide emissions (% change from 1990)',

    'Nitrous oxide emissions in energy sector (% of total)',

    'Nonpregnant and nonnursing women can do the same jobs as men (1=yes; 0=no)',

    'Number of deaths ages 5-14 years',

    'Number of infant deaths',

    'Number of neonatal deaths',

    'Number of under-five deaths',

    'Nurses and midwives (per 1,000 people)',

    'Oil rents (% of GDP)',

    'Ores and metals exports (% of merchandise exports)',

    'Ores and metals imports (% of merchandise imports)',

    'Other taxes payable by businesses (% of commercial profits)',

    'Out-of-pocket expenditure (% of current health expenditure)',

    'Out-of-pocket expenditure per capita (current US$)',

    'Over-age students, primary (% of enrollment)',

    'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)',

    'PM2.5 air pollution, population exposed to levels exceeding WHO guideline value (% of total)',

    'People practicing open defecation (% of population)',

    'People using at least basic drinking water services (% of population)',

    'People using at least basic sanitation services (% of population)',

    'Persistence to last grade of primary, total (% of cohort)',

    'Physicians (per 1,000 people)',

    'Population density (people per sq. km of land area)',

    'Population growth (annual %)',

    'Population, total',

    'Preprimary education, duration (years)',

    'Prevalence of anemia among children (% of children under 5)',

    'Prevalence of anemia among non-pregnant women (% of women ages 15-49)',

    'Prevalence of anemia among pregnant women (%)',

    'Prevalence of anemia among women of reproductive age (% of women ages 15-49)',

    'Primary education, duration (years)',

    'Primary education, pupils (% female)',

    'Primary education, teachers (% female)',

    'Primary school starting age (years)',

    'Private credit bureau coverage (% of adults)',

    'Probability of dying at age 5-14 years (per 1,000 children age 5)',

    'Profit tax (% of commercial profits)',

    'Proportion of seats held by women in national parliaments (%)',

    'Public credit registry coverage (% of adults)',

    'Pupil-teacher ratio, preprimary',

    'Pupil-teacher ratio, primary',

    'Pupil-teacher ratio, secondary',

    'Pupil-teacher ratio, tertiary',

    'Ratio of female to male labor force participation rate (%) (national estimate)',

    'Renewable electricity output (% of total electricity output)',

    'Renewable energy consumption (% of total final energy consumption)',

    'Renewable internal freshwater resources per capita (cubic meters)',

    'Renewable internal freshwater resources, total (billion cubic meters)',

    'Repeaters, primary, total (% of total enrollment)',

    'Rural population (% of total population)',

    'Rural population growth (annual %)',

    'School enrollment, preprimary (% gross)',

    'School enrollment, primary (% gross)',

    'School enrollment, primary, private (% of total primary)',

    'School enrollment, secondary (% gross)',

    'Secondary education, duration (years)',

    'Secure Internet servers (per 1 million people)',

    'Start-up procedures to register a business (number)',

    'Strength of legal rights index (0=weak to 12=strong)',

    'Tariff rate, most favored nation, simple mean, all products (%)',

    'Tariff rate, most favored nation, simple mean, manufactured products (%)',

    'Tariff rate, most favored nation, simple mean, primary products (%)',

    'Terrestrial and marine protected areas (% of total territorial area)',

    'Terrestrial protected areas (% of total land area)',

    'Time required to enforce a contract (days)',

    'Time required to get electricity (days)',

    'Time required to start a business (days)',

    'Time to export, border compliance (hours)',

    'Time to export, documentary compliance (hours)',

    'Time to import, border compliance (hours)',

    'Time to import, documentary compliance (hours)',

    'Time to prepare and pay taxes (hours)',

    'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)',

    'Total natural resources rents (% of GDP)',

    'Total tax and contribution rate (% of profit)',

    'Trade (% of GDP)',

    'Tuberculosis case detection rate (%, all forms)',

    'Tuberculosis treatment success rate (% of new cases)',

    'Urban population (% of total)',

    'Urban population growth (annual %)'

]
# Truncate dataset

df2 = df2[df2['Indicator Name'].isin(selected_feats)]



# Extract year of latest data

df2['latest_year'] = df2.apply(lambda x: np.max(np.arange(1960, 2019)[x[[str(x) for x in np.arange(1960, 2019)]].notnull()]), axis=1)



# Extract data from latest year

df2['value'] = df2.apply(lambda x: x[str(x.latest_year)], axis=1)



# Map data

df2 = df2.merge(ctry.set_index('Country Code'), on='Country Code')
# Prepare data table

df3 = df2[['ctry', 'Indicator Name', 'value']]

df3 = df3.pivot(index='ctry', columns='Indicator Name', values='value')



# Save data

# df3.reset_index().to_csv('wdi_filtered.csv', index=False)
# Group by country and date

train_scaled = train.groupby(['ctry', 'Date'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()



# Merge data

train_scaled = train.merge(df3[['Population, total']], on='ctry')

train_scaled = train_scaled.rename(columns={'Population, total': 'population'})



# Scale data

train_scaled['cc'] = train_scaled.ConfirmedCases / train_scaled.population

train_scaled['ft'] = train_scaled.Fatalities / train_scaled.population
# Extract targets

train_tgts = pd.DataFrame()



for st in tqdm(train_scaled.ctry.unique()):

    

    temp_df = train_scaled[train_scaled.ctry == st]

    temp_df = temp_df[temp_df.ConfirmedCases > 0]

    

    if temp_df.shape[0] < 7:

        continue

    else:

        temp_df['cc07'] = temp_df.cc.iloc[6]

        if temp_df.shape[0] >= 14:

            temp_df['cc14'] = temp_df.cc.iloc[13]

        else:

            temp_df['cc14'] = np.nan

        

        if temp_df.shape[0] >= 21:

            temp_df['cc21'] = temp_df.cc.iloc[20]

        else:

            temp_df['cc21'] = np.nan

        

        if temp_df.shape[0] >= 30:

            temp_df['cc30'] = temp_df.cc.iloc[29]

        else:

            temp_df['cc30'] = np.nan

        

        train_tgts = train_tgts.append(temp_df.iloc[0])
# Select columns

train_tgts = train_tgts.reset_index(drop=True)[['ctry', 'cc07', 'cc14', 'cc21', 'cc30']]



# Merge WDI and COVID-19 data

train_final = train_tgts.merge(df3, on='ctry').drop('Population, total', axis=1)
# Split data

y_df = train_final[['cc07', 'cc14', 'cc21', 'cc30']]

X_df = train_final.drop(['cc07', 'cc14', 'cc21', 'cc30'], axis=1)
plt.figure(figsize=(15,15))

sns.heatmap(train_final.set_index('ctry').isnull(), cmap=sns.color_palette(['darkgrey', '#ffffff']))

plt.yticks(ticks=np.arange(0, train_final.shape[0]), labels=train_final.ctry, fontsize=7)

plt.title('Missing Values', fontsize=13, fontweight='bold')

plt.show()
# Drop countries with too many missing values

drop_ctry = ['liechtenstein', 'monaco', 'andorra', 'san marino', 'somalia', 'cuba', 'montenegro', 'uzbekistan', 'saint kitts and nevis', 'timor-leste', 'eritrea']

y_df = y_df[~X_df.ctry.isin(drop_ctry)]

X_df = X_df[~X_df.ctry.isin(drop_ctry)]
ii = IterativeImputer()

X_cols = X_df.set_index('ctry').columns

X_idx = X_df.ctry

X_df = pd.DataFrame(ii.fit(X_df.set_index('ctry')).transform(X_df.set_index('ctry')), columns=X_cols, index=X_idx)
# Get indicators

ids = pd.read_csv('/kaggle/input/world-development-indicators/wdi-csv-zip-57-mb-/WDISeries.csv')

ids = ids.rename(columns={'Series Code': 'sx', 'Topic': 'topic', 'Indicator Name': 'ind_name'})[['sx', 'topic', 'ind_name']]

ids['category'] = ids.sx.str[:6]



# Extract indicators in dataset

ind_dict = pd.DataFrame(train_final.columns.T[5:], columns=['ind_name'])

ind_dict['sx'] = ind_dict.ind_name.map(ids[['category', 'ind_name']].set_index('ind_name').category)



# Manual replace

ind_dict['sx'].loc[14] = 'NV.AGR'

ind_dict['sx'].loc[73] = 'NV.IND'



# Category

ind_dict['category'] = ind_dict.sx.str[:2]



# Group others

ind_dict['category'].loc[~ind_dict.category.isin(['SH', 'IC', 'TM', 'SE', 'TX', 'NY', 'EN', 'SP', 'AG'])] = 'OTH'
# Feature groups

feat_groups = sorted(ind_dict.category.unique())



# Generate correlation heatmaps

for f in feat_groups:

    feats = ind_dict['ind_name'].loc[ind_dict.category == f]

    temp_df = X_df[feats]

    

    mask = np.tril(np.ones(temp_df.corr().shape)).astype(np.bool)

    corr_df = temp_df.corr().where(mask)

    plt.figure(figsize=(10,10))

    plt.title('Topic: %s' % f)

    sns.heatmap(corr_df, cmap='Blues')

    plt.xticks(rotation=45, ha='right')

    plt.show()
# After inspection, these features will be dropped:

feats_drop = [

    'Tariff rate, most favored nation, simple mean, all products (%)', 'Import volume index (2000 = 100)', 'Export volume index (2000 = 100)',

    'Fertility rate, total (births per woman)', 'Mortality rate, infant (per 1,000 live births)', 'Population growth (annual %)',

    'Prevalence of anemia among children (% of children under 5)', 'Prevalence of anemia among non-pregnant women (% of women ages 15-49)',

    'Prevalence of anemia among women of reproductive age (% of women ages 15-49)', 'Number of deaths ages 5-14 years',

    'Number of neonatal deaths', 'Number of under-five deaths', 'Immunization, measles (% of children ages 12-23 months)',

    'Mortality rate, neonatal (per 1,000 live births)', 'Domestic general government health expenditure per capita (current US$)',

    'Domestic private health expenditure per capita (current US$)', 'Access to electricity (% of population)',

    'Access to electricity, rural (% of rural population)', 'Access to electricity, urban (% of urban population)',

    'GNI per capita, Atlas method (current US$)', 'GDP growth (annual %)',

    'Mineral rents (% of GDP)'

]



# Drop features

X_df = X_df.drop(feats_drop, axis=1)



# Extract indicators in dataset

ind_dict = pd.DataFrame(X_df.columns.T, columns=['ind_name'])

ind_dict['sx'] = ind_dict.ind_name.map(ids[['category', 'ind_name']].set_index('ind_name').category)



# Manual replace

ind_dict['sx'].loc[14] = 'NV.AGR'

ind_dict['sx'].loc[73] = 'NV.IND'



# Category

ind_dict['category'] = ind_dict.sx.str[:2]



# Group others

ind_dict['category'].loc[~ind_dict.category.isin(['SH', 'IC', 'TM', 'SE', 'TX', 'NY', 'EN', 'SP', 'AG'])] = 'OTH'
# Feature groups

feat_groups = sorted(ind_dict.category.unique())



# Generate correlation heatmaps

for f in feat_groups:

    feats = ind_dict['ind_name'].loc[ind_dict.category == f]

    temp_df = X_df[feats]

    

    mask = np.tril(np.ones(temp_df.corr().shape)).astype(np.bool)

    corr_df = temp_df.corr().where(mask)

    plt.figure(figsize=(10,10))

    plt.title('Topic: %s' % f)

    sns.heatmap(corr_df, cmap='Blues')

    plt.xticks(rotation=45, ha='right')

    plt.show()
# Re-merge data

train_final = pd.concat([y_df.reset_index(drop=True), X_df.reset_index(drop=True)], axis=1)



# Plot histograms

train_final.cc07.plot.hist(bins=30, figsize=(10, 6))

plt.title('CC07', fontweight='bold', fontsize=13)

plt.show()



# Plot histograms

train_final.cc14.plot.hist(bins=30, figsize=(10, 6))

plt.title('CC14', fontweight='bold', fontsize=13)

plt.show()



# Plot histograms

train_final.cc21.plot.hist(bins=30, figsize=(10, 6))

plt.title('CC21', fontweight='bold', fontsize=13)

plt.show()



# Plot histograms

train_final.cc30.plot.hist(bins=30, figsize=(10, 6))

plt.title('CC30', fontweight='bold', fontsize=13)

plt.show()
cc07_corrs = train_final.drop(['cc14', 'cc21', 'cc30'], axis=1).corr().iloc[0][1:].sort_values(ascending=False)

cc07_corrs.plot.bar(figsize=(15,8))

plt.ylim(-0.7, 0.7)

plt.show()
train_final.drop(['cc07', 'cc21', 'cc30'], axis=1).corr().iloc[0][1:][cc07_corrs.index].plot.bar(figsize=(15,8))

plt.ylim(-0.7, 0.7)

plt.show()
train_final.drop(['cc07', 'cc14', 'cc30'], axis=1).corr().iloc[0][1:][cc07_corrs.index].plot.bar(figsize=(15,8))

plt.ylim(-0.7, 0.7)

plt.show()
train_final.drop(['cc07', 'cc14', 'cc21'], axis=1).corr().iloc[0][1:][cc07_corrs.index].plot.bar(figsize=(15,8))

plt.ylim(-0.7, 0.7)

plt.show()
import statsmodels.api as sm



from sklearn.feature_selection import RFECV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import make_scorer, mean_absolute_error
# Prepare data

tgt_cols = ['cc07', 'cc14', 'cc21', 'cc30']

X_dat = train_final.drop(tgt_cols, axis=1)

y_dat = train_final.cc07
# Initialise Linreg model

lm = LinearRegression(n_jobs=4)

rfecv_lm = RFECV(lm, min_features_to_select=50, cv=5, scoring=make_scorer(mean_absolute_error), n_jobs=4, verbose=False)

rfecv_lm.fit(X_dat, np.log(y_dat))
# Get selected features

features_selected = X_dat.columns[rfecv_lm.get_support()]



# Add constant

X_dat_const = X_dat[features_selected].copy()

X_dat_const['const'] = 1.0



# Convert y values to %% for readibility

y_dat_plus = np.log(y_dat)



# Estimate LinReg using OLS

linreg = sm.OLS(y_dat_plus, X_dat_const).fit()



# Extract summary

lr_summary = linreg.summary()

res_html = lr_summary.tables[1].as_html()

res_table = pd.read_html(res_html, header=0, index_col=0)[0]
# Display results

display(lr_summary.tables[0])

display(res_table[res_table['P>|t|'] <= 0.05].sort_values('coef', ascending=False))
!pip install pygam

from pygam import LinearGAM, GAM



# Fit GAM

lg = LinearGAM(n_splines=6).fit(X_dat, np.log(y_dat))

lg.summary()
# Plot results

counter = 0

for i, term in enumerate(lg.terms):

    if term.isintercept:

        continue



    XX = lg.generate_X_grid(term=i)

    pdep, confi = lg.partial_dependence(term=i, X=XX, width=0.95)

    

    if np.sum(confi[:, 1] < 0) > 0 or np.sum(confi[:, 0] > 0) > 0:

        plt.figure(figsize=(8,5))

        plt.plot(XX[:, term.feature], pdep)

        plt.plot(XX[:, term.feature], confi, c='r', ls='--')

        plt.fill_between(XX[:, term.feature], y1=confi[:, 0], y2=confi[:, 1], where=(confi[:, 1] < 0), color='red', alpha=0.1)

        plt.fill_between(XX[:, term.feature], y1=confi[:, 0], y2=confi[:, 1], where=(confi[:, 0] > 0), color='green', alpha=0.1)

        plt.title('s(%s) - %s' % (counter, X_dat.columns[counter]), fontweight='bold', fontsize=12)

        plt.show()

    counter+=1