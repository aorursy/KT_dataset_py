import numpy as np

import pandas as pd

import seaborn as sns

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import sqlite3
import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package



wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="worldbank_wdi")
bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")

bq_assistant.list_tables()
query_str = """SELECT country_code, year, indicator_code, indicator_value from `patents-public-data.worldbank_wdi.wdi_2016` 

               WHERE year BETWEEN 1960 AND 2015 AND 

               indicator_code IN ('SL.UEM.TOTL.NE.ZS','FP.CPI.TOTL.ZG',

                                     'IC.REG.DURS','GC.TAX.TOTL.GD.ZS','SI.POV.GINI') 

                AND indicator_value<>0

            """
bq_assistant.estimate_query_size(query_str)
wdi_df = wdi.query_to_pandas_safe(query_str)
wdi_df.groupby('indicator_code').count()['indicator_value']
wdi_df.head()
wdi_df.shape
wdi_df_piv = wdi_df.pivot_table(index=['country_code','year'], 

                                columns=['indicator_code'], 

                                values=['indicator_value'], fill_value=np.nan).reset_index()
wdi_df_piv.shape
wdi_df_piv.head()
wdi_df_piv.columns = ['country_code','year'] + list(wdi_df_piv.columns.droplevel())[2:]
wdi_df_piv.head()
wdi_df_piv.columns
wdi_df_mod = wdi_df_piv[['country_code','SI.POV.GINI','year',                                              

                        'FP.CPI.TOTL.ZG', 'GC.TAX.TOTL.GD.ZS',

                        'IC.REG.DURS',

                        'SL.UEM.TOTL.NE.ZS']]
wdi_df_mod.columns = ['CountryCode','Gini','Year', 

                      'Inflat', 'TaxRev', 'BusDay', 'Unempl']
wdi_df_mod.head()
wdi_df_mod.describe()
wdi_df_mod.groupby(['Year']).count().T
sns.set(rc={'figure.figsize':(11.7,8.27)})
wdi_corr = wdi_df_mod.iloc[:,1:].corr()

mask = np.zeros(wdi_corr.shape, dtype=bool)

mask[np.tril_indices(len(mask))] = True

sns.heatmap(wdi_corr, annot = True, mask = mask);
sns.pairplot(wdi_df_mod);
wdi_df_mod['Gini'].mean()
sns.distplot(wdi_df_mod['Gini'].dropna());
sns.distplot(np.log(wdi_df_mod['Gini'].dropna()));
wdi_df_clean = wdi_df_mod.dropna()
from sklearn.preprocessing import RobustScaler

rob_sc = RobustScaler()
Gini_log = np.log(wdi_df_clean['Gini'])

X_sc = rob_sc.fit_transform(wdi_df_clean.iloc[:,2:])
wdi_sc = pd.concat([wdi_df_clean.iloc[:, 0],

                    wdi_df_clean.iloc[:, 2], 

                    Gini_log,

                    pd.DataFrame(X_sc, 

                                index=wdi_df_clean.index,

                                columns = [x + '_sc' for x in wdi_df_clean.iloc[:,2:].columns])],

                    axis=1)
wdi_df_clean.head()
wdi_sc.head()
wdi_sc.shape
wdi_corr_sc = wdi_sc.iloc[:,2:].corr()

mask = np.zeros(wdi_corr_sc.shape, dtype=bool)

mask[np.tril_indices(len(mask))] = True

sns.heatmap(wdi_corr_sc, annot = True, mask = mask);
sns.pairplot(wdi_sc.iloc[:,2:]);
y = wdi_sc['Gini']
ols = smf.ols('Gini ~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc',

                     data=wdi_sc)

olsf = ols.fit()

# Print out the statistics

olsf.summary()
sns.distplot(olsf.resid);
sns.regplot(np.exp(olsf.fittedvalues),np.exp(y));
sns.regplot(olsf.fittedvalues, olsf.resid, color="g", lowess = True);
fig, axs = plt.subplots(ncols=4, figsize=(30, 5))

sns.regplot(wdi_sc.iloc[:,4], olsf.resid, ax=axs[0], color="r", lowess = True);

sns.regplot(wdi_sc.iloc[:,5], olsf.resid, ax=axs[1], color="r", lowess = True);

sns.regplot(wdi_sc.iloc[:,6], olsf.resid, ax=axs[2], color="r", lowess = True);

sns.regplot(wdi_sc.iloc[:,7], olsf.resid, ax=axs[3], color="r", lowess = True);
import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ['Lagrange multiplier statistic', 'p-value',

        'f-value', 'f p-value']

test = sms.het_breuschpagan(olsf.resid, olsf.model.exog)

lzip(name, test)
rmse_ols = (np.sqrt(mean_squared_error(np.exp(y), np.exp(olsf.fittedvalues))))

performance =  pd.DataFrame([['Simple OLS', rmse_ols, test[1]]], columns=['model','rmse', 'het'])

performance
wdi_res = wdi_sc

wdi_res['Residuals'] = olsf.resid
box = sns.boxplot(x="CountryCode", y="Residuals", data=wdi_res);

box.set_xticklabels(box.get_xticklabels(), rotation=90);
conn = sqlite3.connect('../input/world-development-indicators/database.sqlite')
country_df = pd.read_sql_query("SELECT CountryCode,Region,IncomeGroup FROM Country", conn)
country_df.groupby('IncomeGroup').count()
wdi_res.shape
wdi_region = wdi_res.merge(country_df, left_on='CountryCode', right_on='CountryCode')
wdi_region.shape
wdi_region.head()
box = sns.boxplot(x="IncomeGroup", y="Residuals", data=wdi_region,

                 order=['High income: OECD', 'High income: nonOECD', 'Upper middle income', 'Lower middle income', 'Low income']);

box.set_xticklabels(box.get_xticklabels(), rotation=90);
wdi_region.groupby(['IncomeGroup']).count()
wdi_region['IncomeGroup'] = np.where(wdi_region['IncomeGroup']=='Low income', 

                                'Lower middle income',

                                wdi_region['IncomeGroup'])
box = sns.boxplot(x="IncomeGroup", y="Residuals", data=wdi_region,

                 order=['High income: OECD', 'High income: nonOECD', 'Upper middle income', 'Lower middle income']);

box.set_xticklabels(box.get_xticklabels(), rotation=90);
wdi_region.groupby(['IncomeGroup']).count()
wdi_region.head()
mvi = smf.mixedlm("Gini ~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc", data=wdi_region,

                 groups="IncomeGroup")

mvif = mvi.fit()

print(mvif.summary())
mvif.random_effects
sns.regplot(mvif.fittedvalues,mvif.resid, color="g", lowess = True);
name = ['Lagrange multiplier statistic', 'p-value',

        'f-value', 'f p-value']

test = sms.het_breuschpagan(mvif.resid, mvif.model.exog)

lzip(name, test)
rmse_mvi = np.sqrt(mean_squared_error(np.exp(y), np.exp(mvif.fittedvalues)))

performance.loc[1] =  ['Income Group, Varying Intercept', rmse_mvi,  test[1]]

performance
def plot_df_scatter_columns(df, y_column, grouping, rel_col):

    for z in df[rel_col]:    

        sns.lmplot(x = z, y = y_column, data = df, hue = grouping) 



rel_col = ['Year_sc', 'Inflat_sc', 'TaxRev_sc',

       'BusDay_sc', 'Unempl_sc']



plot_df_scatter_columns(wdi_region, 'Gini', "IncomeGroup", rel_col)
mvis = smf.mixedlm("Gini ~  Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc", data=wdi_region,

                 groups="IncomeGroup",

                  #re_formula="~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc"

                  re_formula="~ BusDay_sc"

                )

mvisf = mvis.fit()

print(mvisf.summary())

mvisf.random_effects
sns.regplot(mvisf.fittedvalues, mvisf.resid, color="g", lowess = True);
name = ['Lagrange multiplier statistic', 'p-value',

        'f-value', 'f p-value']

test = sms.het_breuschpagan(mvisf.resid, mvisf.model.exog)

lzip(name, test)
rmse_mvis = np.sqrt(mean_squared_error(np.exp(y), np.exp(mvisf.fittedvalues)))

performance.loc[2] =  ['Income Group, Varying Intercept and Slope', rmse_mvis, test[1]]

performance
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(wdi_region.iloc[:,4:8], wdi_region['Gini'])
Gini_pred = rf.predict(wdi_region.iloc[:,4:8])
rmse_rf = np.sqrt(mean_squared_error(np.exp(y), np.exp(Gini_pred)))
wdi_region.iloc[:,4:8].columns
rf.feature_importances_
rmse_rf
sns.regplot(Gini_pred,y-Gini_pred, color="g", lowess = True);
performance.loc[3] =  ['Rf', rmse_rf, np.nan]

performance
wdi_region.head()
olsd = smf.ols('Gini ~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc + C(IncomeGroup) ',

                     data=wdi_region)

olsdf = olsd.fit()

# Print out the statistics

olsdf.summary()
sns.regplot(olsdf.fittedvalues, olsdf.resid, color="g", lowess = True);
name = ['Lagrange multiplier statistic', 'p-value',

        'f-value', 'f p-value']

test = sms.het_breuschpagan(olsdf.resid, olsdf.model.exog)

lzip(name, test)
rmse_ols_dummy = (np.sqrt(mean_squared_error(np.exp(y), np.exp(olsdf.fittedvalues))))

performance.loc[4] =  ['Income Group Dummy', rmse_ols_dummy, test[1]]

performance