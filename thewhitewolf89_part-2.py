# Importing libraries

import pandas as pd

import numpy as np 

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import plotly.express as px

from scipy.stats import pearsonr

import warnings

warnings.filterwarnings('ignore')
merged = pd.read_csv('../input/quantium-data-analytics-part-2/QVI_data.csv')
merged.info()
merged['STORE_NBR'] = merged['STORE_NBR'].astype('object')

merged['LYLTY_CARD_NBR'] = merged['LYLTY_CARD_NBR'].astype('object')

merged['PROD_NBR'] = merged['PROD_NBR'].astype('object')

merged['TXN_ID'] = merged['TXN_ID'].astype('object')

merged['DATE']= pd.to_datetime(merged['DATE'])

merged.info()
merged.head()
merged['year_month']=merged['DATE'].dt.strftime('%Y-%m')
merged
measureovertime=merged.groupby(['STORE_NBR','year_month']).agg({"TOT_SALES": np.sum,"PROD_QTY": np.sum,"TXN_ID": lambda x: x.nunique(),"LYLTY_CARD_NBR": lambda x: x.nunique()}).reset_index()
measureovertime.rename(columns = {'TOT_SALES':'OVERALL_TOT_SALES_MONTH', 'PROD_QTY':'OVERALL_PROD_QTY_MONTH', 

                              'TXN_ID':'UNIQUE_TXN_ID','LYLTY_CARD_NBR':'UNIQUE_Monthly_Customers'}, inplace = True) 
measureovertime
measureovertime['AVG_MONTHLY_PRICE_PER_QTY'] = measureovertime['OVERALL_TOT_SALES_MONTH']/measureovertime['OVERALL_PROD_QTY_MONTH']

measureovertime['Unique_Transactions_Per_Customer'] = measureovertime['UNIQUE_TXN_ID']/measureovertime['UNIQUE_Monthly_Customers']

measureovertime['ChipsperTransaction'] = measureovertime['OVERALL_PROD_QTY_MONTH']/measureovertime['UNIQUE_TXN_ID']
measureovertime.STORE_NBR = measureovertime.STORE_NBR.astype('object')
sns.heatmap(measureovertime.corr(),annot=True,cmap="YlGnBu")

plt.show()
measureovertime.STORE_NBR.value_counts()
fullobsarray = measureovertime[measureovertime['STORE_NBR'].isin(measureovertime['STORE_NBR'].value_counts()[measureovertime['STORE_NBR'].value_counts()==12].index)].STORE_NBR.unique()

fullobsarray
pretrial = measureovertime[(measureovertime['STORE_NBR'].isin(fullobsarray))]
pretrial.STORE_NBR.value_counts()
pretrial['yearmonthint']=pretrial['year_month'].str.replace('-','')
pretrial['yearmonthint']=pretrial['yearmonthint'].astype('int')
pretrial_obs = pretrial[pretrial['yearmonthint'] < 201902]
pretrial_obs
storenumbers = pretrial_obs.STORE_NBR.unique()

storenumbers = storenumbers.astype('int')

storenumbers
pretrial_obs.STORE_NBR = pretrial_obs.STORE_NBR.astype("int")
sample77 = pretrial_obs.loc[pretrial_obs['STORE_NBR'] == 77,['OVERALL_TOT_SALES_MONTH']]

sample77
calcCorrTable = pd.DataFrame(columns = ['Store1', 'Store2' , 'corr_measure'],dtype=np.int64)

stores=[]

corr_m=[]

for i in storenumbers:

    sampleothers = pretrial_obs.loc[pretrial_obs['STORE_NBR'] == i,['OVERALL_TOT_SALES_MONTH']]

    stores.append(i)

    corr_m.append(pearsonr(sample77['OVERALL_TOT_SALES_MONTH'],sampleothers['OVERALL_TOT_SALES_MONTH'])[0])

calcCorrTable['Store2'] = stores

calcCorrTable['corr_measure'] = corr_m
calcCorrTable['Store1'] = 77
calcCorrTable.sort_values(by=['corr_measure'], inplace=True,ascending=False)
calcCorrTable
sample77_cust = pretrial_obs.loc[pretrial_obs['STORE_NBR'] == 77,['UNIQUE_Monthly_Customers']]



calcCorrTable2 = pd.DataFrame(columns = ['Store1', 'Store2' , 'corr_measure_cust'],dtype=np.int64)

stores=[]

corr_m=[]

for i in storenumbers:

    sampleothers = pretrial_obs.loc[pretrial_obs['STORE_NBR'] == i,['UNIQUE_Monthly_Customers']]

    stores.append(i)

    corr_m.append(pearsonr(sample77_cust['UNIQUE_Monthly_Customers'],sampleothers['UNIQUE_Monthly_Customers'])[0])

calcCorrTable2['Store2'] = stores

calcCorrTable2['corr_measure_cust'] = corr_m
calcCorrTable2['Store1'] = 77

calcCorrTable2
calcCorrTable2.sort_values(by=['corr_measure_cust'], inplace=True,ascending=False)
calcCorrTable2
merged_pretrail_obs = calcCorrTable.merge(calcCorrTable2, on='Store2', how='inner')
del merged_pretrail_obs['Store1_y']

merged_pretrail_obs = merged_pretrail_obs.rename(columns={"Store1_x": "Store1"})
merged_pretrail_obs.sort_values(by=['corr_measure','corr_measure_cust'], inplace=True,ascending=False)
merged_pretrail_obs
hue = []

for i in pretrial_obs['STORE_NBR']:

    if i == 233:

        hue.append("control store")

    elif i == 77:

        hue.append("trial store")

    else:

        hue.append("others")

pretrial_obs['categorystores'] = hue
pretrial_obs.categorystores.value_counts()
plt.figure(figsize=(20,10))

sns.lineplot(x='year_month',y='OVERALL_TOT_SALES_MONTH',hue = 'categorystores',data = pretrial_obs)

plt.title("TRIAL AND CONTROL STORES Sales Value Comparison for store 233 over time")

plt.show()
plt.figure(figsize=(20,10))

sns.lineplot(x='year_month',y='UNIQUE_Monthly_Customers',hue = 'categorystores',data = pretrial_obs)

plt.title("TRIAL AND CONTROL STORES Unique Customers Comparison for store 233 over time")

plt.show()
scalingfactor_sales = pretrial_obs[(pretrial_obs['yearmonthint'] < 201902) & (pretrial_obs['STORE_NBR']==77)][['OVERALL_TOT_SALES_MONTH']].sum()/pretrial_obs[(pretrial_obs['yearmonthint'] < 201902) & (pretrial_obs['categorystores']=='control store')][['OVERALL_TOT_SALES_MONTH']].sum()

scalingfactor_sales
measureovertimesales=measureovertime.copy()
scaledcontrolsales = measureovertimesales[measureovertimesales['STORE_NBR']== 233]
scaledcontrolsales['OVERALL_TOT_SALES_MONTH']=scaledcontrolsales['OVERALL_TOT_SALES_MONTH'].astype('float')

scaledcontrolsales['controlsales'] =scaledcontrolsales['OVERALL_TOT_SALES_MONTH']*scalingfactor_sales.values

scaledcontrolsales
scaledtrialsales = measureovertimesales[measureovertimesales['STORE_NBR']==77][['OVERALL_TOT_SALES_MONTH','year_month']]
scaledmerged = pd.merge(scaledcontrolsales[['year_month','controlsales']],scaledtrialsales, on = 'year_month',how = 'inner')
scaledmerged['percentdiff'] = abs(scaledmerged['controlsales']-scaledmerged['OVERALL_TOT_SALES_MONTH'])/scaledmerged['controlsales']

scaledmerged
scaledmerged['year_month']=scaledmerged['year_month'].str.replace('-','')

scaledmerged['year_month'] = scaledmerged['year_month'].astype('int')

scaledmerged = scaledmerged.rename(columns={"OVERALL_TOT_SALES_MONTH": "trialstoresales"})
scaledmerged
sample1 = scaledmerged[(scaledmerged['year_month']>201901) & (scaledmerged['year_month']<201905)][['controlsales']].mean()

sample2 = scaledmerged[(scaledmerged['year_month']>201901) & (scaledmerged['year_month']<201905)][['trialstoresales']].mean()
meandiff= sample1.values-sample2.values

meandiff
#calculating the std deviation of population before the trial

stddev_pop = scaledmerged.loc[scaledmerged['year_month'] < 201902,['percentdiff']].std()
scaledmerged['t_value'] = (scaledmerged['percentdiff']-0)/stddev_pop.values
scaledmerged[(scaledmerged['year_month']>201901) & (scaledmerged['year_month']<201905)]
scaledmerged.loc[scaledmerged['year_month'] < 201902].count()

#there are 7 values totally pre trial

deg_freedom = 7
from scipy.stats import norm

t_critical = norm.ppf(0.95)

t_critical
hue = []

for i in measureovertimesales['STORE_NBR']:

    if i == 233:

        hue.append("control store")

    elif i == 77:

        hue.append("trial store")

    else:

        hue.append("others")

measureovertimesales['categorystores'] = hue
measureovertimesales['year_month']=measureovertimesales['year_month'].str.replace('-','')

measureovertimesales['year_month'] = measureovertimesales['year_month'].astype('int')
pastsales= measureovertimesales.groupby(['year_month','categorystores'])[['OVERALL_TOT_SALES_MONTH']].mean().reset_index()

pastsales_new = pastsales[pastsales['year_month']<201903]

pastsales_new = pastsales_new.rename(columns={"OVERALL_TOT_SALES_MONTH": "AVG_sales"})

pastsales_new
pastsales_new['Month']=pastsales_new['year_month'].astype('str')

pastsales_new['Month'] = pastsales_new['Month'].str[-2:]

pastsales_new['Month']= pastsales_new['Month'].astype('int')

import calendar

pastsales_new['Month']=pastsales_new['Month'].apply(lambda x: calendar.month_abbr[x])
plt.figure(figsize=(20,10))

sns.lineplot(x='Month',y='AVG_sales',hue = 'categorystores',data = pastsales_new)

plt.title("TRIAL AND CONTROL STORES vs others on Sales over time")

plt.show()