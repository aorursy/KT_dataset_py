#Import all required libraries for reading data, analysing and visualizing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

efw = pd.read_csv('../input/efw_cc.csv')
efw.shape
efw.head()
efw.year.value_counts().sort_index().index
efw.info()
efw_eff=efw[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
sns.set(font_scale=1.3)
x,ax=plt.subplots(figsize=(12,12))
sns.heatmap(efw_eff.corr(),cbar=True,annot=True,fmt='.2f',square=True)
efw_=efw[['ECONOMIC FREEDOM','year']]
#plt.plot(efw_['year'],efw_['ECONOMIC FREEDOM'])
efw_.groupby('year').mean().plot(figsize=[9,9],)
egypt=efw[ efw['countries']=='Egypt']
egypt.head(5)
egy_year=egypt.groupby('year')
def plot_stat(select,plot):
       egy_year[select].mean().plot(plot,figsize=[9,9],title=select)

plot_stat('ECONOMIC FREEDOM','line')
plot_stat('3b_std_inflation','line')
egy_year=egypt[egypt['year']>2010].groupby('year')
_ = egy_year.mean().plot(y=["1_size_government","1c_gov_enterprises",'1a_government_consumption','1d_top_marg_tax_rate','4a_tariffs',"3_sound_money"], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)
_x = egy_year.mean().plot(y=["1b_transfers","3a_money_growth",'4c_black_market','ECONOMIC FREEDOM'], figsize = (15,20), subplots=True)
_x = plt.xticks(rotation=360)
egy_year=egypt[egypt['year']>2008].groupby('year')
_x = egy_year.mean().plot(kind='barh',y=["5a_credit_market_reg","5b_labor_market_reg",'5c_business_reg','5_regulation','2_property_rights','2c_protection_property_rights'], figsize = (15,20), subplots=True)
egy_year=egypt[egypt['year']>1990].groupby('year')
_x = egy_year.mean().plot(kind='area',y=["2h_reliability_police",'2e_integrity_legal_system','2b_impartial_courts','2a_judicial_independence','2d_military_interference'], figsize = (15,20), subplots=True)
_x = plt.xticks(rotation=360)
