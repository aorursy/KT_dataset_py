

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv('../input/efw_cc.csv') 
data.info()
data.year.value_counts().sort_index().index  #it is count in data file.
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(),cbar=True, annot=True , linewidths=.5, fmt=".1f" , ax=ax)
plt.show()
#colleration Map , several value
#Year - ECONOMIC FREEDOM 
data.plot(kind="scatter",x="year", y="ECONOMIC FREEDOM" , alpha=0.5 , color="red", grid=True)
plt.xlabel("year")
plt.ylabel("Economic Freedom")
plt.title("Countries-Economic Freedom Scatter Plot")
plt.show()
#Rank - ECONOMIC FREEDOM
data.plot(kind="scatter",x="rank", y="ECONOMIC FREEDOM" , alpha=0.5 , color="red", grid=True)
plt.xlabel("year")
plt.ylabel("Economic Freedom")
plt.title("Rank-Economic Freedom Scatter Plot")
plt.show()
data_eff=data[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
sns.set(font_scale=1.3)
x,ax=plt.subplots(figsize=(12,12))
sns.heatmap(data_eff.corr(),cbar=True,annot=True,fmt='.1f')
plt.show()
turkey=data[data["countries"]=="Turkey"]
turkey.head(15)
tur_year=turkey.groupby('year')
def plot_stat(select,plot):
       tur_year[select].mean().plot(plot,figsize=[9,9],title=select)

data.plot(kind="hist",y="ECONOMIC FREEDOM",color="r")
data.plot(kind="hist",y="4_trade",color="g")
plt.show()
tur_year=turkey[turkey['year']>2009].groupby('year')
_ = tur_year.mean().plot(kind="barh",y=["1_size_government","1c_gov_enterprises",'1a_government_consumption','1d_top_marg_tax_rate'], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)
tur_year=turkey[turkey['year']>2009].groupby('year')
_ = tur_year.mean().plot(kind="barh",y=['2a_judicial_independence',"2b_impartial_courts","2c_protection_property_rights","2d_military_interference","2e_integrity_legal_system","2f_legal_enforcement_contracts","2g_restrictions_sale_real_property","2h_reliability_police","2i_business_costs_crime","2j_gender_adjustment","2_property_rights"], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)
tur_year=turkey[turkey['year']>2009].groupby('year')
_ = tur_year.mean().plot(kind="barh",y=["3b_std_inflation" ,"3c_inflation","3d_freedom_own_foreign_currency" ,"3_sound_money"], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)
tur_year=turkey[turkey['year']>2009].groupby('year')
_ = tur_year.mean().plot(kind="barh",y=["4a_tariffs","4b_regulatory_trade_barriers","4c_black_market","4d_control_movement_capital_ppl","4_trade"], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)
tur_year=turkey[turkey['year']>2009].groupby('year')
_ = tur_year.mean().plot(kind="barh",y=["5a_credit_market_reg","5b_labor_market_reg","5c_business_reg","5_regulation"], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)