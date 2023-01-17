import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization

import matplotlib.pyplot as plt #visualization

from sklearn import linear_model # our linear regression code



beta, mse, r_squared, name = [], [], [], []



indicators = pd.read_csv('../input/Indicators.csv')



regions = ['ARB','CSS','EAS','EAP','ECS','ECA','EUU','FCS','HPC','HIC','NOC','OEC','LCN',

           'LAC','LDC','LMY','LIC','LMC','MEA','MNA','MIC','NAC','MNP','OED','OSS','PSS',

           'SST','SAS','ZAF','SSF','SSA','UMC','WLD']
#percent

irrigated_agri_lnd = indicators.loc[indicators.IndicatorCode == 'AG.LND.IRIG.AG.ZS'][['CountryCode','Year','Value']]

total_agri_lnd = indicators.loc[indicators.IndicatorCode == 'AG.LND.AGRI.K2'][['CountryCode','Year','Value']]

value_added_agri  = indicators.loc[indicators.IndicatorCode == 'NV.AGR.TOTL.CD'][['CountryCode','Year','Value']]

#billion cubic meters

annual_freshwater_withdrawl = indicators.loc[indicators.IndicatorCode == 'ER.H2O.INTR.K3'][['CountryCode','Year','Value']]

#percent of total freshwater withdrawal

annual_agri_freshwater = indicators.loc[indicators.IndicatorCode == 'ER.H2O.FWAG.ZS'][['CountryCode','Year','Value']]

total_arable_lnd = indicators.loc[indicators.IndicatorCode == 'AG.LND.ARBL.HA'][['CountryCode','Year','Value']]

#arable land as percentage of total land

percent_arable_land = indicators.loc[indicators.IndicatorCode == 'AG.LND.ARBL.ZS'][['CountryCode','Year','Value']]

#average precipitation in depth (mm per year)

average_annual_precipitation = indicators.loc[indicators.IndicatorCode == 'AG.LND.PRCP.MM'][['CountryCode','Year','Value']]

total_cereal_production = indicators.loc[indicators.IndicatorCode == 'AG.PRD.CREL.MT'][['CountryCode','Year','Value']]

#yield per hectare

cereal_yield_hectare = indicators.loc[indicators.IndicatorCode == 'AG.YLD.CREL.KD'][['CountryCode','Year','Value']]

#2004-2006 = 100

crop_production_index = indicators.loc[indicators.IndicatorCode == 'AG.PRD.CROP.XD'][['CountryCode','Year','Value']]

#kg per hectare of arable land

fertilizer_per_hectare_arable = indicators.loc[indicators.IndicatorCode == 'AG.CON.FERT.ZS'][['CountryCode','Year','Value']]

total_land_area = indicators.loc[indicators.IndicatorCode == 'AG.LND.TOTL.K2'][['CountryCode','Year','Value']]

land_area_b_5_perc = indicators.loc[indicators.IndicatorCode == 'AG.LND.EL5M.ZS'][['CountryCode','Year','Value']]

total_population = indicators.loc[indicators.IndicatorCode == 'SP.POP.TOTL'][['CountryCode','Year','Value']]

total_tractors = indicators.loc[indicators.IndicatorCode == 'AG.AGR.TRAC.NO'][['CountryCode','Year','Value']]
def initial_merge(df1, df2, columns, log):  

    df = pd.merge(df1, df2, how='left', on=['CountryCode','Year'])

    df = df.dropna(axis=0, how='any')

    if columns:

        return df

    else:

        for name in regions:

            df.drop(df.loc[df.CountryCode == name].index, inplace=True)

        if 'Value_x' in df.columns.values:

            df.drop(df.loc[df.Value_x==0].index, inplace=True)

            df.drop(df.loc[df.Value_y==0].index, inplace=True)

        elif 'Value' in df.columns.values:

            df.drop(df.loc[df.Value==0].index, inplace=True)

        df.drop(['CountryCode','Year'], axis=1, inplace=True)

        if log:

            df = np.log(df)

        else:

            df.Value_y = np.log(df.Value_y)

        x = np.array(df.Value_x).reshape(-1, 1)

        y = np.array(df.Value_y).reshape(-1, 1)

        return x, y



def cleaner(df):

    for name in regions:

        df.drop(df.loc[df.CountryCode == name].index, inplace=True)

    df.drop(df.loc[df.Value == 0].index, inplace=True)

    return df



def calc_merge(df1, df2, calc, percent):

    df = initial_merge(df1, df2, True, None)

    if percent is True:

        if calc == '*':

            df['Value'] = (df.Value_x / 100) * df.Value_y

    else:

        if calc == '*':

            df['Value'] = df.Value_x * df.Value_y

        elif calc == '/':

            df['Value'] = df.Value_x / df.Value_y

    df.drop(['Value_x','Value_y'], axis=1, inplace=True)

    return df



def model(x, y, z):

    reg = linear_model.LinearRegression()

    reg.fit(x, y)

    beta.append(reg.coef_[0][0])

    mse.append(np.mean((reg.predict(x) - y) ** 2))

    r_squared.append(reg.score(x, y))

    name.append(z)

    return reg
x1, y1 = initial_merge(total_cereal_production, value_added_agri, False, True)

x2, y2 = initial_merge(total_agri_lnd, total_cereal_production, False, True)

x3, y3 = initial_merge(annual_freshwater_withdrawl, total_cereal_production, False, True)

x4, y4 = initial_merge(total_arable_lnd, total_cereal_production, False, True)



reg1 = model(x1, y1, 'Total Cereal Production')

reg2 = model(x2, y2, 'Total Agri Land')

reg3 = model(x3, y3, 'Annual Fresh W/d')

reg4 = model(x4, y4, 'Total Arable Land')



f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, dpi=100)

plt.tight_layout(h_pad=1.5)

ax1.scatter(x1, y1)

ax1.plot(x1, reg1.predict(x1), color='orange')

ax1.set_xlabel('Total Cereal Production')

ax1.set_ylabel('Agriculture, Value Added')

ax2.scatter(x2, y2)

ax2.plot(x2, reg2.predict(x2), color='orange')

ax2.set_xlabel('Agricultural Land')

ax2.set_ylabel('Cereal Production')

ax3.scatter(x3, y3)

ax3.plot(x3, reg3.predict(x3), color='orange')

ax3.set_xlabel('Annual Freshwater W/d')

ax3.set_ylabel('Cereal Production')

ax4.scatter(x4, y4)

ax4.plot(x4, reg4.predict(x4), color='orange')

ax4.set_xlabel('Total Arable Land')

ax4.set_ylabel('Cereal Produciton')
x5, y5 = initial_merge(irrigated_agri_lnd, total_cereal_production, False, False)

x6, y6 = initial_merge(annual_agri_freshwater, total_cereal_production, False, False)

x7, y7 = initial_merge(percent_arable_land, total_cereal_production, False, False)

x8, y8 = initial_merge(fertilizer_per_hectare_arable, total_cereal_production, False, True)



reg5 = model(x5, y5, 'Percentage Irrigated Land')

reg6 = model(x6, y6, 'Annual Agri Freshwater')

reg7 = model(x7, y7, 'Percentage Arable Land')

reg8 = model(x8, y8, 'Fertilizer per Hectare')



f, ((ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=2, ncols=2, dpi=100)

plt.tight_layout(h_pad=1.5)

ax5.scatter(x5, y5)

ax5.plot(x5, reg5.predict(x5), color='orange')

ax5.set_xlabel('Irrigated Land, % of Total')

ax5.set_ylabel('Cereal Produciton')

ax6.scatter(x6, y6)

ax6.plot(x6, reg6.predict(x6), color='orange')

ax6.set_xlabel('Annual Agriculture Freshwater W/d, % of Total')

ax6.set_ylabel('Cereal Produciton')

ax7.scatter(x7, y7)

ax7.plot(x7, reg7.predict(x7), color='orange')

ax7.set_xlabel('Percentage Arable Land, % of Total')

ax7.set_ylabel('Cereal Produciton')

ax8.scatter(x8, y8)

ax8.plot(x8, reg8.predict(x8), color='orange')

ax8.set_xlabel('Fertilizer per Hectare')

ax8.set_ylabel('Cereal Produciton')
# Fixing certain data frames to change from percentages amounts or different units

total_irrigated = calc_merge(irrigated_agri_lnd, total_agri_lnd, '*', True)

agri_freshwater = calc_merge(annual_freshwater_withdrawl, annual_agri_freshwater, '*', True)

fertilizer_per_arable = fertilizer_per_hectare_arable.copy()

fertilizer_per_arable.Value = fertilizer_per_arable.Value * 100



x9, y9 = initial_merge(total_irrigated, total_cereal_production, False, True)

x10, y10 = initial_merge(agri_freshwater, total_cereal_production, False, True)

x11, y11 = initial_merge(fertilizer_per_arable, total_cereal_production, False, True)

x12, y12 = initial_merge(total_tractors, total_cereal_production, False, True)



reg9 = model(x9, y9, 'Irrigated Agricultural Land')

reg10 = model(x10, y10, 'Agriculture Freshwater Withdrawal')

reg11 = model(x11, y11, 'Fertilizer per Arable Land')

reg12 = model(x12, y12, 'Total Tractors')



f, ((ax9, ax10), (ax11, ax12)) = plt.subplots(nrows=2, ncols=2, dpi=100)

plt.tight_layout(h_pad=1.5)

ax9.scatter(x9, y9)

ax9.plot(x9, reg9.predict(x9), color='orange')

ax9.set_xlabel('Irrigated Agricultural Land')

ax9.set_ylabel('Cereal Produciton')

ax10.scatter(x10, y10)

ax10.plot(x10, reg10.predict(x10), color='orange')

ax10.set_xlabel('Agriculture Freshwater Withdrawal')

ax10.set_ylabel('Cereal Produciton')

ax11.scatter(x11, y11)

ax11.plot(x11, reg11.predict(x11), color='orange')

ax11.set_xlabel('Fertilizer per Arable Land, Hectares')

ax11.set_ylabel('Cereal Produciton')

ax12.scatter(x12, y12)

ax12.plot(x12, reg12.predict(x12), color='orange')

ax12.set_xlabel('Total Tractors')

ax12.set_ylabel('Cereal Produciton')
tractors_per_arable = calc_merge(total_tractors, total_arable_lnd, '/', False)

precipitation_per_arable = calc_merge(average_annual_precipitation, total_arable_lnd, '/', False)

fertilizer_total = calc_merge(fertilizer_per_arable, total_arable_lnd, '*', False)



x13, y13 = initial_merge(tractors_per_arable, total_cereal_production, False, True)

x14, y14 = initial_merge(fertilizer_total, total_cereal_production, False, True)

x15, y15 = initial_merge(average_annual_precipitation, total_cereal_production, False, True)

x16, y16 = initial_merge(precipitation_per_arable, total_cereal_production, False, True)



reg13 = model(x13, y13, 'Tractors per Arable Land')

reg14 = model(x14, y14, 'Total Fertilizer')

reg15 = model(x15, y15, 'Precipitation')

reg16 = model(x16, y16, 'Precip per Arable')



f, ((ax13,ax14),(ax15,ax16)) = plt.subplots(nrows=2, ncols=2, dpi=100)

plt.tight_layout(h_pad=1.5)

ax13.scatter(x13, y13)

ax13.plot(x13, reg13.predict(x13), color='orange')

ax13.set_xlabel('Tractors per Arable Land')

ax13.set_ylabel('Cereal Produciton')

ax14.scatter(x14, y14)

ax14.plot(x14, reg14.predict(x14), color='orange')

ax14.set_xlabel('Total Fertilizer')

ax14.set_ylabel('Cereal Produciton')

ax15.scatter(x15, y15)

ax15.plot(x15, reg15.predict(x15), color='orange')

ax15.set_xlabel('Annual Average Precipitation')

ax15.set_ylabel('Cereal Produciton')

ax16.scatter(x16, y16)

ax16.plot(x16, reg16.predict(x16), color='orange')

ax16.set_xlabel('Precipitation per Arable Land')

ax16.set_ylabel('Cereal Produciton')
coeff_df = pd.DataFrame({

    'Beta':beta,

    'MSE':mse,

    'R**2':r_squared

}, index=name)

coeff_df
df = initial_merge(total_arable_lnd, total_cereal_production, True, True)

df = df.rename(columns={'Value_x':'arable','Value_y':'cereal'})

df = initial_merge(df, fertilizer_total, True, True)

df = df.rename(columns={'Value':'fertilizer'})

df = initial_merge(df, precipitation_per_arable, True, True)

df = df.rename(columns={'Value':'precipitation per arable'})

df.drop(['CountryCode','Year'], axis=1, inplace=True)

df.head()
coeff = [[],[],[],[]]



for x in range(1000):

    train = df.sample(frac=0.5)

    x_train = train.drop('cereal', axis=1)

    y_train = pd.DataFrame(train['cereal'])

    x_test = df.sample(frac=0.5).drop('cereal', axis=1)

    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    coeff[0].append(regr.coef_[0][0])

    coeff[1].append(regr.coef_[0][1])

    coeff[2].append(regr.coef_[0][2])

    coeff[3].append(regr.score(x_train, y_train))

    

coeff_df2 = pd.DataFrame(train.columns.delete(1))

coeff_df2.columns = ['Features']

coeff_df2['Coefficient Estimate'] = pd.Series([np.mean(coeff[0]),np.mean(coeff[1]),np.mean(coeff[2])])

coeff_df2['R-squared'] = np.mean(coeff[3])

coeff_df2