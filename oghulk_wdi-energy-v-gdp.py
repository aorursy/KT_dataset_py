import numpy as np

import pandas as pd

import sqlite3 as sql

import scipy.stats as sp

import matplotlib.pyplot as plt



plt.style.use('fivethirtyeight')



conn = sql.connect('../input/database.sqlite')
Series = pd.read_sql('''

                     SELECT IndicatorName

                            ,SeriesCode

                            ,UnitOfMeasure

                            ,DevelopmentRelevance

                     FROM   Series

                     ''', con = conn)



names, codes = [], []

for v in Series['IndicatorName']:

    if v not in names:

        names.append(v)

for v in Series['SeriesCode']:

    if v not in codes:

        codes.append(v)

index = list(zip(names, codes))
nrg = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE  IndicatorCode IS 'EG.USE.PCAP.KG.OE' ''', con = conn)

pop = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE IndicatorCode IS 'SP.POP.TOTL' ''', con = conn)

gdp = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE IndicatorCode IS 'NY.GDP.MKTP.KD' ''', con = conn)

#Final consumption expenditure

fce = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE  IndicatorCode IS 'NE.CON.TOTL.KD' ''', con = conn)

#Manufacturing value added

mva = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE  IndicatorCode IS 'NV.IND.MANF.KD' ''', con = conn)

#Gross fixed capital formation

gfc = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE IndicatorCode IS 'NE.GDI.FTOT.KD' ''', con = conn)

co2 = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE IndicatorCode IS 'EN.ATM.CO2E.KT' ''', con = conn)

hit = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                  WHERE IndicatorCode IS 'TX.VAL.TECH.CD' ''', con = conn)



#List of country codes that basically cause a double count -- they refer to regions of the world

names = ['EAS','EAP','EMU','ECS','EUU','HPC','HIC','OEC','LCN','LAC','LMY','LMC','MIC','NAX','OED',

        'SAS','SSF','SSA','UMC','WLD','DZA']
energy = pd.merge(nrg, pop, how='left', on=['CountryCode','Year'])

energy['energy'] = energy.Value_x * energy.Value_y

energy = energy.drop(['Value_x','Value_y'], axis=1)

for n in names:

    drop   = energy.loc[energy.CountryCode == n].index

    energy = energy.drop(drop, axis=0)



def model(df1, df2):

    df = pd.merge(df1,  df2, how='left', on=['CountryCode','Year'])

    df = df.drop(['CountryCode','Year'], axis=1)

    df = df.dropna(axis=0, how='any')

    df = np.log(df)



    results = sp.linregress(df)

    plt.scatter(df.energy, df.Value)

    plt.plot(   df.energy, (df.energy*results[0])+results[1], color='orange')

    print('Beta:      {0} \n'

          'Alpha:     {1} \n'

          'R-Coef:    {2} \n'

          'R-Squared: {3}'.format(results[0],results[1],results[2],results[2]**2))



model(energy, gdp)
model(energy, fce)
model(energy, mva)
model(energy, gfc)
model(energy, co2)
pat = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                     WHERE IndicatorCode IS 'IP.PAT.RESD' ''', con = conn)

pat2 = pd.read_sql('''SELECT CountryCode, Year, Value FROM Indicators

                      WHERE IndicatorCode IS 'IP.PAT.NRES' ''', con = conn)

pat3 = pd.merge(pat, pat2, how='left', on=['CountryCode','Year'])

patents = pd.DataFrame({'CountryCode':pat3.CountryCode,

                        'Year':pat3.Year,

                        'Value':pat3.Value_x + pat3.Value_y})

model(energy, patents)