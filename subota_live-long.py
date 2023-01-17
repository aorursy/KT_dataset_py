# Soft part:

#  imported libraries

#  helper data processing functions

#  helper visualization functions



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from math import copysign



pd.set_option('display.max_rows', 500)

pd.options.display.max_rows = 200

pd.options.display.max_colwidth = 200



ROWS, COLUMNS = 0, 1



def countries_coorelation(df, iX, iY):

    corrs = pd.DataFrame(

        data = {'Correlation':float('nan'), 'Count':0}

        , index=df.index.get_level_values('CountryCode'))

    for country in df.index.get_level_values('CountryCode'):

        country_data = df.loc[ country ]

        if country_data.shape[0] > 3 and np.var(country_data[iX])>0:

            correlation = country_data[iX].corr(country_data[iY])

            corrs.loc[country] = (correlation, len(country_data))

    return corrs

    

#def years_coorelation(df, iX, iY):

#    corrs = pd.DataFrame(columns=['Correlation', 'Year', 'Count'])

#    data = get_two_indicators(df, iX, iY)

#    i = 0

#    for year in set(data.Year):

#        year_data = data[ data.Year == Year ]

#        if len(year_data) > 2:

#            correlation = year_data.Value1.corr(year_data.Value2)

#            i += 1

#            corrs.loc[i] = (correlation, year, len(year_data))

#    return corrs



def scatter_color(df, iX, iY):

    d = pvt[[iX, iY]].dropna(ROWS)

    X = d[iX]

    Y = d[iY]

    Year = d.reset_index('Year')['Year']



    plt.figure(1).set_size_inches(14,5)

    plot = plt.scatter(X, Y, c=Year, marker=',', s=6, alpha=0.25)

    plt.xlabel( ser.loc[iX].IndicatorName )

    plt.ylabel( ser.loc[iY].IndicatorName )

    plt.colorbar(plot)

    

    m, b = np.polyfit(X, Y, 1)

    fitX = np.array([min(X), max(X)])

    plt.plot(fitX, fitX*m + b, 'r-', linewidth=4)

    

    r_sq = (X*m + b).var() / Y.var()

    plt.title( 'Determination {:.0f}% , corr={:.2f}'.format( r_sq*100, X.corr(Y) ), loc='right' )



    plt.grid()

    

    return plot



def demo_indicators(iX, iY='SP.DYN.LE00.IN'):

    def signed_sqare(x):

        # converts correlation into R-sqared, but of the same sign

        return copysign( x**2, x)

    scatter_color(ind, iX, iY)

    plt.show()

    by_country = countries_coorelation(pvt, iX, iY)

    plt.hist(by_country.Correlation.dropna().apply(signed_sqare) * 100,

            bins = 20)

    plt.xlim(-100, 100)

    plt.xlabel('determination, %')

    plt.ylabel('# conutries')

    plt.grid()

    plt.show()
# load data

ind = pd.read_csv('../input/Indicators.csv', usecols=['CountryCode','IndicatorCode','Year','Value'])

ser = pd.read_csv('../input/Series.csv', index_col='SeriesCode')

countries = pd.read_csv('../input/Country.csv', index_col='CountryCode')



pvt = ind.pivot_table(index=['CountryCode','Year'], columns='IndicatorCode', values='Value')

#pvt.loc[('USA',1995):('USA',1998)].iloc[:,4:8]
correlations = pd.DataFrame(

    np.full_like(pvt.columns, np.nan, dtype=float)

    , columns = ['correlation']

    , index = pvt.columns)

for column in pvt.columns:

    tmpDat = pvt[[column,'SP.DYN.LE00.IN']].dropna()

    if tmpDat.size > 0 and np.var(tmpDat.iloc[:,0]) != 0:

        correlations.loc[column] = np.corrcoef(

            x = tmpDat

            , rowvar = False

        )[0,1]

correlations = correlations[['correlation']].join(

    ser[['Topic','IndicatorName','ShortDefinition','LongDefinition']]

)

correlations = correlations[correlations.Topic != 'Health: Mortality']

correlations['abscorr'] = np.abs(correlations.correlation)

#correlations.nlargest(20, 'abscorr').drop('abscorr',COLUMNS)
demo_indicators('SP.DYN.CBRT.IN')
print( "Countries involved: {}".format(len(set(ind[ind.IndicatorCode == 'SE.SEC.NENR'].CountryCode))) )
demo_indicators('SE.SEC.NENR')
demo_indicators('NV.AGR.TOTL.ZS')
demo_indicators('EG.USE.CRNW.ZS')
demo_indicators('IT.MLT.MAIN.P2')
demo_indicators('FP.WPI.TOTL')
demo_indicators('NE.CON.PRVT.PC.KD')
# row and column sharing

f, sub_plots = plt.subplots(4, 4, sharex='all', sharey='all')

f.set_size_inches(14,14)

sub_plots = [ x for a in sub_plots for x in a ]



for cList in [

    ['WLD'],['ARE'],['TUR'],['SYR'],

    ['BGR'],['BRB'],['BGD'],['BHR'],

    ['AUS'],['USA'],['UKR'],['CHE'],

    ['GBR'],['MKD'],['RUS'],['JPN']]:

    d = pvt.loc[cList]

    p = sub_plots.pop(0)

    p.scatter(d['NE.CON.PRVT.PC.KD'], d['SP.DYN.LE00.IN'], c=d.reset_index('Year').Year, marker=',', s=4)

    p.set_title("+".join(countries.loc[cList].ShortName), loc='left')
correlations[correlations.abscorr>0.5].sort_values('abscorr', ascending=False)[

    ['correlation', 'IndicatorName']

]