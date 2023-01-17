import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (7.0, 4.5)

plt.rcParams['font.size'] = 9

plt.rcParams['figure.titlesize'] = 10

plt.rcParams['axes.titlesize'] = 10

plt.rcParams['axes.labelsize'] = 9

plt.rcParams['legend.fontsize'] = 8

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['axes.formatter.useoffset'] = False

plt.rcParams['xtick.major.size'] = 0

plt.rcParams['ytick.major.size'] = 0



pd.set_option('display.float_format', lambda x: '%.3f' % x)



f = pd.read_csv('WDI_Data.csv', encoding='windows-1251')

f.rename(columns={'Country Name': 'CountryName', 'Country Code': 'CountryCode', 'Indicator Name': 'IndicatorName', 'Indicator Code': 'IndicatorCode'}, inplace=True)



country = ['Albania', 'Croatia', 'European Union', 'Macedonia, FYR', 'Montenegro', 'Serbia', 'Turkey']

df = f.loc[f.CountryName.isin(country)].copy()

df.loc[df.CountryName == 'Macedonia, FYR', 'CountryName'] = 'Macedonia'

df.loc[df.CountryName == 'European Union', 'CountryName'] = 'EUU'



colors = {'Albania': '#348ABD',

          'Croatia': '#9E95D6',

          'EUU': '#777777',

          'Macedonia': '#E24D37',

          'Montenegro': '#64C1A4',

          'Serbia': '#FBC15E',

          'Turkey': '#8EBA42'}



columns = ['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode']

df2 = pd.melt(df, id_vars=columns, var_name='Year', value_name='Value')

df2.dropna(inplace=True)



df2['Year'] = df2['Year'].astype(int)

df2 = df2.loc[df2.Year.isin(range(2000,2017))]



def plot_aid(indicators):

    col = ['CountryName', 'IndicatorName', 'IndicatorCode', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

    ind_stacked = df.loc[df.CountryName != 'EUU']

    ind_stacked = ind_stacked[col]

    

    ind_stacked = ind_stacked.fillna(0)

    label = []

    

    for key, color in colors.items():

        if key == 'EUU':

            continue

        label.append(key)

    label.sort()

        

    for indicator in indicators:

        ind_code = ind_stacked.loc[ind_stacked.IndicatorCode == indicator]

        

        index = np.arange(len(col[3:])) + 0.3

        bar_width = 0.8

        y_offset = np.array([0.0] * len(col[3:]))

        

        l = np.arange(len(ind_code)) #df length

        cl = np.arange(len(col[3:])) #column length

        years = []

        for j in cl:

            years.append(col[j+3])

            for i in l:

                c = colors[ind_code['CountryName'].iloc[i]]

                value = ind_code.iloc[i,j+3]/10000

                plt.bar(index[j], value, bar_width, bottom=y_offset[j], color=c)

                y_offset[j] += value

                plt.xticks(np.arange(len(col[3:])) + 0.6, years, rotation=90, horizontalalignment='left')

            title = ind_code['IndicatorName'].iloc[i]

        plt.title(title)

        plt.legend(label, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.grid(b=None, which='major', axis='x')

        plt.show()

        



def plot_ind(indicators):

    l = len(indicators)

    for indicator in indicators:

        fig, ax = plt.subplots()

        ind_code = df2.loc[df2.IndicatorCode == indicator]

        ind_name_country = ind_code.groupby(['IndicatorName', 'CountryName'])

        for name, values in ind_name_country:   

            ax.plot(values['Year'], values['Value'], label=name[1], color = colors[name[1]], linewidth = 2) 

            title = name[0]

            co2 = '(% of total fuel combustion)'

            if len(title) > 57 and co2 in title:

                ind = title.rfind(co2)

                title = title[0:ind]

            ax.set_title(title)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   

        plt.tick_params(axis='x', which='both', bottom='on', top='off')        

        plt.tick_params(axis='y', which='both', left='on', right='off')

        plt.tight_layout()

        plt.show()  