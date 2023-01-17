import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





%matplotlib inline

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10.0, 6.5)

plt.rcParams['font.size'] = 12

plt.rcParams['figure.titlesize'] = 16

plt.rcParams['figure.titleweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 14

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams["axes.labelsize"] = 13

plt.rcParams["axes.labelweight"] = 'bold'

plt.rcParams["xtick.labelsize"] = 11

plt.rcParams["ytick.labelsize"] = 11

plt.rcParams['legend.fontsize'] = 11
database = '../input/database.sqlite'



with sqlite3.connect(database) as con:

    df = pd.read_sql_query(

        """

        SELECT *

        FROM Indicators

        WHERE CountryCode in ('BIH', 'HRV', 'MKD', 'MNE', 'SRB', 'SVN', 'EUU')

        """,

        con

    )



df.loc[df.CountryCode == 'EUU', 'CountryName'] = 'avg. EU'

df.loc[df.CountryCode == 'MKD', 'CountryName'] = 'Macedonia'

    

df.info()
colors = {

    'Bosnia and Herzegovina': "#66C2A5",

    'Croatia': "#FA8D62",

    'Macedonia': "#FED82F",

    'Montenegro': "#E68AC3",

    'Serbia': "#8D9FCA",

    'Slovenia': "#A6D853",

    'avg. EU': "#CCCCCC"

}



def plot_indicator(indicators, title=None, 

                   xlim=None, ylim=None, xlabel='Year',

                   loc=0, loc2=0,

                   drop_eu=False):

    

    lines = ['-', '--']

    line_styles = []

    fig, ax = plt.subplots()

    

    indicators = indicators if isinstance(indicators, list) else [indicators]

    

    for line, (name, indicator) in zip(lines, indicators):

        ls, = plt.plot(np.nan, linestyle=line, color='#999999')

        line_styles.append([ls, name])



        df_ind = df[(df.IndicatorCode == indicator) & (df.Year >= 1990)]

        group = df_ind.groupby(['CountryName'])

        

        for country, values in group:

            country_values = values.groupby('Year').mean()

            

            if country == 'avg. EU':

                if drop_eu:

                    continue

                ax.plot(country_values, label=country, 

                        linestyle='--', color='#666666', linewidth=1.5, zorder=0)

            elif country_values.shape[0] > 1:

                ax.plot(country_values, label=country, linestyle=line,

                        color=colors[country], linewidth=2.5)

        

        if line == lines[0]:

            legend = plt.legend(loc=loc)



    ax.set_xlabel(xlabel)

    ax.set_xlim(xlim)

    ax.set_ylim(ylim)

    

    plt.tight_layout()

    fig.subplots_adjust(top=0.94)

    

    

    if title:

        ax.set_title(title)

    else:

        ax.set_title(df_ind.IndicatorName.values[0])

    

    if len(indicators) > 1:

        plt.legend(*zip(*line_styles), loc=loc2)

        ax.add_artist(legend)
population = [

    ('pop_dens', 'EN.POP.DNST'),     # Population density 

    ('rural', 'SP.RUR.TOTL.ZS'),     # Rural population 

    ('under14', 'SP.POP.0014.TO.ZS'),# Population, ages 0-14 

    ('above65', 'SP.POP.65UP.TO.ZS'),# Population ages 65 and above 

]



for indicator in population:

    plot_indicator(indicator, loc=3)
birth_death = [

    ('life_exp', 'SP.DYN.LE00.IN'), # Life expectancy at birth

    ('birth', 'SP.DYN.CBRT.IN'),    # Birth rate, crude 

    ('death', 'SP.DYN.CDRT.IN'),    # Death rate, crude 

    ('mort', 'SP.DYN.IMRT.IN'),     # Mortality rate, infant 

]



for indicator in birth_death:

    plot_indicator(indicator, loc=0)
education = [

    ('education', 'NY.ADJ.AEDU.GN.ZS'), # education expenditure 

    ('literacy', 'SE.ADT.LITR.ZS'),     # Adult literacy rate

    ('teach_prim', 'SE.PRM.ENRL.TC.ZS'),# Pupil-teacher ratio in primary education

    ('teach_sec', 'SE.SEC.ENRL.TC.ZS'), # Pupil-teacher ratio in secondary education 

]



for indicator in education:

    plot_indicator(indicator, xlim=(1995, 2015))
gdp = [

    ('gdp', 'NY.GDP.PCAP.KD'),          # GDP per capita 

    ('gdp_growth', 'NY.GDP.PCAP.KD.ZG'),# GDP per capita growth

]



for indicator in gdp:

    if indicator[0] == 'gdp_growth':

        plot_indicator(indicator, loc=0,

                       xlim=(1995, 2015), ylim=(-15, 15), 

                      )

    else:

        plot_indicator(indicator, loc=0, xlim=(1995, 2015))
unemployment = [

    ('unemp', 'SL.UEM.TOTL.ZS'),       # Unemployment, total 

    ('unemp_young', 'SL.UEM.1524.ZS'), # Unemployment, youth total 

]



for indicator in unemployment:

    plot_indicator(indicator, loc=2)
internet = [

    ('int_users', 'IT.NET.USER.P2'),    # Internet users 

    ('int_sups', 'IT.NET.BBND.P2'),     # Fixed broadband subscriptions 

]



for indicator in internet:

    plot_indicator(indicator, loc=0)
imp_exp = [

    ('export', 'NE.EXP.GNFS.KD'),   # Exports of goods and services 

    ('import', 'NE.IMP.GNFS.KD'),   # Imports of goods and services 

]



plot_indicator(imp_exp, drop_eu=True, loc=0, loc2=(0.01, 0.63),

               xlim=(1995, 2015), ylim=(0, 3.5e10),

               title='Imports and exports of goods and services (constant 2005 US$)')
energetics = [

    ('electro', 'EG.USE.ELEC.KH.PC'),   # Electric power consumption 

    ('energy', 'EG.USE.PCAP.KG.OE'),    # Energy use 

]



for indicator in energetics:

    plot_indicator(indicator, loc=0)
expenditures = [

    ('military', 'MS.MIL.XPND.GD.ZS'),  # Military expenditure 

    ('research', 'GB.XPD.RSDV.GD.ZS'),  # Research and development expenditure 

    ('health', 'SH.XPD.PUBL.ZS'),       # Health expenditure, public 

]



for indicator in expenditures:

    plot_indicator(indicator, xlim=(1995, 2015))