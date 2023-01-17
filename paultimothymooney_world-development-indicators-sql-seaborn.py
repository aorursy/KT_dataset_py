import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

import os

%matplotlib inline

import warnings
conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()
for row in c.execute(

                    # SQL statement 

                    """

                        SELECT   * 

                        FROM     Country 

                        LIMIT    1

                        

                     """ ):

    print(row)
Country = pd.read_sql(

                       """

                       

                        SELECT  * 

                        FROM    Country

                        

                       """, con=conn)

Country.head(3)
pd.read_sql(

            """

                SELECT   * 

                FROM     Country 

                LIMIT    3 

                

            """, con=conn)
pd.read_sql(

        """ 

            SELECT   * 

            FROM     Country 

            WHERE    CountryCode = 'USA'

            

        """, con=conn)

# same as 

# Country[Country['CountryCode'] == 'USA']
pd.read_sql(

        """ 

            SELECT      Region

                        ,COUNT(*) AS [Count]

            FROM        Country 

            GROUP BY    Region

            ORDER BY    2 DESC

        """, con=conn)
# LEFT JOIN

pd.read_sql(

        """ 

           

            SELECT      A.CountryCode

                        ,B.ShortName

                        ,B.CurrencyUnit

                        ,B.IncomeGroup

            FROM       ( 

                            -- First subquery (i.e the Left table)

                            

                           SELECT      CountryCode

                                        ,ShortName

                                        ,CurrencyUnit

                                        ,IncomeGroup

                           FROM        Country

                           WHERE       CountryCode IN ('USA','MEX', 'GBR', 'FRA')

                        ) AS A

            LEFT JOIN   (

                            -- Second subquery (i.e the right table )

                            

                            SELECT      CountryCode

                                        ,ShortName

                                        ,CurrencyUnit

                                        ,IncomeGroup

                            FROM        Country AS A

                            WHERE       CountryCode IN ('USA','MEX', 'URY', 'BEL')

                            

                          ) AS B

            ON          A.CountryCode = B.CountryCode    

            

        """, con=conn)
# UNION 

pd.read_sql(

        """ 

                           SELECT      CountryCode

                                        ,ShortName

                                        ,CurrencyUnit

                                        ,IncomeGroup

                           FROM        Country

                           WHERE       CountryCode IN ('USA','MEX', 'GBR', 'FRA')

                       

                           UNION

                           

                           SELECT      CountryCode

                                        ,ShortName

                                        ,CurrencyUnit

                                        ,IncomeGroup

                           FROM        Country AS A

                           WHERE       CountryCode IN ('USA','MEX', 'URY', 'BEL')

            

        """, con=conn)
# INTERSECT 

pd.read_sql(

        """ 

                           SELECT      CountryCode

                                        ,ShortName

                                        ,CurrencyUnit

                                        ,IncomeGroup

                           FROM        Country

                           WHERE       CountryCode IN ('USA','MEX', 'GBR', 'FRA')

                       

                           INTERSECT

                           

                           SELECT      CountryCode

                                        ,ShortName

                                        ,CurrencyUnit

                                        ,IncomeGroup

                           FROM        Country AS A

                           WHERE       CountryCode IN ('USA','MEX', 'URY', 'BEL')

            

        """, con=conn)
Indicators = pd.read_sql(""" SELECT   * 

                             FROM     Indicators 

                             WHERE    IndicatorCode IN 

                                      (  'AG.LND.PRCP.MM, AG.LND.FRST.K2'

                                       , 'EG.ELC.ACCS.ZS', 'EG.ELC.FOSL.ZS'

                                       , 'EN.POP.DNST', 'SG.VAW.REAS.ZS'

                                       , 'SM.POP.NETM', 'SP.POP.65UP.TO.ZS'

                                       , 'FI.RES.TOTL.DT.ZS', 'GC.DOD.TOTL.GD.ZS'

                                       , 'MS.MIL.XPND.GD.ZS','SI.POV.GINI'

                                       , 'IP.JRN.ARTC.SC', 'SE.ADT.1524.LT.ZS'

                                      )  

                        """, con=conn)

#Regions = ['ARB', 'EUU', 'LCN' , 'NAC',  'EAS', 'SSF', 'World']

gini = Indicators[Indicators['IndicatorCode']== 'SI.POV.GINI']

gini.CountryCode.unique()



# Plotting a Subplot of the Seaborn regplot

f, ((ax1, ax2, ax3), (ax4,ax5,ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(12,10))



# Plot of GINI index of China

points = ax1.scatter(gini[gini['CountryCode'] == 'CHN']["Year"], gini[gini['CountryCode'] == 'CHN']["Value"],

                     c=gini[gini['CountryCode'] == 'CHN']["Value"], s=100, cmap="viridis")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CHN'], ax=ax1)

ax1.set_title("GINI Index of China")



# Plot of GINI of Argentina

points = ax2.scatter(gini[gini['CountryCode'] == 'ARG']["Year"], gini[gini['CountryCode'] == 'ARG']["Value"],

                     c=gini[gini['CountryCode'] == 'ARG']["Value"], s=85, cmap="viridis")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ARG'], ax=ax2)

ax2.set_title("GINI Index of Argentina")



points = ax3.scatter(gini[gini['CountryCode'] == 'UGA']["Year"], gini[gini['CountryCode'] == 'UGA']["Value"],

                     c=gini[gini['CountryCode'] == 'UGA']["Value"], s=100, cmap="afmhot")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'UGA'], ax=ax3)

ax3.set_title("GINI Index of Uganda")



points = ax4.scatter(gini[gini['CountryCode'] == 'USA']["Year"], gini[gini['CountryCode'] == 'USA']["Value"],

                     c=gini[gini['CountryCode'] == 'USA']["Value"], s=100, cmap="Purples_r")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'USA'], ax=ax4)

ax4.set_title("GINI Index of USA")



points = ax5.scatter(gini[gini['CountryCode'] == 'COL']["Year"], gini[gini['CountryCode'] == 'COL']["Value"],

                     c=gini[gini['CountryCode'] == 'COL']["Value"], s=100, cmap="YlOrBr")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'COL'], ax=ax5)

ax5.set_title("GINI Index of Colombia")



points = ax6.scatter(gini[gini['CountryCode'] == 'RWA']["Year"], gini[gini['CountryCode'] == 'RWA']["Value"],

                     c=gini[gini['CountryCode'] == 'RWA']["Value"], s=100, cmap="Blues")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'RWA'], ax=ax6)

ax6.set_title("GINI Index of Rwanda")



points = ax7.scatter(gini[gini['CountryCode'] == 'RUS']["Year"], gini[gini['CountryCode'] == 'RUS']["Value"],

                     c=gini[gini['CountryCode'] == 'RUS']["Value"], s=100, cmap="Blues")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'RUS'], ax=ax7)

ax7.set_title("GINI Index of Russia")



points = ax8.scatter(gini[gini['CountryCode'] == 'ECU']["Year"], gini[gini['CountryCode'] == 'ECU']["Value"],

                     c=gini[gini['CountryCode'] == 'ECU']["Value"], s=100, cmap="winter")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ECU'], ax=ax8)

ax8.set_title("GINI Index of Ecuador")



points = ax9.scatter(gini[gini['CountryCode'] == 'CAF']["Year"], gini[gini['CountryCode'] == 'CAF']["Value"],

                     c=gini[gini['CountryCode'] == 'CAF']["Value"], s=100, cmap="magma")

sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CAF'], ax=ax9)

ax9.set_title("GINI Index of Central African Republic")

sns.set_style(style="dark")

plt.tight_layout()