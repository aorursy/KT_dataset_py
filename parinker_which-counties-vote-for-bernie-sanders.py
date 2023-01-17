import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/primary_results.csv")
county_facts = pd.read_csv("../input/county_facts.csv")
county_facts_desc = pd.read_csv("../input/county_facts_dictionary.csv")
df = pd.merge(df, county_facts, on="fips")
df.head()
# create a dictionary of shortened names for ease of plotting
county_facts_short_dict = {
 'AFN120207': 'AccomFoodSales',
 'AGE135214': 'AgeUnder5Perc',
 'AGE295214': 'AgeUnder18Perc',
 'AGE775214': 'AgeOver65Perc',
 'BPS030214': 'BuildingPermits',
 'BZA010213': 'PrivateEstabs',
 'BZA110213': 'PrivateEmploy',
 'BZA115213': 'PrivateEmployPerChange',
 'EDU635213': 'HSGradOrHigher',
 'EDU685213': "BacherlorsOrHigher",
 'HSD310213': 'PersonsPerHousehold',
 'HSD410213': 'Households',
 'HSG010214': 'HousingUnits',
 'HSG096213': 'HousingUnitsInMulti',
 'HSG445213': 'HomeownershipRate',
 'HSG495213': 'MedianHousingValue',
 'INC110213': 'MedianHHIncome',
 'INC910213': 'PerCapitaIncomePastYear',
 'LFE305213': 'MeanTravelTimeToWork',
 'LND110210': 'LandArea',
 'MAN450207': 'ManufactShipments',
 'NES010213': 'NonemployerEstabls',
 'POP010210': 'Population2010',
 'POP060210': 'PopulationPerSquareMile2010',
 'POP645213': 'ForeignBornPerc',
 'POP715213': 'LiveSameHouseOneYearPerc',
 'POP815213': 'OtherLanguagePerc',
 'PST040210': 'Population2010EstimatesBase',
 'PST045214': 'Population2014',
 'PST120214': 'PopulationPercChange',
 'PVY020213': 'PercBelowPoverty',
 'RHI125214': 'WhitePerc',
 'RHI225214': 'BlackPerc',
 'RHI325214': 'AIANPerc',
 'RHI425214': 'AsianPerc',
 'RHI525214': 'NHPIPerc',
 'RHI625214': 'TwoOrMorePerc',
 'RHI725214': 'HispPerc',
 'RHI825214': 'NonHispWhitePerc',
 'RTN130207': 'RetailSales',
 'RTN131207': 'RetailSalesPerCapita',
 'SBO001207': 'NoOfFirms',
 'SBO015207': 'PercFirmsWomenOwned',
 'SBO115207': 'PercFirmsAIANOwned',
 'SBO215207': 'PercFirmsAsianOwned',
 'SBO315207': 'PercFirmsBlackOwned',
 'SBO415207': 'PercFirmsHispOwned',
 'SBO515207': 'PercFirmsNHPIOwned',
 'SEX255214': 'PercWomen',
 'VET605213': 'Veterans',
 'WTN220207': 'MerchantWholesalerSales'}
df.columns
df.groupby('state_abbreviation_x').agg(np.sum).sort_values(by='votes', ascending=False)['votes']
# let's look at how many votes each candidate recieved
df.groupby('candidate').agg(np.sum).sort_values(by='votes', ascending=False)['votes']
# do a quick plot of each variable for Sanders
cols = ['fraction_votes', 'PST045214', 'PST040210', 'PST120214', 'POP010210', 'AGE135214',
       'AGE295214', 'AGE775214', 'SEX255214', 'RHI125214', 'RHI225214',
       'RHI325214', 'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214',
       'RHI825214', 'POP715213', 'POP645213', 'POP815213', 'EDU635213',
       'EDU685213', 'VET605213', 'LFE305213', 'HSG010214', 'HSG445213',
       'HSG096213', 'HSG495213', 'HSD410213', 'HSD310213', 'INC910213',
       'INC110213', 'PVY020213', 'BZA010213', 'BZA110213', 'BZA115213',
       'NES010213', 'SBO001207', 'SBO315207', 'SBO115207', 'SBO215207',
       'SBO515207', 'SBO415207', 'SBO015207', 'MAN450207', 'WTN220207',
       'RTN130207', 'RTN131207', 'AFN120207', 'BPS030214', 'LND110210',
       'POP060210']

sanders = df.loc[df.candidate == 'Bernie Sanders', :]
sanders_melt = pd.melt(sanders.loc[:, cols], id_vars='fraction_votes')
sanders_melt['variable_desc'] = sanders_melt.variable.replace(county_facts_short_dict)
sanders_melt_test = sanders_melt.sample(frac=0.25)   # useful to test plotting options
# one-off linear regression for each variable
# warning: this may take some time to run
sns.lmplot(data=sanders_melt, x='value', y='fraction_votes', col="variable_desc", 
           col_wrap=3, ci=None, robust=True, sharex=False).set_titles("{col_name}");
