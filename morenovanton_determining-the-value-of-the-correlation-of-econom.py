import zipfile

import re

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import numpy as np

import seaborn as sns



from sklearn import preprocessing

from math import sqrt



from scipy import stats

from scipy import stats as st
Country = pd.read_csv('../input/education-statistics/edstats-csv-zip-32-mb-/EdStatsCountry.csv') 

gapminder = px.data.gapminder()
print(Country.shape)

Country.columns.values
Country = Country.rename(columns={'Table Name': 'T', 'Long Name': 'L', 'Country Code': 'CC', 

                                  '2-alpha code': 'AC','WB-2 code': 'WB', 'Other groups': 'og',

                                  'Balance of Payments Manual in use':'BPMuse', 

                                  'Special Notes': 'SN', 'Currency Unit': 'CU'})



Country = Country.drop(['T','L', 'CC', 'AC', 'WB', 'og', 'BPMuse', 'SN', 'CU'], axis=1)
l_contr = []

for i in Country[['System of National Accounts']].iterrows():

    countr = str(i[1])

    if 'NaN' not in countr:

        countr = re.sub(r'\D', '', countr[48:-51])  # countr[48:-51]

        l_contr.append(countr)

    else:

        l_contr.append('NaN')

        

for y in range(241):

    Country.at[y, 'System of National Accounts'] = l_contr[y]



Country.loc[Country['System of trade']=='General trade system', 'System of trade'] = 'General'

Country.loc[Country['System of trade']=='Special trade system', 'System of trade'] = 'Special'



Country.loc[Country['Government Accounting concept']=='Budgetary central government', 

            'Government Accounting concept'] = 'Government'

Country.loc[Country['Government Accounting concept']=='Consolidated central government', 

            'Government Accounting concept'] = 'Consolidated'



Country.loc[Country['IMF data dissemination standard']=='General Data Dissemination System (GDDS)', 

            'IMF data dissemination standard'] = 'GDDS'

Country.loc[Country['IMF data dissemination standard']=='Special Data Dissemination Standard (SDDS)', 

            'IMF data dissemination standard'] = 'SDDS'



Country.loc[Country['IMF data dissemination standard']=='Special Data Dissemination Standard (SDDS)', 

            'IMF data dissemination standard'] = 'SDDS'



Country.loc[Country['SNA price valuation']=='Value added at basic prices (VAB)', 

            'SNA price valuation'] = 'VAB'

Country.loc[Country['SNA price valuation']=='Value added at producer prices (VAP)', 

            'SNA price valuation'] = 'VAP'
tra_Country = Country[['National accounts base year', 'System of National Accounts', 'PPP survey year','Latest population census', 

                       'Latest agricultural census', 'Latest industrial data', 'Latest trade data', 'Latest water withdrawal data', 

                       'Alternative conversion factor', 'National accounts reference year']]

print(tra_Country.shape[0])
d = {}

for t in tra_Country.columns.values:

    d[t] = []

    for i in tra_Country['{}'.format(t)]:

        if type(i) is str:

            countr = re.sub(r'\D', '', i)[0:4]

            d[t].append(countr)

        else:

            d[t].append(i)

            

pop6, pop7, pop10 = [], [], []

for tim in range(tra_Country.shape[0]):

    x = d['National accounts base year'][tim]

    pop1 = filter(lambda x: type(x) is not float and x!='', d['National accounts base year'])

    

    y = d['System of National Accounts'][tim]

    pop2 = filter(lambda y: type(y) is not float and y!='', d['System of National Accounts'])

    

    e = d['PPP survey year'][tim]

    pop3 = filter(lambda e: type(e) is not float and e!='', d['PPP survey year'])

    

    w = d['Latest population census'][tim]

    pop4 = filter(lambda w: type(w) is not float and w!='', d['Latest population census'])

    

    q = d['Latest agricultural census'][tim]

    pop5 = filter(lambda q: type(q) is not float and q!='', d['Latest agricultural census'])

    

    st_1 = str(tra_Country['Latest industrial data'][tim])

    if st_1 != 'nan':

        pop6.append(int(st_1[0:4]))

    

    st_2 = str(tra_Country['Latest trade data'][tim])

    if st_2 != 'nan':

        pop7.append(int(st_2[0:4]))

    

    c = d['Latest water withdrawal data'][tim]

    pop8 = filter(lambda c: type(c) is not float and c!='', d['Latest water withdrawal data'])

    

    h = d['Alternative conversion factor'][tim]

    pop9 = filter(lambda h: type(h) is not float and h!='', d['Alternative conversion factor'])

    

    st_3 = str(tra_Country['National accounts reference year'][tim])

    if st_3 != 'nan':

        pop10.append(int(st_3[0:4]))



pop1 = list(map(int, pop1))

pop2 = list(map(int, pop2))

pop3 = list(map(int, pop3))

pop4 = list(map(int, pop4))

pop5 = list(map(int, pop5))

pop8 = list(map(int, pop8))

pop9 = list(map(int, pop9))





general_maen = [int(np.mean(pop1)), int(np.mean(pop2)), int(np.mean(pop3)), int(np.mean(pop4)), 

                int(np.mean(pop5)), int(np.mean(pop6)), int(np.mean(pop7)), int(np.mean(pop8)),

                int(np.mean(pop9)), int(np.mean(pop10))]



f = 0

for g in tra_Country.columns.values:

    for h in range(tra_Country.shape[0]):

        tra_Country.at[h, '{}'.format(g)] = d[g][h] 

        tra_Country.loc[tra_Country['{}'.format(g)]=='', '{}'.format(g)] = general_maen[f]

    tra_Country = tra_Country.fillna({'{}'.format(g): general_maen[f]})

    f+=1

    
ft_Country = pd.merge(Country[['Short Name', 'Region', 'Income Group']].reset_index(), tra_Country.reset_index(),

                      how='outer', on='index')

ft_Country = ft_Country.drop(['index'], axis=1)
Country_reg = ft_Country[['Short Name', 'Region']]

region_aggreg = ft_Country.groupby('Region').aggregate({'Short Name': 'count'})

region_aggreg.plot.bar();
EastAsia_Pacific_map = ft_Country.loc[ft_Country['Region'] == 'East Asia & Pacific']



EastAsia_Pacific_map = EastAsia_Pacific_map[['National accounts base year', 'System of National Accounts', 'PPP survey year',

                        'Latest population census', 'Latest agricultural census', 'Latest industrial data','Latest trade data', 

                        'Latest water withdrawal data', 'Alternative conversion factor', 'National accounts reference year']]



l = EastAsia_Pacific_map.transpose()



old_colum_list = l.columns.values

new_colum_list = ft_Country['Short Name'].values 

for t in range(len(old_colum_list)):

    old_colum = int(old_colum_list[t])

    new_colum = new_colum_list[old_colum]

    l = l.rename(columns={old_colum: new_colum})
cor_map = plt.cm.RdBu

plt.figure(figsize=(25,20))

plt.title('Pearson Correlation South_Asia', y=1.05, size=15)

sns.heatmap(l.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=cor_map, linecolor='white', annot=True)  
countrygap = []

for i in l.columns.values:

    if i in set(gapminder['country']):

        countrygap.append(i)

countrygap.append('Hong Kong, China')

countrygap.append('Korea, Dem. Rep.')

countrygap.append('Korea, Rep.')





df = pd.DataFrame()

for h in countrygap:

    gapminder_f = gapminder[(gapminder['country'] == '{}'.format(h))]

    df = pd.concat([df, gapminder_f])



ax = px.line(df, x='year', y='gdpPercap', color='country')

ax.show()
df_f = df[(df['country'] == 'Australia') | (df['country'] == 'Hong Kong, China')]

ax = sns.lmplot(x='year', y='gdpPercap', data = df_f, hue='country', fit_reg=False)
normalized_X = preprocessing.normalize(df_f[['year', 'gdpPercap']])

standardized_X = preprocessing.scale(df_f[['year', 'gdpPercap']])



year_standardized_list = []

gdpPercap_standardized_list = []

for h in range(len(standardized_X)):

    year_standardized_list.append(standardized_X[h][0])  

    gdpPercap_standardized_list.append(standardized_X[h][1])

standardized_df_f = {'standardized_year': pd.Series(year_standardized_list), 'standardized_gdpPercap': 

                   pd.Series(gdpPercap_standardized_list)}

standardized_df_f = pd.DataFrame(standardized_df_f)

df_f.loc[:,'standardized_year'] = list(standardized_df_f['standardized_year'])

df_f.loc[:,'standardized_gdpPercap'] = list(standardized_df_f['standardized_gdpPercap'])



ax_normalized = sns.lmplot(x='standardized_year', y='standardized_gdpPercap', data = df_f, fit_reg=True)    
x = list(df_f['standardized_year'])

y = list(df_f['standardized_gdpPercap'])   

r = stats.pearsonr(x,y)[0]

digres_freedom = df_f.shape[0]-2

print('r = {} digres_freedom = {}'.format(r,digres_freedom))

t = (r*sqrt(digres_freedom))/sqrt(1-(r**2))

# calculate the critical value

alpha = 0.05

cv = st.t.ppf(1.0 - alpha, digres_freedom)

# Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

p = st.t.sf(np.abs(t), digres_freedom)*2  

# вывод результата

print('t-statistic = {} p-value = {}'.format(t,p))

if abs(t) <= cv:

    print('Accept the null hypothesis.')

else:

    print('Reject the null hypothesis.')