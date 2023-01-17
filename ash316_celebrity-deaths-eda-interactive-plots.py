import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
deaths=pd.read_csv('../input/celebrity_deaths_4.csv',encoding='ISO-8859-1')
deaths.head(2) #checking the top 2 rows
deaths.isnull().sum() #checking for null values
for i in deaths[['cause_of_death','famous_for','nationality']]: #checking for no of unique values

    print('number of Unique Values for ',i,'are:',deaths[i].nunique())
deaths['cause_of_death'].fillna('unknown',inplace=True) #replacing the null values in cause_of_death and famous_for

deaths['famous_for'].fillna('unknown',inplace=True)

deaths.drop('fame_score',axis=1,inplace=True) #the fame_score was no use to me
deaths['nationality'].replace(['American','Canadian','German','British','Indian','Russian','Italian','French','Australian','English','Turkish','Irish','Israeli','Emirati','Jordanian','Indian-born','Korean','Syrian','Malaysian','Swedish','Bulgarian', 'Greek', 'Chilean', 'Finnish', 'Iraqi',

       'Austrian','Bangladeshi','Norwegian','Brazilian','Japanese','Dutch','Spanish','Scottish','Polish','Mexican','New','Argentine','Hungarian','Filipino','Romanian','Chinese','Belgian','Danish','Iranian','Pakistani','Ukrainian','Indonesian','Columbian','Nigerian','Swiss','Sri','Thai','Cuban','Taiwanese', 'Jamaican','Serbian','Colombian','Egyptian','Peruvian','Kenyan','Vietnamese','Tanzanian','Soviet','Hong','Argentinian','Singaporean','Canadian-born','German-born','Polish-born','Trinidadian','Trinidad','Namibian','Nepali','Portuguese','U.S.','Tibetan','Nepalese','Croatian','Afghan','Turkish-born','Spanish-born','Azerbaijani','Soviet-born','Zambian', 'Ghanaian'],

                              

                              ['United States','Canada','Germany','United Kingdom','India','Russia','Italy','France','Australia','United Kingdom','Turkey','Ireland','Israel','United Arab Emirates','Jordan',

                              'India','Korea','Syria','Malaysia','Sweden','Bulgaria','Greece','Chile','Finland','Iraq','Austria','Bangladesh','Norway','Brazil','Japan','Netherlands','Spain','Scotland','Poland','Mexico','New Zealand','Argentina','Hungary','Philippines','Romania','China','Belgium','Denmark','Iran','Pakistan','Ukraine','Indonesia','Columbia','Nigeria','Switzerland','Sri Lanka','Thailand','Cuba','Taiwan','Jamaica','Serbia','Colombia','Egypt','Peru','Kenya','Vietnam','Tanzania','Russia','Hongkong','Argentina','Singapore','Canada','Germany','Poland','Trinidad & Tobago','Trinidad & Tobago','Namibia','Nepal','Protugal','United States','Tibet','Nepal','Croatia','Afghanistan','Turkey','Spain','Azerbaijan','Russia','Zambia','Ghana'],inplace=True)

mask1 = deaths['nationality'] == 'South'

mask2 = deaths['famous_for'].str.contains('Korean')

mask3 = deaths['famous_for'].str.contains('Africa')



deaths.loc[mask1 & mask2, 'nationality']='South Korea'

deaths.loc[mask1 & mask3, 'nationality']='South Africa'
mask1 = deaths['famous_for'].str.contains('business|Business|entrepreneur')

mask2 = deaths['famous_for'].str.contains('Olympic|football|baseball|rugby|cricket|player|soccer|basketball|NFL|golf|hockey')

mask3 = deaths['famous_for'].str.contains('music|Music|sing|Sing|musician|composer|song|Song')

mask4 = deaths['famous_for'].str.contains('politician|minister|Politician|Minister|Parliament|parliament|Governor|governor|leader|Leader|council|Council|Assembly|Mayor|mayor')

mask5 = deaths['famous_for'].str.contains('actor|Actor|actress|Actress|film|Film|cinema|screenwriter')

mask6 = deaths['famous_for'].str.contains('author|Author')

mask7 = deaths['famous_for'].str.contains('Engineer|engineer')

mask8 = deaths['famous_for'].str.contains('Military|military|army|Army')



deaths.loc[True & mask1, 'famous_for']='Business'

deaths.loc[True & mask2, 'famous_for']='Sports'

deaths.loc[True & mask3, 'famous_for']='Music'

deaths.loc[True & mask4, 'famous_for']='Politics'

deaths.loc[True & mask5, 'famous_for']='Movies'

deaths.loc[True & mask6, 'famous_for']='Authors'

deaths.loc[True & mask7, 'famous_for']='Engineers'

deaths.loc[True & mask8, 'famous_for']='Military'

deaths['famous_for'].nunique()
deaths['age_category']='' #adding a age category for the celebs

mask1=deaths['age']<18

mask2=(deaths['age']>=18) & (deaths['age']<30)

mask3=(deaths['age']>=30) & (deaths['age']<=60)

mask4=deaths['age']>60



deaths.loc[True & mask1, 'age_category']='Children'

deaths.loc[True & mask2, 'age_category']='Young'

deaths.loc[True & mask3, 'age_category']='Middle-Age'

deaths.loc[True & mask4, 'age_category']='Old'
deaths.head()
deaths['age'].hist(color='#ff0cd5')

plt.axvline(deaths['age'].mean(),linestyle='dashed',color='blue')
data = [go.Bar(

            x=deaths['death_year'].value_counts().index,

            y=deaths['death_year'].value_counts().values,

        marker = dict(

        color = 'rgba(255, 0, 0,0.8)',)

            

)]



py.iplot(data, filename='horizontal-bar')
type_deaths=deaths[deaths['cause_of_death']!='unknown']

type_deaths=type_deaths['cause_of_death'].value_counts()[:10]

data = [go.Bar(

            x=type_deaths.index,

            y=type_deaths.values,

        marker = dict(

        color = 'rgba(190, 130, 30,0.8)',)

            

)]



py.iplot(data, filename='horizontal-bar')
deaths_2=deaths.copy()



deaths_2.loc[deaths_2.cause_of_death.str.contains('cancer|Cancer'),'cause_of_death']='Cancer'

deaths_2.loc[deaths_2.cause_of_death.str.contains('heart|Heart|Cardiac|cardiac'),'cause_of_death']='Heart Problems'

deaths_2.loc[deaths_2.cause_of_death.str.contains('brain|Brain|stroke'),'cause_of_death']='Brain Complications'

deaths_2.loc[deaths_2.cause_of_death.str.contains('Alzheimer|alzheimer'),'cause_of_death']='Alzheimers'

deaths_2.loc[deaths_2.cause_of_death.str.contains('Parkinson|parkinson'),'cause_of_death']='Parkinsons'

deaths_2.loc[deaths_2.cause_of_death.str.contains('Suicide|suicide'),'cause_of_death']='Suicide'

deaths_2.loc[deaths_2.cause_of_death.str.contains('accident|Accident|crash|collision'),'cause_of_death']='Accident'

deaths_2.loc[deaths_2.cause_of_death.str.contains('Shot|shot|murder|Murder'),'cause_of_death']='Murdered'
deaths_2=deaths_2[deaths_2['cause_of_death']!='unknown']

index=list(deaths_2['cause_of_death'].value_counts()[:10].index)

values=list(deaths_2['cause_of_death'].value_counts()[:10].values)



data = [go.Bar(

            x=index,

            y=values,

        marker = dict(

        color = 'rgba(190, 130, 130,0.8)',)

            

)]



py.iplot(data, filename='horizontal-bar')
dd1=deaths_2.groupby(['cause_of_death','death_year'])['death_month'].count().reset_index()

dd1=dd1[dd1['cause_of_death'].isin(deaths_2['cause_of_death'].value_counts()[:10].index)]

dd1.columns=[['cause','year','count']]

dd1.pivot('year','cause','count').plot(marker='o',colormap='Paired')

fig=plt.gcf()

fig.set_size_inches(12,6)
data = [go.Bar(

            x=deaths['nationality'].value_counts()[:10].index,

            y=deaths['nationality'].value_counts()[:10].values,

        marker = dict(

        color = 'rgba(41, 221, 239,0.8)',)

            

)]



py.iplot(data, filename='horizontal-bar')
top_countries=deaths['nationality'].value_counts()[:15]

all_deaths=deaths['cause_of_death'].value_counts()

countries=deaths[deaths['nationality'].isin(top_countries.index)]

countries=countries[countries['cause_of_death'].isin(all_deaths.index)]

countries=countries[['cause_of_death','death_year','nationality']]

countries=countries.groupby(['nationality','death_year'])['cause_of_death'].count().reset_index()

countries.columns=[['nationality','death_year','count']]

countries=countries.pivot('nationality','death_year','count')

sns.heatmap(countries,cmap='Set3',annot=True,fmt='2.0f',linewidths=0.4)

plt.title('Total Deaths By Countries Per Year')

fig=plt.gcf()

fig.set_size_inches(12,12)
_deaths=deaths[deaths['cause_of_death']!='unknown']

type_deaths=_deaths['cause_of_death'].value_counts()[:15]

type_deaths.index

deaths_1=deaths[deaths['famous_for'].isin(['Business','Sports','Politics','Movies','Authors','Engineers','Military','Music'])]

deaths_1=deaths_1[deaths_1['cause_of_death']!='unknown']

deaths_1=deaths_1[deaths_1['cause_of_death'].isin(type_deaths.index)] 

deaths_1=deaths_1.groupby(['famous_for','cause_of_death'])['death_year'].count().reset_index()

deaths_1=deaths_1.pivot('famous_for','cause_of_death','death_year')

sns.heatmap(deaths_1,cmap='RdYlGn',annot=True,fmt='2.0f',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(15,6)
_countries=deaths[deaths['cause_of_death']!='unknown']

_countries=_deaths['nationality'].value_counts()[:10]

deaths_1=deaths[deaths['nationality'].isin(_countries.index)]

deaths_1=deaths_1[deaths_1['cause_of_death']!='unknown']

deaths_1=deaths_1[deaths_1['cause_of_death'].isin(type_deaths.index)] 

deaths_1=deaths_1.groupby(['cause_of_death','nationality'])['death_year'].count().reset_index()

deaths_1=deaths_1.pivot('cause_of_death','nationality','death_year')

sns.heatmap(deaths_1,cmap='RdYlGn',annot=True,fmt='2.0f',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(8,8)
l1=list(['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra',

       'Angola', 'Anguilla', 'Antigua and Barbuda', 'Argentina', 'Armenia',

       'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The',

       'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize',

       'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',

       'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei',

       'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cabo Verde',

       'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands',

       'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',

       'Comoros', 'Congo, Democratic Republic of the',

       'Congo, Republic of the', 'Cook Islands', 'Costa Rica',

       "Cote d'Ivoire", 'Croatia', 'Cuba', 'Curacao', 'Cyprus',

       'Czech Republic', 'Denmark', 'Djibouti', 'Dominica',

       'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',

       'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia',

       'Falkland Islands (Islas Malvinas)', 'Faroe Islands', 'Fiji',

       'Finland', 'France', 'French Polynesia', 'Gabon', 'Gambia, The',

       'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland',

       'Grenada', 'Guam', 'Guatemala', 'Guernsey', 'Guinea-Bissau',

       'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary',

       'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',

       'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey',

       'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea, North',

       'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',

       'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein',

       'Lithuania', 'Luxembourg', 'Macau', 'Macedonia', 'Madagascar',

       'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',

       'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',

       'Micronesia, Federated States of', 'Moldova', 'Monaco', 'Mongolia',

       'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',

       'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua',

       'Nigeria', 'Niger', 'Niue', 'Northern Mariana Islands', 'Norway',

       'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea',

       'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',

       'Puerto Rico', 'Qatar', 'Romania', 'Russia', 'Rwanda',

       'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin',

       'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines',

       'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia',

       'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',

       'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands',

       'Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka',

       'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria',

       'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste',

       'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',

       'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',

       'United Arab Emirates', 'United Kingdom', 'United States',

       'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',

       'Virgin Islands', 'West Bank', 'Yemen', 'Zambia', 'Zimbabwe']) #Country names
l2=list(['AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATG', 'ARG',

       'ARM', 'ABW', 'AUS', 'AUT', 'AZE', 'BHM', 'BHR', 'BGD', 'BRB',

       'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BIH', 'BWA',

       'BRA', 'VGB', 'BRN', 'BGR', 'BFA', 'MMR', 'BDI', 'CPV', 'KHM',

       'CMR', 'CAN', 'CYM', 'CAF', 'TCD', 'CHL', 'CHN', 'COL', 'COM',

       'COD', 'COG', 'COK', 'CRI', 'CIV', 'HRV', 'CUB', 'CUW', 'CYP',

       'CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ',

       'ERI', 'EST', 'ETH', 'FLK', 'FRO', 'FJI', 'FIN', 'FRA', 'PYF',

       'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 'GIB', 'GRC', 'GRL', 'GRD',

       'GUM', 'GTM', 'GGY', 'GNB', 'GIN', 'GUY', 'HTI', 'HND', 'HKG',

       'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR',

       'ITA', 'JAM', 'JPN', 'JEY', 'JOR', 'KAZ', 'KEN', 'KIR', 'KOR',

       'PRK', 'KSV', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR',

       'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MKD', 'MDG', 'MWI', 'MYS',

       'MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA',

       'MCO', 'MNG', 'MNE', 'MAR', 'MOZ', 'NAM', 'NPL', 'NLD', 'NCL',

       'NZL', 'NIC', 'NGA', 'NER', 'NIU', 'MNP', 'NOR', 'OMN', 'PAK',

       'PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'PRI',

       'QAT', 'ROU', 'RUS', 'RWA', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT',

       'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP',

       'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SSD', 'ESP', 'LKA',

       'SDN', 'SUR', 'SWZ', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA',

       'THA', 'TLS', 'TGO', 'TON', 'TTO', 'TUN', 'TUR', 'TKM', 'TUV',

       'UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VUT', 'VEN',

       'VNM', 'VGB', 'WBG', 'YEM', 'ZMB', 'ZWE']) #Country Codes
df=pd.DataFrame(l1,l2)

df.reset_index(inplace=True)

df.columns=[['Code','Country']]
dea1=deaths.merge(df,left_on='nationality',right_on='Country',how='outer')

dea1=dea1.groupby('Country')['death_year'].count().reset_index().sort_values(by='death_year',ascending=False)

dea1=dea1.merge(df,left_on='Country',right_on='Country',how='right')

dea2=dea1[dea1['death_year']!=0]

dea2.columns=[['Country','Deaths_Count','Code']]

dea2.shape
data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = dea2['Code'],

        z = dea2['Deaths_Count'],

        locationmode = 'Code',

        text = dea2['Country'].unique(),

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Deaths')

            )

       ]



layout = dict(

    title = 'Total Deaths By Country',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,0,0)',

        projection = dict(

        type = 'Mercator',

            

        ),

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2010')
deaths_cancer=deaths[deaths['cause_of_death'].str.contains('cancer')]

deaths_cancer=deaths_cancer.groupby(['cause_of_death'])['death_year'].count().reset_index()

deaths_cancer=deaths_cancer.sort_values(by='death_year',ascending=False)[1:15]

sns.barplot(x='death_year',y='cause_of_death',data=deaths_cancer,palette='RdYlGn').set_title('Top Types Of Death Causing Cancer')

plt.xlabel('Total Deaths')

plt.ylabel('Type of cancer')

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()
deaths_2=deaths[deaths['cause_of_death'].isin(deaths_cancer['cause_of_death'])]

abc=deaths_2.groupby(['cause_of_death','death_year'])['death_month'].count().reset_index()

deaths=deaths_2

abc.columns=[['cancer_type','death year','Count']]

abc=abc.pivot('death year','cancer_type','Count')

abc.plot(marker='o',colormap='Paired_r')

ticks=range(2006,2017)

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.xticks(ticks)
deaths_cancer=deaths.copy()

mask1 = deaths_cancer['cause_of_death'].str.contains('cancer|Cancer')

deaths_cancer.loc[True & mask1, 'cause_of_death']='Cancer'

deaths_cancer=deaths_cancer[deaths_cancer['cause_of_death']=="Cancer"]

deaths_cancer=deaths_cancer.groupby(['nationality'])['death_year'].count().reset_index()

deaths_cancer=deaths_cancer.merge(df,left_on='nationality',right_on='Country',how='right')

deaths_cancer.dropna(inplace=True)

deaths_cancer=deaths_cancer[['nationality','death_year','Code']]

deaths_cancer.columns=[['Country','Count','Code']]



data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = deaths_cancer['Code'],

        z = deaths_cancer['Count'],

        locationmode = 'Code',

        text = deaths_cancer['Country'],

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Deaths')

            )

       ]



layout = dict(

    title = 'Total Deaths By Cancer',

    geo = dict(

        showframe = True,

        showocean = True,

        coastlines = True,

        oceancolor = 'rgb(0,0,0)',

        projection = dict(

        type = 'Mercator',

            

        ),

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2010')
year2016=deaths[deaths['death_year']==2016]

before2016=deaths[deaths['death_year']!=2016]

df1=year2016['nationality'].value_counts().reset_index()

df2=before2016['nationality'].value_counts().reset_index()

df1=df1.merge(df2,left_on='index',right_on='index',how='right')[1:17]

df1.columns=[['country','2016','before 2016']]
x=list(df1['country'].values)

y=list(df1['before 2016'].values)

y1=list(df1['2016'].values)

trace0 = go.Bar(

    x=x,

    y=y,

    name='Deaths before 2016',

    

)

trace1 = go.Bar(

    x=x,

    y=y1,

    name='Deaths 2016',

    

)



data = [trace0, trace1]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='stack',

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='angled-text-bar')
year2016=deaths_2[deaths_2['death_year']==2016]

before2016=deaths_2[deaths_2['death_year']!=2016]

df1=year2016['cause_of_death'].value_counts().reset_index()[1:]

df2=before2016['cause_of_death'].value_counts().reset_index()[1:]

df1=df1.merge(df2,left_on='index',right_on='index',how='right')[:10]

df1.columns=[['cause','2016','before 2016']]



x=list(df1['cause'].values)

y=list(df1['before 2016'].values)

y1=list(df1['2016'].values)

trace0 = go.Bar(

    x=x,

    y=y,

    name='Deaths before 2016',

    marker=dict(

                color='rgb(158,225,225)',)

    

)

trace1 = go.Bar(

    x=x,

    y=y1,

    name='Deaths 2016',

    

)



data = [trace0, trace1]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='stack',

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='angled-text-bar')
year2016=deaths[deaths['death_year']==2016]

dea1=year2016.merge(df,left_on='nationality',right_on='Country',how='outer')

dea1=dea1.groupby('Country')['death_year'].count().reset_index().sort_values(by='death_year',ascending=False)

dea1=dea1.merge(df,left_on='Country',right_on='Country',how='right')

dea2=dea1[dea1['death_year']!=0]

dea2.columns=[['Country','Deaths_Count','Code']]





data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = dea2['Code'],

        z = dea2['Deaths_Count'],

        locationmode = 'Code',

        text = dea2['Country'].unique(),

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Deaths')

            )

       ]



layout = dict(

    title = 'Total Deaths By Country in 2016',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'Mercator',

            

        ),

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2010')