# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/who_suicide_statistics.csv")
data.head()
data.shape
data.loc[:,'age'] = data['age'].str.replace('years','')
data.loc[data['age'] == '5-14', 'age'] = '05-14'

#Crud rate calculation
data['crude_rate_per_100k'] = data['suicides_no'] / (data['population'] / 100000)

#Preview
pd.concat([data[:2], data[10000:10002], data[-2:]])
data.shape

data.head()
sns.countplot(x='sex', data=data)
sns.set()

cd = (data.loc[(data['country'].isin(['India', 'Poland','United States of America'])) &
              (data['year']==2015),['country', 'sex','age', 'crude_rate_per_100k']]
      .sort_values(['sex','age']))

sns.catplot(x='age', hue='sex', col='country', y='crude_rate_per_100k'
            , data=cd, kind='bar', col_wrap=3)

bd = (data.loc[(data['country'].isin(['Poland'])) &
              (data['year']),['country', 'sex','age','suicides_no' 'crude_rate_per_100k']]
      .sort_values(['sex','age',]))
plt.figure(figsize=(20,8))
sns.countplot(x='country', hue='age', data=bd)
data.isnull().any()
data= data.dropna()
data.isnull().any()
#Grouping into two decades of interest
data.loc[(data['year'] >= 1996) & (data['year'] <= 2005), 'decade'] ='1996-2005'
data.loc[(data['year'] >= 2006) & (data['year'] <= 2015), 'decade'] = '2006-2015'


# median of the crude rate in a decade, along with information about how many years of data are available
stats_by_decade_df = (data.groupby(['country','decade','sex','age'])['crude_rate_per_100k']
                      .agg(['median','count'])
                      .reset_index()
                     )

stats_by_decade_df.head()
data.head()
# select only these cases where we have at least X years of data
valid_threshold = 5

# fraction_above_threshold = sum(stats_by_decade_df['count'] >= valid_threshold) / len(stats_by_decade_df)
# print('There are {0:.1%} rows with at least {1} years of data.'
#       .format(fraction_above_threshold, valid_threshold))
# difference in median of crude rates between two decades
# perhaps there is a better way to calculate this, instead of doing two calculations and joining them into one table
d1_df = stats_by_decade_df[(stats_by_decade_df['decade'] == '1996-2005') & (stats_by_decade_df['count'] >= valid_threshold)]
d2_df = stats_by_decade_df[(stats_by_decade_df['decade'] == '2006-2015') & (stats_by_decade_df['count'] >= valid_threshold)]

final_df = d1_df.merge(d2_df, left_on=['country','sex','age'], 
            right_on=['country','sex','age'], how='inner',
           suffixes=['_d1','_d2'])
           
final_df['crude_rate_diff'] = final_df['median_d2'] - final_df['median_d1']

# capping on crude rates difference
final_df.loc[final_df['crude_rate_diff'] > 20, 'crude_rate_diff'] = 20
final_df.loc[final_df['crude_rate_diff'] < -20, 'crude_rate_diff'] = -20

print('There are', final_df['country'].nunique(), 'countries with at least', valid_threshold, 'years of data.')
final_df.head()
sns.catplot(x='sex', y="crude_rate_diff", col="age", col_wrap=3, sharey=True, data=final_df, alpha=0.5)
# decide what qualifies as change (increase or decrease)
def categorize_differences(x, threshold):
    if (x <= -threshold):
        diff_category = 'Decrease'
    elif (x >= threshold):
        diff_category = 'Increase'
    else:
        diff_category = 'No change'
    
    return diff_category
        
final_df['crude_rate_diff_cat'] = final_df['crude_rate_diff'].map(lambda x: categorize_differences(x, 2))

# results
sns.catplot(x='sex', hue='crude_rate_diff_cat', col='age', col_wrap=3, 
            data=final_df.sort_values(['age','crude_rate_diff_cat','sex'], ascending=[True,False,True]), 
            kind='count')                                  
                                                                  
# let's rerun all the stuff just once more
data.loc[(data['year'] >= 1996) & (data['year'] <= 2005), 'decade'] = '1996-2005'
data.loc[(data['year'] >= 1986) & (data['year'] <= 1995), 'decade'] = '1986-1995'

stats_by_decade_df = (data.groupby(['country','decade','sex','age'])['crude_rate_per_100k']
                      .agg(['median','count'])
                      .reset_index()
                     )

d1_df = stats_by_decade_df[(stats_by_decade_df['decade'] == '1986-1995') & (stats_by_decade_df['count'] >= valid_threshold)]
d2_df = stats_by_decade_df[(stats_by_decade_df['decade'] == '1996-2005') & (stats_by_decade_df['count'] >= valid_threshold)]

final_df = d1_df.merge(d2_df, left_on=['country','sex','age'], 
            right_on=['country','sex','age'], how='inner',
           suffixes=['_d1','_d2'])
           
final_df['crude_rate_diff'] = final_df['median_d2'] - final_df['median_d1']

final_df.loc[final_df['crude_rate_diff'] > 20, 'crude_rate_diff'] = 20
final_df.loc[final_df['crude_rate_diff'] < -20, 'crude_rate_diff'] = -20

print('There are', final_df['country'].nunique(), 'countries with at least', valid_threshold, 'years of data.')
sns.catplot(x='sex', y="crude_rate_diff", col="age", col_wrap=3, sharey=True, data=final_df, alpha=0.5)

final_df['crude_rate_diff_cat'] = final_df['crude_rate_diff'].map(lambda x: categorize_differences(x, 2))

sns.catplot(x='sex', hue='crude_rate_diff_cat', col='age', col_wrap=3, 
            data=final_df, kind='count')
data1 = pd.read_csv("../input/who_suicide_statistics.csv")
data1.head()
data1.shape

data1.info()




age_coder = {'5-14 years':0,
            '15-24 years':1,
            '25-34 years':2,
            '35-54 years':3,
            '55-74 years':4,
            '75+ years':5}
gender_coder = {'female':0,'male':1}
data1['age_encoder'] = data1['age'].map(age_coder)
data1['sex_encoder'] = data1['sex'].map(gender_coder)
data1.head()


data1.suicides_no.fillna(0,inplace=True)
suicide = data1.groupby('age_encoder')[['suicides_no']].sum().plot()
suicide = data1.groupby('age_encoder')[['suicides_no']].sum()
#Suicide based on age groups
en = {0:'5-14 years',
      1:'15-24 years',
      2:'25-34 years',
      3:'35-54 years',
      4:'55-74 years',
      5:'75+ years'}
gen = {0:'female',1:'male'}

plt.figure(figsize=(12,5))
sns.barplot(x=suicide.index.map(en.get),y=suicide.suicides_no)
plt.title("Total Suicide based in Age group")
plt.xlabel("Age Group")
plt.ylabel("Number of Suicide")

#Total Suicide male and female
#Total Suicide male and female
male_suicide = data1[data1.sex_encoder == 0]['suicides_no'].values.sum()
female_suicide = data1[data1.sex_encoder == 1]['suicides_no'].values.sum()

age_differance = pd.DataFrame([male_suicide,female_suicide],index=['male','female'])
age_differance.head()
age_differance.plot(kind='bar',title="Total Suicide Based In Gender")
plt.legend()
#Suicide of top most 10 country
plt.figure(figsize=(14,6))
data.groupby('country').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']][:10].plot(kind='bar',figsize=(16,8),title='Suicide Based in Country')
d = data.groupby('country').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']][:10]
data1['sucide_by_per_100k'] =data1['suicides_no'] / (data1['population'] / 10000)
#Total Suicide on each year in descending order
data1.groupby('year').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']].plot(kind='bar',figsize=(16,8),title='Suicide Based in Year')
data1.groupby('age').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']].plot(kind='bar',figsize=(16,8),title='Suicide Based in Year')
age = data1.groupby('age').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']]
age
sucide_per_year = data1.groupby('year').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']]
sucide_per_year.head()
suicide_data = data.groupby(['year','age']).sum()['suicides_no'].reset_index()
suicide_data
suicide_data.plot()
#Sucide Based On Year and age
plt.figure(figsize=(16,12))
sns.swarmplot(x='year',y='suicides_no',hue='age',data=suicide_data)
plt.title("Suicide Based On The Year And Age Group")
plt.xticks(rotation=90)
plt.ylabel("Suicide Number")
plt.figure(figsize=(16,12))
sns.catplot(x='year',y='suicides_no',hue='age',data=suicide_data)

plt.figure(figsize=(16,12))
sns.stripplot(x='year',y='suicides_no',hue='age',data=suicide_data,jitter=True)
plt.figure(figsize=(16,12))
sns.lineplot(x='year',y='suicides_no',hue='age',data=suicide_data)
suicide_country_age = data1.groupby(['country','age']).sum()['suicides_no'].reset_index()
suicide_country_age.head()
#Suicide Based on Country And Age
plt.figure(figsize=(18,20))
sns.stripplot(x='suicides_no',y='country',hue='age',data=suicide_country_age,jitter=True)
plt.ylabel("Country")
plt.xlabel("Suicide Number")
plt.title('Suicide Based On The Country and Age')
#sns.lineplot(x='suicides_no',y='country',hue='age',data=suicide_country_age,jitter=True)

df = pd.read_csv('../input/who_suicide_statistics.csv')
x = df[df.year == 2016]
s_bc_2016 = x[['country','suicides_no']].groupby(x.country).suicides_no.sum()
s_bc_2016 = pd.DataFrame(s_bc_2016)
s_bc_2016['c']=s_bc_2016.index
a = s_bc_2016.c.values
# there is probably a clean way to get the 3digit iso codes... but yeah this works, too
b = ['AIA','ARM','AUT','HRV','CYP','CZE','GRD','HUN','ISL','ISR','LTE','MUS',
 'MNG','MSR','NLD', 'PSE', 'QAT', 'MDA', '', 'ROU', 'SWE', 'TJK', 'THA', 'GBR', 'USA' ]
s_bc_2016['code'] = b
#s_bc_2016
# Couldn't figure out how to show all the countries...
# so i found a messy way to include all the countries in my df. Yeah I know
cs = ['AFG','ALB','DZA','ASM','AND','AGO','AIA','ATG','ARG','ARM','ABW','AUS','AUT','AZE','BHM','BHR','BGD','BRB','BLR','BEL','BLZ','BEN','BMU','BTN','BOL','BIH','BWA','BRA','VGB','BRN','BGR','BFA','MMR','BDI','CPV','KHM','CMR','CAN','CYM','CAF','TCD','CHL','CHN','COL','COM','COD','COG','COK','CRI','CIV','HRV','CUB','CUW','CYP','CZE','DNK','DJI','DMA','DOM','ECU','EGY','SLV','GNQ','ERI','EST','ETH','FLK','FRO','FJI','FIN','FRA','PYF','GAB','GMB','GEO','DEU','GHA','GIB','GRC','GRL','GRD','GUM','GTM','GGY','GNB','GIN','GUY','HTI','HND','HKG','HUN','ISL','IND','IDN','IRN','IRQ','IRL','IMN','ISR','ITA','JAM','JPN','JEY','JOR','KAZ','KEN','KIR','PRK','KOR','KSV','KWT','KGZ','LAO','LVA','LBN','LSO','LBR','LBY','LIE','LTU','LUX','MAC','MKD','MDG','MWI','MYS','MDV','MLI','MLT','MHL','MRT','MUS','MEX','FSM','MDA','MCO','MNG','MNE','MAR','MOZ','NAM','NPL','NLD','NCL','NZL','NIC','NGA','NER','NIU','MNP','NOR','OMN','PAK','PLW','PAN','PNG','PRY','PER','PHL','POL','PRT','PRI','QAT','ROU','RUS','RWA','KNA','LCA','MAF','SPM','VCT','WSM','SMR','STP','SAU','SEN','SRB','SYC','SLE','SGP','SXM','SVK','SVN','SLB','SOM','ZAF','SSD','ESP','LKA','SDN','SUR','SWZ','SWE','CHE','SYR','TWN','TJK','TZA','THA','TLS','TGO','TON','TTO','TUN','TUR','TKM','TUV','UGA','UKR','ARE','GBR','USA','URY','UZB','VUT','VEN','VNM','VGB','WBG','YEM','ZMB','ZWE']
csf = []
for c in cs:
    if(s_bc_2016['code'] == c).any():
        pass
    else:
        csf.append(c)
        
z = pd.DataFrame(csf, columns=['code'])
z['suicide_no'] = 0
z['c'] = ''

x = pd.concat((s_bc_2016,z),sort=False)
#x.shape
# Combination of Rachels code from here https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-python
# And stuff from here https://plot.ly/python/choropleth-maps/

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# specify what we want our map to look like
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = x['code'],
        z = x['suicides_no'],
        text = x['suicides_no']
        #locationmode = 'USA-states'
       ) ]

# chart information
layout = dict(
    title = '2016 Global Suicides',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-world-map' )
country = pd.read_csv('../input/who_suicide_statistics.csv')
# drop NaNs
country.dropna(axis=0,inplace =True)
country.isnull().sum()
# finding duplicates
Dub = country.duplicated()
np.unique(Dub)
# Labeling by using LabelEncoder
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
country.sex = le.fit_transform(country.sex) # female:0 , male:1
country.age = le.fit_transform(country.age) # 15-24: 0, 25-34:1, 35-54:2 , 5-14:3, 55-74:4, 75+:5

country.head()


bins = [3, 0, 1, 2, 4, 5]
n = len(bins)
agedata = [country[country['age']==bins[i]]['suicides_no'].values.sum() for i in range(n)]

Age_df = pd.DataFrame(agedata)
Age_df.index = ('5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years')
Age_df.columns = ['Sum of suicides_no']
Age_df
Age_df.max()
n = np.arange(6)
plt.bar(n, Age_df['Sum of suicides_no'] , 0.5)

plt.ylabel('Suicides')
plt.xlabel('Year groups')
plt.xticks(n, Age_df.index)
plt.title('Suicides based on age groups')
plt.yticks([500000,1000000,1500000,2000000,2500000,3000000],
           ['0.5M','1M','1.5M','2M','2.5M','3M'])
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()
bins = [3, 0, 1, 2, 4, 5]
n = len(bins)
Female = country[country['sex']==0]
femaledata = [Female[Female['age']==bins[i]]['suicides_no'].values.sum() for i in range(n)]

Male = country[country['sex']==1]
maledata = [Male[Male['age']==bins[i]]['suicides_no'].values.sum() for i in range(n)]
Female_df = pd.DataFrame(femaledata)
Female_df.index = ('5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years')
Female_df.columns = ['Sum of female suicides_no']

Male_df = pd.DataFrame(maledata)
Male_df.index = Female_df.index
Male_df.columns = ['Sum of male suicides_no']

Sex_concat = pd.concat([Male_df, Female_df], axis = 1)
Sex_concat
fig, ax = plt.subplots()
n = np.arange(6)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(n, Female_df['Sum of female suicides_no'], bar_width,
                 alpha=opacity,
                 color='red',
                 label='Female')
 
rects2 = plt.bar(n + bar_width, Male_df['Sum of male suicides_no'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Male')

ax.set_ylabel('Suicides')
ax.set_title('Suicides based on gender and age')
ax.set_xticks(n)
ax.set_xticklabels(Sex_concat.index)
plt.yticks([500000,1000000,1500000,2000000,2500000],
              ['0.5M','1M','1.5M','2M','2.5M'])
plt.autoscale(enable=True, axis='x', tight=True)
ax.legend()
plt.show()
US = country[country['country']=='United States of America']
years = np.unique(US.year)

a = []
dict = {}
for i in years:
    sum = US[US['year']==i]['suicides_no'].values.sum()
    dict[i] = sum 
a.append(dict)

USyear_df = pd.DataFrame(a)
USyear_df = np.transpose(USyear_df)
USyear_df.columns = ['Suicides'] 

plt.plot(USyear_df)
plt.title('Suicides in United States over years')
plt.xlabel('years')
plt.ylabel('Suicides')
plt.yticks([27000,30000,33000,36000,39000,42000,45000],
           ['27K','30K','33K','36K','39K','42K','45K'])
plt.show()
RUSSIA = country[country['country']=='Russian Federation']
years = np.unique(RUSSIA.year)

a = []
dict = {}
for i in years:
    sum = RUSSIA[RUSSIA['year']==i]['suicides_no'].values.sum()
    dict[i] = sum 
a.append(dict)
RUSyear_df = pd.DataFrame(a)
RUSyear_df = np.transpose(RUSyear_df)
RUSyear_df.columns = ['Suicides'] 

plt.plot(RUSyear_df)
plt.title('Suicides in Russia over years')
plt.xlabel('years')
plt.ylabel('Suicides')
plt.yticks([10000,20000,30000,40000,50000,60000],
           ['10K','20K','30K','40K','50K','60K'])
plt.show()
plt.plot(USyear_df , ls = '-', lw = 2)
plt.plot(RUSyear_df , ls = '--', lw = 2)
plt.title('Suicides over years- RUSSIA VS US')
plt.xlabel('years')
plt.ylabel('Suicides')
plt.yticks([10000,20000,30000,40000,50000,60000],
           ['10K','20K','30K','40K','50K','60K'])
plt.legend(['US','RUSSIA'] , loc ='best')
plt.grid()
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Use only one feature: year
US_data = US.values
year = US_data[:,1]
x = np.unique(year).reshape(-1,1)
y = USyear_df.Suicides.values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3, random_state=42)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Years')
plt.ylabel('No of Suicides')
plt.yticks([25000,30000,35000,40000,45000],['25K','30K','35K','40K','45K'])
plt.show()
# The coefficients
print('Coefficients: \n', reg.coef_)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
