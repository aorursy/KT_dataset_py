# COVID-19 new cases in Vilnius (the capital of Lithuania) prediction
# This is my first notebook shared, the model was done and run a month ago. But really wanted to share and maybe that will be an interest to someone.
# Any type of feedback would be enourmosly valuable and appreciated :) !! 

# APPROACH SUMMARY:
# for 2020-04-06 to 2020-04-12 predicted avg.daily growth rate of 1.049%
# not that bad result 1.008% daily increase predicted in Vilnius from 2020-04-01 to 2020-04-05, actually was 0.94%
# governement measurement time features added
# gdp, density and % of age >65 features added (3)
# same as 2.1. based on daily growth rates on the first 17 days after covid19 break through (i.e. when cases started increasing) .
# based on most similar 4 countries


# DATA SOURCES:
# World Development Indicators & Lithuania Statistics
# Acaps https://data.humdata.org/dataset/acaps-covid19-government-measures-dataset
# COVID19 https://github.com/starschema/COVID-19-data/blob/master/notebooks/JHU_COVID-19.ipynb  and it uses data from 2019 Novel Coronavirus Visual Dashboard operated by Johns Hopkins University


# CHALLENGES:
# Even though actual avg. daily growth rate was 1.17%  (i.e. it was higher than predicted by 12 %) 
# however the total actual number of new cases in Vilnius duing that period was lower (141 real vs 226 predicted - 60% diff)
# The reason for such a big difference was set high number of new cases on the first prediction day (2020-04-06) (21 real vs 28 predicted) which inflated the total numbers (as each day was multiplied by avg.daily growth rate).
# To get the first day of prediction I took an avg of last 6 days. 
# I think better would have been to use 2 models: 1) to get a number for the first day of prediction and only then 2) use cosine similarity to get daily growth rate based on similar countries.
# what what could be that best model to get best prediction for the first day.
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("../input/covid-world-countries-0407/world_countries_0407.csv",sep=";")
df.drop('Unnamed: 0',axis=1,inplace=True)
### pandemic start if avg of 3 days in a row growth rate > 0: ###

# 1st lag 
df['pre_growth_rate'] = df.groupby('Country/Region')['growth_rate'].shift(1,fill_value=0)
df['Date2'] = pd.DatetimeIndex(df['Date']) - pd.DateOffset(1)
df['pre_Date'] = [time.date() for time in df['Date2']]
df.drop(['Date2'],axis=1, inplace=True)

# 2nd lag:
df['pre_pre_growth_rate'] = df.groupby('Country/Region')['growth_rate'].shift(2,fill_value=0)
df['Date4'] = pd.DatetimeIndex(df['Date']) - pd.DateOffset(2)
df['pre_pre_Date'] = [time.date() for time in df['Date4']]
df.drop(['Date4'],axis=1, inplace=True)

df['pandemic_sum'] = df['growth_rate'] + df['pre_growth_rate'] + df['pre_pre_growth_rate']
df['pandemic_mean'] = df[['growth_rate', 'pre_growth_rate', 'pre_pre_growth_rate']].mean(axis=1)

# filter out with avg of 3 days in a row growth rate > 0:
df = df[df['pandemic_mean']>0]

# create new day variable:
df.drop('Day',axis=1,inplace=True)
df = df.sort_values(by = ['Country/Region',"Date"])
df['Day'] = df.groupby('Country/Region').cumcount()+1

df = df[['Country/Region','Day', 'Date','Difference','growth_rate','pandemic_mean']]
#samples:
df[df['Country/Region'] =='Lithuania']
#df[df['Country/Region'] =='Vilnius']
#df[df['Country/Region'] =='Afghanistan']
# model(find similarity) based on the same number of days that Lithuania has reached now:
number_days = df[df['Country/Region'] == 'Lithuania']['Day'].max()-10
df2 = df[df['Day'] <= number_days]
number_days
# only take those that had at least the same number of days with COVID19
df3 = df2[df2.groupby(['Country/Region'])['Day'].transform('count') == number_days]

# number of countries we will compare LT with:
df3['Country/Region'].nunique()
df4 = df3.pivot(index='Country/Region', columns = 'Day', values = 'growth_rate').reset_index()
df4.describe()
# keep country names for final output:
countries = df4[['Country/Region']]

#drop country/branch column:
#df4.drop('Country/Region',axis=1,inplace=True)
countries 
#df4['Country/Region'].unique()
fa = pd.read_csv('../input/population-65-and-older/population_65_and_older.csv', sep=';')

#cleaning data non-matching country names and fixing San Marino null value:
#1.cleaning data non-matching country names
fa.replace({'Vietnam': 'Viet Nam','Slovak Republic': 'Slovakia', 'Czech Republic': 'Czechia' },inplace=True)
fa1 = df4.merge(fa,on='Country/Region')
fa1_countries = fa1['Country/Region'].unique()
# 5 countries that are not in 65 and older dataset:
df4[~df4['Country/Region'].isin(fa1_countries)]
#checking which column has nulls:
fa1.isna().sum()
fa1[fa1.isnull().any(axis=1)]
#2. cleaning: fixing San Marino 19.8
fa1[fa1['Country/Region']=='San Marino'] = fa1[fa1['Country/Region']=='San Marino'].fillna(19.8)
fa1[fa1['Country/Region']=='Andorra'] = fa1[fa1['Country/Region']=='Andorra'].fillna(13.8)
fa1[fa1['Country/Region']=='Liechtenstein'] = fa1[fa1['Country/Region']=='Liechtenstein'].fillna(15)
fa1[fa1['Country/Region']=='Monaco'] = fa1[fa1['Country/Region']=='Monaco'].fillna(26.9)

fd = pd.read_csv('../input/population-density-by-country/population_density_by_country.csv',sep=';')
fd.replace({'Vietnam': 'Viet Nam','Slovak Republic': 'Slovakia', 'Czech Republic': 'Czechia' },inplace=True)
fd1 = fa1.merge(fd,on='Country/Region')
fd1.info()
fg = pd.read_csv('../input/gdp-per-capita-by-country/gdp_per_capita_by_country.csv',sep=';')
fg.replace({'Vietnam': 'Viet Nam','Slovak Republic': 'Slovakia', 'Czech Republic': 'Czechia' },inplace=True)
fg1 =fd1.merge(fg,on='Country/Region')
#fg1.info()
fg1.drop('Country Code',axis=1,inplace=True)
fg1
countries = fg1[['Country/Region']]
#fs2['DATE_IMPLEMENTED'].unique()
fs = pd.read_csv('../input/government-restriction-dates/government_restriction_dates.csv',sep=';')
fs2 = fs[['COUNTRY', 'ISO', 'CATEGORY', 'MEASURE', 'DATE_IMPLEMENTED']]
fs2.dropna(inplace=True)
fs2['DATE_IMPLEMENTED'].max()
#fs2['MEASURE'].value_counts()

#fs2[fs2['CATEGORY'] == 'Lockdown']['MEASURE'].unique()
#fs2.groupby(['COUNTRY'])[]
#fs2[fs2['CATEGORY','MEASURE']].nunique()

#fs2.groupby(['COUNTRY','CATEGORY','MEASURE']).agg({'DATE_IMPLEMENTED':'min'}).reset_index()
fs3 = fs2.groupby(['COUNTRY','CATEGORY']).agg({'DATE_IMPLEMENTED':'min'}).reset_index()
fs4 = fs3.pivot(index = 'COUNTRY', columns = 'CATEGORY', values = 'DATE_IMPLEMENTED').reset_index()
fs4.columns = ['Country/Region', 'Humanitarian exemption', 'Lockdown',
       'Movement restrictions', 'Public health measures',
       'Social and economic measures', 'Social distancing']

vno = fs4[fs4['Country/Region'] == 'Lithuania'].copy()
vno.replace('Lithuania', 'Vilnius',inplace = True)

fs5 = fs4.append(vno)

#fg1 =fd1.merge(fg,on='Country/Region')
#fg1
with_outbreak_date = fg1.merge(df[df['Day']==1][['Country/Region', 'Date']], on = 'Country/Region')
fs55 = with_outbreak_date.merge(fs5, on = 'Country/Region')
fs6 = fs55.merge(df.groupby(['Country/Region']).agg({'Day':'max'}), on ='Country/Region')
#fs6.info()
fs6.drop(columns = ['Humanitarian exemption'],inplace=True)
fs6

#fs6['Lockdown'] = pd.to_datetime(df["Lockdown"]).dt.strftime('%Y-%m-%d')

#fs6["Date"] = pd.to_datetime(fs6["Date"]).dt.strftime('%Y-%m-%d')
#fs6["Lockdown"] = pd.to_datetime(fs6["Lockdown"]).dt.strftime('%Y-%m-%d')
#fs6["Movement restrictions"] = pd.to_datetime(fs6["Movement restrictions"]).dt.strftime('%Y-%m-%d')
#fs6["Social and economic measures"] = pd.to_datetime(fs6["Social and economic measures"]).dt.strftime('%Y-%m-%d')
#fs6["Social distancing"] = pd.to_datetime(fs6["Social distancing"]).dt.strftime('%Y-%m-%d')

fs6[['Lockdown', 'Movement restrictions', 'Public health measures', 'Social and economic measures', 
     'Social distancing', 'Date']] = fs6[['Lockdown', 'Movement restrictions', 'Public health measures', 
                                          'Social and economic measures', 'Social distancing', 'Date']].apply(pd.to_datetime)

fs6['Lockdown_diff'] = (fs6['Date']-fs6['Lockdown']).dt.days
fs6['Public health measures_diff'] = (fs6['Date']-fs6['Public health measures']).dt.days
fs6['Movement restrictions_diff'] = (fs6['Date']-fs6['Movement restrictions']).dt.days
fs6['Social and economic measures_diff'] = (fs6['Date']-fs6['Social and economic measures']).dt.days
fs6['Social distancing_diff'] = (fs6['Date']-fs6['Social distancing']).dt.days

fs6.drop(['Movement restrictions','Date', 'Lockdown','Public health measures','Social and economic measures',
        'Social distancing' ],axis=1, inplace=True)
fs6.fillna(-75,inplace=True)
fs6

countries = fs6[['Country/Region']]
fs6.drop(columns = ['Country/Region'],inplace=True)
#df[['A','B']] = df[['A','B']].apply(pd.to_datetime) #if conversion required
#df['C'] = (df['B'] - df['A']).dt.days

#fs6.columns

fs6

# then add 1st date to fg1 for final sactions days from outbreak calculation.
#fg1
fg1.drop(['Country/Region'],axis=1, inplace=True)
output = pd.DataFrame(cosine_similarity(fs6)) #fg1

#add country/branch:
final_output = countries.merge(output, left_index = True, right_index=True)
#final_output.to_csv('cosine_similarity_growth_rate_by_day_3dMeanPandemic_Vilnius_added.csv')
final_output
#select Lithuania and Vilnius only:
selected = final_output[final_output['Country/Region'].isin(['Lithuania', 'Vilnius'])].T
selected.columns = selected.iloc[0]
selected.drop(selected.index[0],inplace=True)
selected = countries.merge(selected, left_index=True, right_index=True)
selected = selected.sort_values(by=['Lithuania'], ascending=False)
selected['order_by_similarity_lithuania'] = np.arange(len(selected))
selected = selected.sort_values(by=['Vilnius'], ascending=False)
selected['order_by_similarity_vilnius'] = np.arange(len(selected))
selected
#p = pd.read_csv('similarity_score_pandemic_3dmean_done_on_up_to_0327_for_LT.csv',sep=';')
p = selected.copy()
p
# CHOOSE Vilnius or Lithuania:
order_choice = 'order_by_similarity_vilnius' #order_by_similarity_vilnius, order_by_similarity_lithuania
region_choice = 'Vilnius' #vilnius, Lithuania
number_of_top_similar = 5
p = p[p[order_choice] !=0]

top = p[(p[order_choice] <= number_of_top_similar) & (p['Country/Region'] != 'Lithuania') ]
top
# visualize future in similar countries VERSION NO 1 (similar countries trend in each) (Singapore was also similar but taken out due to being a exception to the rule):
similar_countries = top['Country/Region'].unique()
for_trends = df[df['Country/Region'].isin(similar_countries)]
for_trends[['Country/Region','Day','growth_rate']]
# visualize:
sns.set(rc={'figure.figsize':(14,10)})
sns.lineplot(x="Day", y="growth_rate", hue="Country/Region", data=for_trends)
# visualize future in similar countries VERSION NO 2 (mean of similar countries) (Singapore was also similar but taken out due to being a exception to the rule):
similar_countries_without_lt = similar_countries

avg = df[df['Country/Region'].isin(similar_countries_without_lt)].groupby('Day').agg({'growth_rate': 'mean'}).reset_index()
avg['Country/Region'] = 'Mean of Similar Countries'
lt = df[df['Country/Region'].isin([region_choice])][['Day', 'growth_rate','Country/Region']]
for_trends_mean = pd.concat([avg, lt])

# visualize:
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x="Day", y="growth_rate", hue="Country/Region", data=for_trends_mean)

# here similarity is done based on first 17 days of pandemic:
#df = pd.read_csv('pasaulio_salys_0404.csv')
df[df['Country/Region'] == region_choice]
# prediction validation:
#include new data that we have on LT only:
# but only after cleaning data in the same way (only once the pandemic start)

# DATA CLEANING:

### pandemic start if avg of 3 days in a row growth rate > 0: ###

df = pd.read_csv('/../input/covid-world-countries-0407/world_countries_0407.csv',sep=";")

# 1st lag 
df['pre_growth_rate'] = df.groupby('Country/Region')['growth_rate'].shift(1,fill_value=0)
df['Date2'] = pd.DatetimeIndex(df['Date']) - pd.DateOffset(1)
df['pre_Date'] = [time.date() for time in df['Date2']]
df.drop(['Date2'],axis=1, inplace=True)

# 2nd lag:
df['pre_pre_growth_rate'] = df.groupby('Country/Region')['growth_rate'].shift(2,fill_value=0)
df['Date4'] = pd.DatetimeIndex(df['Date']) - pd.DateOffset(2)
df['pre_pre_Date'] = [time.date() for time in df['Date4']]
df.drop(['Date4'],axis=1, inplace=True)

df['pandemic_sum'] = df['growth_rate'] + df['pre_growth_rate'] + df['pre_pre_growth_rate']
df['pandemic_mean'] = df[['growth_rate', 'pre_growth_rate', 'pre_pre_growth_rate']].mean(axis=1)

# filter out with avg of 3 days in a row growth rate > 0:
df = df[df['pandemic_mean']>0]

# create new day variable:
df.drop('Day',axis=1,inplace=True)
df = df.sort_values(by = ['Country/Region',"Date"])
df['Day'] = df.groupby('Country/Region').cumcount()+1

df = df[['Country/Region','Day', 'Date','Difference','growth_rate','pandemic_mean']]
#samples:
#df[df['Country/Region'] =='Lithuania']
#df[df['Country/Region'] =='Afghanistan']
#LT actual growth rate with later dates:
lt_for_validation = df[df['Country/Region']==region_choice][['Day', 'growth_rate','Country/Region']]
#mean of similar countries:
mean_for_validation = for_trends_mean[for_trends_mean['Country/Region'] == 'Mean of Similar Countries']
for_validation = pd.concat([lt_for_validation,mean_for_validation])
# visualize:
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x="Day", y="growth_rate", hue="Country/Region", data=for_validation)
# what was the average growth rate of similar countries in the future 7 days (18-24) and in Vilnius?
for_validation[(for_validation['Country/Region']==region_choice) & (for_validation['Day'] > 17) & (for_validation['Day'] < 23)].agg({'growth_rate':'mean'})
for_validation[(for_validation['Country/Region']=='Mean of Similar Countries') & (for_validation['Day'] > 17) & (for_validation['Day'] < 23)].agg({'growth_rate':'mean'})
for_validation[(for_validation['Country/Region']=='Mean of Similar Countries') & (for_validation['Day'] > 22) & (for_validation['Day'] < 30)].agg({'growth_rate':'mean'})
