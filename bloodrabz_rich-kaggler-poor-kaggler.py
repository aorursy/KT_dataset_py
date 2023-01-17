import math
import numpy as np 
import pandas as pd 
import seaborn as sn
import matplotlib.pyplot as plt
import os
from pandas.api.types import CategoricalDtype
import cufflinks as cf
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Import all data sources into pandas DataFrame
'''

kag18 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv',low_memory=False)[1:]
sof18 = pd.read_csv('../input/stack-overflow-2018-developer-survey/survey_results_public.csv',low_memory=False)
kag17=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1',low_memory=False)
sof17 = pd.read_csv('../input/so-survey-2017/survey_results_public.csv',low_memory=False)

kag18 = kag18[kag18.Q9 !='I do not wish to disclose my approximate yearly compensation' ] # Exclude all people that did not disclose their income.

# all valid salary ranges for the purpose of ordering them for the plot that follows.
salary_bracket= ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000',
                       '70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000',
                       '200-250,000','250-300,000','300-400,000','400-500,000','500,000+']
#Plot the data using seaborns Countplot that counts all the occurrences of a particular value, the income range in this case
fig, ax = plt.subplots(figsize=(18,6))
g = sn.countplot(x='Q9',data=kag18, order=salary_bracket, ax=ax,palette = 'Spectral')
g.set_xticklabels(ax.get_xticklabels(),rotation=90)
g.set_title('Yearly salary distribution')
g.set_xlabel('Salary USD')
g.set_ylabel('Total Repsondents')

# Add the percentage values above each bar
ncount = kag18.shape[0]
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),ha='center', va='bottom')

'''
Calculate the cumulative income and draw a barplot
'''

salGroupby = kag18.groupby('Q9').count()['Q1'] # Group the dataframe by Q9 ( the question regarding income)
#Reindex the dataframe and map the income to the below list
salGroupby.sort_index
salary_bracket_simple= ['5000','15000','115000','135000','175000','25000','225000','275000','35000','350000','45000','450000',
                       '55000','500000','65000','75000','85000','95000']
salary_bracket_simple = list(map(int, salary_bracket_simple))
salGroupby.index = salary_bracket_simple
salary_bracket_simple = sorted(salary_bracket_simple)
salGroupby = salGroupby.sort_index()


# add new columns to the dataframe that will contain the cumulative income.
salGroupbyDF = salGroupby.to_frame()
salGroupbyDF.columns = ['amount']
salGroupbyDF = salGroupbyDF.reset_index()
salGroupbyDF.columns = ['salary','amount']
salGroupbyDF['total'] = salGroupbyDF['amount']

#Calculate the total amount of wealth each interval holds
for indx , row in salGroupbyDF.iterrows():
    salGroupbyDF['total'][indx] =salGroupbyDF['amount'][indx]*salGroupbyDF['salary'][indx] 

ncount1 = salGroupbyDF['total'].sum()   # Kagglers total wealth
salGroupbyDF['percentage'] = salGroupbyDF['amount'].astype(float)

#Calculate the percentage of income that each Income interval holds
for indx , row in salGroupbyDF.iterrows():
    salGroupbyDF['percentage'][indx] =float(salGroupbyDF['total'][indx])*100/float(ncount1)

#Use Seaborn's barplot to plot the cumulative income
fig1, ax1 = plt.subplots(figsize=(18,6))
g1 = sn.barplot(x=salGroupbyDF['salary'],y = salGroupbyDF['total'] ,data=salGroupbyDF, order=salary_bracket_simple,palette = 'Spectral' ,ax=ax1)
g1.set_xticklabels(salary_bracket,rotation=90)
g1.set_title('Aggregate Wealth Distribution')
g1.set_xlabel('Avg Salary')
g1.set_ylabel('Aggregate Salary USD')

#Add Percantage values above each bar
for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{:.1f}%'.format(100*y/ncount1), (x.mean(), y),ha='center', va='bottom')

'''
Get Some basic Descriptive Statistics regarding Income ex. Mean , Mode , 25th and 75th percentile
'''

#Get the mean income
salGroupbyDF['aggregate_amount'] = salGroupbyDF['amount']
for indx , row in salGroupbyDF.iterrows():
    if indx == 0:
        salGroupbyDF['aggregate_amount'][indx] = salGroupbyDF['amount'][indx]
    else:
        salGroupbyDF['aggregate_amount'][indx] = salGroupbyDF['aggregate_amount'][indx-1] + salGroupbyDF['amount'][indx]
mean = ncount1/ncount
print(mean)
median_index = (ncount//2)
median = 45000
mode = 5000

perc25_cutoff_sal = (0.25*ncount)
perc75_cutoff_sal =(0.75 * ncount) 

#Get the 25th percentile
avg25_salary = 0
for indx , row in salGroupbyDF.iterrows():
    if salGroupbyDF['aggregate_amount'][indx] < perc25_cutoff_sal and salGroupbyDF['aggregate_amount'][indx+1] > perc25_cutoff_sal :
        diff =perc25_cutoff_sal - salGroupbyDF['aggregate_amount'][indx]
        for i in range(indx+1):
            avg25_salary =avg25_salary+ salGroupbyDF['total'][i]
        avg25_salary =avg25_salary+ diff*salGroupbyDF['salary'][indx+1]
        avg25_salary = avg25_salary/(salGroupbyDF['amount'][indx]+diff)
print(avg25_salary, '     25%',perc25_cutoff_sal)

#Get the 75th percentile
for indx , row in salGroupbyDF.iterrows():
    if salGroupbyDF['aggregate_amount'][indx] < perc75_cutoff_sal and salGroupbyDF['aggregate_amount'][indx+1] > perc75_cutoff_sal :
        diff = salGroupbyDF['aggregate_amount'][indx+1]-perc75_cutoff_sal 
        avg75_salary = diff*salGroupbyDF['salary'][indx]
        peeps = diff
        for i in range(indx+1,len(salGroupbyDF)):
            avg75_salary =avg75_salary + salGroupbyDF['total'][i]
            peeps = peeps + salGroupbyDF['amount'][i]
        avg75_salary = avg75_salary/(peeps)
print(avg75_salary, '    75%',peeps)
salGroupbyDF.head(20)

'''
Import 2017 Kaggle Survey and preprocess it to analyse the differences regarding Income
'''
#SOURCE https://www.kaggle.com/drgilermo/salary-analysis

#Read the 2017 Data set as well as the exchange rate set to convert salaries to USD
exchange = pd.read_csv('../input/kaggle-survey-2017/conversionRates.csv', encoding="ISO-8859-1", low_memory=False)
kag17=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1', low_memory=False)
kag17 = pd.merge(left=kag17, right=exchange, how='left', left_on='CompensationCurrency', right_on='originCountry')

#Clean the Income information to filter out NaNs and other unexpected characters
kag17['exchangeRate'] = kag17['exchangeRate'].fillna(0)
kag17['CompensationAmount'] = kag17['CompensationAmount'].fillna(0)
kag17['CompensationAmount'] =kag17.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0)) else float(x.replace(',','')))
kag17['CompensationAmount'] = kag17['CompensationAmount']*kag17['exchangeRate']
kag17 = kag17[kag17['CompensationAmount']>0] 

#Calculate basic descriptive statistics
kag17_len = kag17.shape[0]
numAbove100 = 100*kag17[kag17['CompensationAmount']>100000].shape[0]/kag17_len
numBelow30 = 100*kag17[kag17['CompensationAmount']<30000].shape[0]/kag17_len
kag17_salary = kag17['CompensationAmount'].sort_values()
low_25= kag17_salary[50:kag17_len//4].mean()
high_25= kag17_salary[3*(kag17_len//4):-10].mean()

temp18sal = pd.DataFrame(columns = ['salary','year'])
temp18sal['salary'] = kag18['Q9']
temp18sal['year'] = '2018'
temp18sal=temp18sal.dropna()

#Convert the salary from String format to a float
def removeHyphen(x):
    y = x
    if  not type(y)==float: 
        if '-' in y:
            y =float(str(x).split('-')[0])*1000+1
    if type(y) is str:
        y = re.sub('[^0-9]','', y)
        y = float(y)
    return y

#Create a new DataFrame that contains both the 2018 and 2017 salaries labelled with the year.
temp18sal['salary'] = temp18sal['salary'].apply(removeHyphen)
count18 = temp18sal['salary'].apply(lambda x : float(round(int(x),-4))).value_counts().sort_index()/(temp18sal.shape[0])
Count18 = count18.to_frame()
Count18['year'] = '2018'
data_for_graph = pd.DataFrame(columns = ['salary','year'])
data_for_graph['salary'] = kag17['CompensationAmount'].sort_values()[3:-20]
data_for_graph['year'] = '2017'
count17 = data_for_graph['salary'].apply(lambda x : float(round(int(x),-4))).value_counts().sort_index()/(data_for_graph['salary'].shape[0]-20)
Count17 = count17.to_frame()
Count17['year'] = '2017'
data_for_graph = pd.concat([data_for_graph,temp18sal],ignore_index=True)
data_for_graph=data_for_graph.fillna(0)
data_for_graph['salary'] = data_for_graph['salary'].apply(lambda x : float(round(int(x),-4)))

#Concatinate the Dataframes to create final one including both 2017 and 2018
GraphData = pd.concat([Count17,Count18])
GraphData.reset_index(drop = False,inplace=True)
GraphData.columns = ['bracket','salary','year']

#order of the barplot xticks
order = [     0.0,  10000.0,  20000.0,  30000.0,  40000.0,  50000.0,
               60000.0,  70000.0,  80000.0,  90000.0, 100000.0, 150000.0, 200000.0, 300000.0, 400000.0, 500000.0]

#Draw the barplot with the hue set to the year.
fig, ax = plt.subplots(figsize=(18,6))
g = sn.barplot(x='bracket',y = 'salary',data=GraphData,order = order, ax=ax,hue = 'year',palette = 'hls')
g.set_xticklabels(salary_bracket,rotation=90)
g.set_title('2018 vs 2017 salary distribution*')
g.set_xlabel('Salary USD')
g.set_xlim()
g.set_ylabel('Total Repsondent %')
g.set(xlim=(-0.5, 17))
g.set(ylim=(0, 0.3))
'''
Isolate the Country and Income columns of my dataframe , group it by Countries and sort it in a descending manner wrt Income
'''
kag18 = kag18[pd.notnull(kag18['Q9']) ]
kag18['Q9.1'] = kag18['Q9'].apply(removeHyphen)# remove characters from Income entries
#Replace some of the country names. The purpose will be clear in a little bit
kag18['Q3'].replace({'United States of America':'United States','Viet Nam':'Vietnam','China':"People 's Republic of China","United Kingdom of Great Britain and Northern Ireland":'United Kingdom',"Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)
country_gb18 = kag18.groupby('Q3')['Q9.1'].mean().sort_values(ascending = False)[:30]#select 30 higest values
country_gb17 = kag17.groupby('Country')['CompensationAmount'].median().sort_values(ascending = False)
country_gb18 = country_gb18.to_frame().reset_index()
#Plot the graph
fig, ax = plt.subplots()
fig.set_size_inches(20, 12)
g = sn.barplot(x = 'Q3',y ='Q9.1',palette = 'Spectral' ,data = country_gb18,orient='v')
g.set_xticklabels(country_gb18.Q3,rotation=90)
g.set_title( "2018 Country vs Income")
g.set_xlabel('Country')
g.set_ylabel('Median Salary USD')
#Add values above each bar in barplot
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
'''
Create new DataFrame that I will use throughout the whole kernel. Only select columns that I will use , rename columns appropriatly and categorize the categorial values
'''

Answers = kag18.iloc[1:,:]#a dataframe of the answers

#Fill my dataframe with all the columns that I am going to use.
my_df = Answers.iloc[:,:13].copy()
my_df['language'] =Answers['Q17']
my_df['years_coding_to_analyze'] = Answers['Q24']
my_df['you_a_data_scientist'] = Answers['Q26']
cols = ['survey_duration', 'gender', 'gender_text', 'age', 'country', 'education_level', 'undergrad_major', 'role', 'role_text',
        'employer_industry', 'employer_industry_text', 'years_experience', 'salary','language','years_coding_to_analyze','you_a_data_scientist']
my_df.columns = cols
my_df.drop(['survey_duration', 'gender_text', 'role_text', 'employer_industry_text'], axis=1, inplace=True)# Drop these columns as I am not going to use them

# Map all of the salary intervals into the below categories
categ = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
         '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
         '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',
         '300-400,000', '400-500,000', '500,000+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
my_df.salary = my_df.salary.astype(cat_type)

# Map all of the age intervals into the below categories
categ = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', 
         '45-49', '50-54', '55-59', '60-69', '70-79', '80+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
my_df.age = my_df.age.astype(cat_type)

# Map all of the experience intervals into the below categories
categ = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10',
         '10-15', '15-20', '20-25', '25-30', '30+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
my_df.years_experience = my_df.years_experience.astype(cat_type)

#Function that strips a string of all non-numeric characters.
def stripYears(x):
    y = str(x)
    if y == '< 1 year':
        y = '0-1'
    if 'years' in y : 
        y = y.strip('years')
    elif 'year' in y :
        y = y.strip('year')
    if '<' in y :
        y.strip('<')
    return y 

#Map all the education levels into categories.
my_df.years_coding_to_analyze = my_df.years_coding_to_analyze.apply(stripYears)
categ = ['No formal education past high school', 'Some college/university study without earning a bachelor’s degree',
         'Professional degree', 'Bachelor’s degree', 'Master’s degree', 'Doctoral degree', 'I prefer not to answer']
cat_type = CategoricalDtype(categories=categ, ordered=True)
my_df.education_level = my_df.education_level.astype(cat_type)

my_df = my_df[~my_df.salary.isnull()].copy()# exclude all null values
compensation = my_df.salary.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')

'''
Since salary information was given in ranges ,I added columns of the mean , min and max of each salary interval.
'''
my_df['salary_value'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2)  
my_df['salary_max'] = compensation.apply(lambda x: (int(x[1])))  
my_df['salary_min'] = compensation.apply(lambda x: (int(x[0]) * 1000)) 

'''
Get the countries with the most respondents , since they will provide a better estimate.Use them to see how wealth is distributed amongst them.
'''
#  count the number of respondents per country.
mostRespondentCountries = my_df.groupby('country').count().sort_values(by='gender')[-15:].reset_index().country.values
mostRespondentCountriesCount = my_df.groupby('country').count().sort_values(by='gender').gender[-15:].reset_index()
barplot = my_df.groupby('country')#['salary']#.value_counts(sort=False)#.to_frame().reset_index()

#Build the dataframe used to plot the barplot of countries vs income.
barplot = barplot['salary'].value_counts(sort=False).to_frame()
barplot.columns = ['count']
barplot =barplot.reset_index(1)
barplot =barplot.reset_index(0)
barplot.columns = ['country','salary','count']
barplot.head(20)
barplot['count_perc'] =  barplot['count'].astype(float)

#Only include countries that is in the 15'mostResponded' list
for indx, row in barplot.iterrows():
    if (barplot['country'][indx] not in mostRespondentCountriesCount['country'].values) :
        barplot = barplot.drop(indx,axis = 0)
    else :
        barplot['count_perc'][indx] =(float(barplot['count'][indx])/float(mostRespondentCountriesCount['gender'][mostRespondentCountriesCount['country'] == barplot['country'][indx]]))*100
barplot = barplot[barplot.country != 'Other']# Exclude 'other' countries

categ = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
         '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
         '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',
         '300-400,000', '400-500,000', '500,000+']# The order of xticks

hue_order = ['United States', 'Australia'  , 'Japan' ,'France', 'Canada', 'United Kingdom' ,'Germany' ,'Spain','Poland','Italy','Brazil' ,'Russia', "People 's Republic of China", 'India' ] # order of hue

#Plot the barplot
fig, ax = plt.subplots()
fig.set_size_inches(20, 12)
g = sn.barplot(x = 'salary',y ='count_perc',hue = 'country',hue_order =hue_order,palette = 'Spectral' ,data = barplot,orient='v')
g.set_xticklabels(categ,rotation=90)
g.set_title('Percantage of Country in Income brackets')
g.set_xlabel('Salary USD')
plt.legend(loc='upper right')
g.set_ylabel('Repsondent %')

fig, ax = plt.subplots()
fig.set_size_inches(30, 12)
g = sn.swarmplot(x = 'country',y ='count_perc',hue = 'salary',hue_order =categ,palette = 'RdBu' ,size=15,data = barplot)
g.set_xticklabels(barplot.country.sort_values().unique(),rotation=90)
g.set_title('Percantage of Country in Income brackets')
g.set_xlabel('Salary')
plt.legend(loc='best')
g.set_ylabel('Repsondent %')
'''
Use Plotly to draw a Wolrd Map showing the income of each country
'''
#Create a new dataframe for the world map
contry_salaries = pd.DataFrame({'name':my_df['country'],'salary':my_df['salary_value'],'salary_min':my_df['salary_min'],'salary_max':my_df['salary_max']})
contry_salaries = contry_salaries.groupby('name').mean().reset_index()

# Specify the data used in the plotly figure
data = [dict(
        type = 'choropleth',# World map
        locations = contry_salaries.name,
        locationmode = 'country names',
        z = contry_salaries.salary,
        colorscale = [[0,"#081D58"],[0.35,"#253494"],
                      [0.5,"#225EA8"],
            [0.6,"#1D91C0"],[0.7,"#41B6C4"],
                      [1,"#7FCDBB"]],
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(0,0,0)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = ''),
      ) ]

layout = dict(
    title = 'Average Income Per Country',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig1 = dict(data=data, layout=layout)
py.iplot(fig1, validate=False)
'''
Read in the Grocery Index and level out salaries by dividing peoples' salaries with their Country's corresponding index value
'''

price_level = pd.read_csv('../input/living-cost-by-country/Groceries index.csv')
del price_level['Date']# we dont need the date
price_level['Country'].replace({"Iran":"Iran, Islamic Republic of...",'China':'People \'s Republic of China'},inplace=True)# Replace Country names so that the match Kaggles survey names

#Make a new dataframe with the leveled out salaries.
my_df_level = my_df[['salary_value','gender','age','country','education_level','undergrad_major','role','employer_industry','years_experience','language','years_coding_to_analyze','you_a_data_scientist']][(my_df['country']!='Republic of Korea')&(my_df['country']!='Other') &(my_df['country']!='I do not wish to disclose my location') ].reset_index(drop=True)
my_df_level['salary_value_level'] =my_df_level['salary_value'] 

#Find Indeces with corresponding  Countries and divide
def getPriceLevel(x):
    for indx , row in x.iterrows():
        my_df_level['salary_value_level'][indx] = x['salary_value'][indx]/price_level[price_level['Country'] == x['country'][indx]]['Amount']
getPriceLevel(my_df_level)

# select 30 countries with highest mean income
my_df_level_plot = my_df_level.groupby('country')['salary_value_level'].mean().sort_values(ascending = False)[:30].to_frame().reset_index()
my_df_level_plot1 = my_df_level.groupby('country')['salary_value'].mean().sort_values(ascending = False)[:30].to_frame().reset_index()

#Plot them using seaborn
fig, (ax1,ax2) = plt.subplots(2,1)
fig.set_size_inches(20, 15)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=0.6)

g = sn.barplot(x = 'country',y ='salary_value_level',palette = 'Spectral' ,data = my_df_level_plot,orient='v',ax = ax1)
g.set_xticklabels(my_df_level_plot.country,rotation=90)
g.set_title( "Level :Country vs Income")
g.set_xlabel('Country')
g.set_ylabel('Mean Level Salary ')

g1 = sn.barplot(x = 'country',y ='salary_value',palette = 'Spectral' ,data = my_df_level_plot1,orient='v',ax = ax2)
g1.set_xticklabels(my_df_level_plot1.country,rotation=90)
g1.set_title( "USD: Country vs Income")
g1.set_xlabel('Country')
g1.set_ylabel('Mean Salary USD')

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
    
for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
'''
Use Plotly to draw a Wolrd Map showing the level income of each country
'''
#Create a new dataframe for the world map
contry_salaries = pd.DataFrame({'name':my_df['country'],'salary':my_df_level['salary_value_level']})
contry_salaries = contry_salaries.groupby('name').mean().reset_index()

# Specify the data used in the plotly figure
data = [dict(
        type = 'choropleth',# World map
        locations = contry_salaries.name,
        locationmode = 'country names',
        z = contry_salaries.salary,
        colorscale = [[0,"#081D58"],[0.35,"#253494"],
                      [0.5,"#225EA8"],
            [0.6,"#1D91C0"],[0.7,"#41B6C4"],
                      [1,"#7FCDBB"]],
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(0,0,0)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = ''),
      ) ]

layout = dict(
    title = 'Level Avg Income Per Country',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig1 = dict(data=data, layout=layout)
py.iplot(fig1, validate=False)
agePlot_no_student = my_df[my_df['employer_industry'] != 'I am a student']
agePlot_no_student = agePlot_no_student.groupby('age').median().reset_index()[:11]
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'age',y ='salary_value',palette = 'Spectral',data = agePlot_no_student)
g.set_xticklabels(agePlot_no_student.age[:11],rotation=90)
g.set_title('Salary by Age ')
g.set_xlabel('Age')
g.set_ylabel('Salary USD')
g.set(xlim=(-0.5, 10.5))

for p in ax.patches[:-1]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
genderPlot = my_df[(my_df['employer_industry'] != 'I am a student') & (my_df['gender'] != 'Prefer not to say') & (my_df['gender'] != 'Prefer to self-describe')].reset_index()[['gender','age','salary_value']]
genderPlot = genderPlot.groupby(('age','gender'),as_index=True)['salary_value'].mean().reset_index()[:22]

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'age',y ='salary_value',hue ='gender',palette = 'hls',data = genderPlot)
g.set_xticklabels(genderPlot.age[::2],rotation=90)
g.set_title('Salary by Gender and Age')
g.set_xlabel('Age')
g.set_ylabel('Average Salary USD')
g.set(xlim=(0, 10.5))
ax.axis('tight')

for p in ax.patches[:]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    if math.isnan(y):
        y = 0
    else:
        ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
xpPlot = my_df[(my_df['employer_industry'] != 'I am a student') & (my_df['years_experience'] != math.nan) ].reset_index()[['years_experience','age','salary_value']]
xpPlot = xpPlot.groupby('years_experience')['salary_value'].mean().reset_index().sort_values(by='salary_value').dropna()

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'years_experience',y ='salary_value',palette = 'Spectral',data = xpPlot)
g.set_xticklabels(xpPlot.years_experience,rotation=90)
g.set_title('Salary by experience')
g.set_xlabel('Years Experience')
g.set_ylabel('Salary USD')
g.set(xlim=(-0.5, 9.5))

for p in ax.patches[:-1]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
xpCodePlot = my_df[(my_df['employer_industry'] != 'I am a student') & (my_df['years_coding_to_analyze'] != math.nan) ][['years_coding_to_analyze','salary_value']]
xpCodePlot = xpCodePlot.groupby('years_coding_to_analyze')['salary_value'].mean()[:10].reset_index().sort_values(by='salary_value')#.sort_index()[:9]
xpCodePlot.head(20)

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'years_coding_to_analyze',y ='salary_value',order =xpCodePlot.years_coding_to_analyze ,palette = 'Spectral',data = xpCodePlot)
g.set_xticklabels(xpCodePlot.years_coding_to_analyze,rotation=90)
g.set_title('Salary by experience in coding to analyse')
g.set_xlabel('Years Experience in Analysis Coding')
g.set_ylabel('Salary USD')
g.set(xlim=(-0.5, 9.5))

for p in ax.patches[:]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
personal_data = my_df[(my_df['employer_industry'] != 'I am a student')][['undergrad_major','role','employer_industry','education_level','salary_value']]
major = personal_data.groupby('undergrad_major')['salary_value'].mean().reset_index().sort_values(by='salary_value').dropna()

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'undergrad_major',y ='salary_value',order =major.undergrad_major ,palette = 'Spectral',data = major)
g.set_xticklabels(major.undergrad_major,rotation=90)
g.set_title('Salary by major')
g.set_xlabel('Major')
g.set_ylabel('Salary USD')
g.set(xlim=(-0.5, 9.5))
ax.axis('tight')


for p in ax.patches[:]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    if math.isnan(y):
        y = 0
    else:
        ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
majorCount = my_df[['undergrad_major','salary_value']]
majorCount = majorCount.groupby(('undergrad_major'),as_index=True).count().reset_index()
majorCount.columns = ['major','count']
majorCount.sort_values(by='count').reset_index(drop=True).head(13)


countryCount = my_df[['country','undergrad_major','salary_value']]
countryCount = countryCount.groupby(('undergrad_major','country'),as_index=False).count()#.reset_index()
countryCount.columns = ['undergrad_major','country','count']
countryCount = countryCount.sort_values(by='count',ascending = False).reset_index(drop=True)
countryCount.head(10)

compSciPlot = my_df[(my_df['undergrad_major']=='Computer science (software engineering, etc.)')|(my_df['undergrad_major']=='Information technology, networking, or system administration')][['undergrad_major','salary']].groupby('salary').count().reset_index()#['salaryy'].mean().sort_values(ascending = False)[:30].to_frame().reset_index()
fineArtsPlot =my_df[(my_df['undergrad_major']=='Fine arts or performing arts')|(my_df['undergrad_major']=='Humanities (history, literature, philosophy, etc.)')][['undergrad_major','salary']].groupby('salary').count().reset_index()

totalCompSci = float(compSciPlot.sum()[1])
totalFineArts = float(fineArtsPlot.sum()[1])

fig, (ax1,ax2) = plt.subplots(2,1)
fig.set_size_inches(20, 15)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=0.6)
g = sn.barplot(x = 'salary',y ='undergrad_major',palette = 'Spectral' ,data = compSciPlot,orient='v',ax = ax1)
g.set_xticklabels(compSciPlot.salary,rotation=90)
g.set_title( "Computer Science and IT Undergrad Salary Distribution")
g.set_xlabel('Salary')
g.set_ylabel('Frequency')

g1 = sn.barplot(x = 'salary',y ='undergrad_major',palette = 'Spectral' ,data = fineArtsPlot,orient='v',ax = ax2)
g1.set_xticklabels(fineArtsPlot.salary,rotation=90)
g1.set_title( "Fine Arts and Humanities Undergrad Salary Distribution")
g1.set_xlabel('Salary')
g1.set_ylabel('Frequency')

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{:.1f}%'.format(100*y/totalCompSci), (x.mean(), y),ha='center', va='bottom')
    
for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{:.1f}%'.format(100*y/totalFineArts), (x.mean(), y),ha='center', va='bottom')
#personal_data = my_df[(my_df['employer_industry'] != 'I am a student')][['undergrad_major','role','employer_industry','education_level','salary_value']]
level = personal_data.groupby('education_level')['salary_value'].mean().reset_index().sort_values(by='salary_value').dropna()

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'education_level',y ='salary_value',order =level.education_level ,palette = 'Spectral',data = level)
g.set_xticklabels(level.education_level,rotation=90)
g.set_title('Salary by education level (no students)')
g.set_xlabel('Education Level')
g.set_ylabel('Salary USD')
g.set(xlim=(-0.5, 9.5))
ax.axis('tight')


for p in ax.patches[:]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    if math.isnan(y):
        y = 0
    else:
        ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
#personal_data = my_df[(my_df['employer_industry'] != 'I am a student')][['undergrad_major','role','employer_industry','education_level','salary_value']]
role = personal_data.groupby('role')['salary_value'].mean().reset_index().sort_values(by='salary_value').dropna()

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
g = sn.barplot(x = 'role',y ='salary_value',order =role.role ,palette = 'Spectral',data = role)
g.set_xticklabels(role.role,rotation=90)
g.set_title('Salary by Role')
g.set_xlabel('Role')
g.set_ylabel('Salary USD')
g.set(xlim=(-0.5, 9.5))
ax.axis('tight')


for p in ax.patches[:]:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    if math.isnan(y):
        y = 0
    else:
        ax.annotate('{0:0.0f}'.format(y), (x.mean(), y),ha='center', va='bottom')
my_df_level['status'] = 'None'
def getStatus(x):
        for indx , row in x.iterrows():
            if x['salary_value_level'][indx] < 400:
                my_df_level['status'][indx] = 'Poor'
            elif 1000 >= x['salary_value_level'][indx] >= 400:
                my_df_level['status'][indx] = 'Average'
            elif x['salary_value_level'][indx] > 1000:
                my_df_level['status'][indx] = 'Rich'

getStatus(my_df_level)
my_df_level.tail(20)

my_df_matrix = my_df_level[['education_level','age','years_experience','salary_value_level','status','years_coding_to_analyze','salary_value']].dropna()
#educationDict = 
my_df_matrix['education_level'].replace({"No formal education past high school":float(0.0),'Some college/university study without earning a bachelor’s degree':1.0,'Professional degree':2.0,'Bachelor’s degree':float(3.0),'Master’s degree':4.0,'Doctoral degree':5.0},inplace=True)
my_df_matrix = my_df_matrix[my_df_matrix['education_level']!='I prefer not to answer']

my_df_matrix['age'].replace({'18-21':20.0, '22-24':23.0, '25-29':27.0, '30-34':32.0, '35-39':37.0, '40-44':42.0,'45-49':47.0, '50-54':52.0, '55-59':57.0, '60-69':65.0, '70-79':75.0, '80+':82.0},inplace=True)

my_df_matrix['years_experience'].replace({'0-1':0.5, '1-2':1.5, '2-3':2.5, '3-4':3.5, '4-5':4.5, '5-10':7.5, '10-15':13, '15-20':18, '20-25':23, '25-30':28, '30+':35},inplace=True)
#my_df_matrix['years_coding_to_analyze'].replace({'0-1':0.5, '1-2':1.5, '2-3':2.5, '3-4':3.5, '4-5':4.5, '5-10':5.5, '10-20':15, '20-25':23, '25-30':28, '30+':35},inplace=True)
#def convertValues(df):
matrixData = my_df_matrix[['education_level','age','status','years_experience','salary_value_level']]
matrixData['education_level'] = matrixData['education_level'].apply(lambda x : float(x))
matrixData.tail(20)
sn.set(style="ticks", color_codes=True)
plot_kws={"s": 5}
g = sn.pairplot(matrixData, kind = 'reg', palette='Set1',hue="status",markers=[".", "_", "|"],hue_order = ('Rich','Average','Poor'))
matrixData1 = my_df_matrix[['education_level','age','status','years_experience','salary_value_level','salary_value']]
matrixData1['education_level'] = matrixData1['education_level'].apply(lambda x : float(x))
matrixData1.head()


fig, (ax1,ax2) = plt.subplots(2,2)
fig.set_size_inches(20, 15)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=0.6)
g1 = sn.violinplot(x="status", y="age", data=matrixData1,ax = ax1[0],scale = 'count')
g2 = sn.violinplot(x="status", y="education_level", data=matrixData1,ax = ax1[1],scale = 'count')
g3= sn.violinplot(x="status", y="years_experience", data=matrixData1,ax = ax2[0],scale = 'count')
g4 = sn.violinplot(x="status", y="salary_value", data=matrixData1,ax = ax2[1],scale = 'count')
# g.set_xticklabels(compSciPlot.salary,rotation=90)
# g.set_title( "Computer Science and IT Undergrad Salary Distribution")
# g.set_xlabel('Salary')
# g.set_ylabel('Frequency')



import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
train_cols = ['gender','age','country','education_level','undergrad_major','role','employer_industry','years_experience','language','years_coding_to_analyze','you_a_data_scientist']
df_ml = my_df[['gender','age','country','education_level','undergrad_major','role','employer_industry','years_experience','language','years_coding_to_analyze','you_a_data_scientist','salary_value']]
df_train = df_ml[train_cols]

for x in df_train.columns :
    lbl = LabelEncoder()
    df_train[x] = lbl.fit_transform(df_train[x].values.astype('str'))

    
df_ml_level = my_df_level[['gender','age','country','education_level','undergrad_major','role','employer_industry','years_experience','language','years_coding_to_analyze','you_a_data_scientist','salary_value']]
df_train_level = df_ml_level[train_cols]
df_train_level.head()

for x in df_train_level.columns :
    lbl_lvl = LabelEncoder()
    df_train_level[x] = lbl_lvl.fit_transform(df_train_level[x].values.astype('str'))


params = {
    'task':'train',
    'objective':'regression',
    'metric':'mape',
    'nthread':4,
    'learning_rate':0.08,
    'num_leaves':31,
    'colsample_bytree':0.9,
    'subsample':0.8,
    'max_depth':5,
    'verbose':-1
}

lgb_data = lgb.Dataset(df_train, df_ml['salary_value'].values)
clf = lgb.train(params, lgb_data, 150)
importance_df = pd.DataFrame()
importance_df['name'] = list(train_cols)
importance_df['importance'] = clf.feature_importance()
importance_df.sort_values(by = 'importance',ascending = False).head(20)
params = {
    'task':'train',
    'objective':'regression',
    'metric':'mape',
    'nthread':4,
    'learning_rate':0.08,
    'num_leaves':31,
    'colsample_bytree':0.9,
    'subsample':0.8,
    'max_depth':5,
    'verbose':-1
}

lgb_data = lgb.Dataset(df_train_level, my_df_level['salary_value_level'].values)
clf_lvl = lgb.train(params, lgb_data, 150)
importance_df_level = pd.DataFrame()
importance_df_level['name'] = list(train_cols)
importance_df_level['importance'] = clf_lvl.feature_importance()
test_lgb   = list(importance_df_level['importance'])

layout = dict(yaxis=go.layout.YAxis(title='Features'),
                   xaxis=go.layout.XAxis(
                       range=[-700, 700],
                        tickvals=[-700, -400,-200,0,200,400,700],
                       ticktext=[700, 400,200,0,200,400,700],
                       title='Importance'),
                   barmode='overlay',
                   bargap=0.1)
data = [go.Bar(
    
    y=importance_df_level.sort_values(by = 'importance',ascending = False)['name'],
               x=importance_df_level.sort_values(by = 'importance',ascending = False)['importance'],
               orientation='h',
               name='Level Salary',
               hoverinfo='x',
                xaxis='x1',
    yaxis='y1',
               marker=dict(color='#E91E63')
               ),
        go.Bar( y=importance_df.sort_values(by = 'importance',ascending = False)['name'],
               x=importance_df.sort_values(by = 'importance',ascending = False)['importance']*(-1),
               orientation='h',
               name='Salary in USD',
               hoverinfo='text', xaxis='x1',
    yaxis='y1',
               marker=dict(color='#9C27B0')
               )]

py.offline.iplot(dict(data=data, layout=layout), validate=False,filename='GRAPHS/doubleBarLGB') 
#source https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(clf_lvl)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(df_train_level)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values, df_train_level)

explainer = shap.TreeExplainer(clf)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(df_train)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values, df_train)

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

scores_df = pd.DataFrame()
scores_df['features'] = df_train_level.columns.tolist()

test_K_best = SelectKBest(score_func=chi2, k=4)
fit = test_K_best.fit(df_train_level, my_df_level['status'].values)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
scores_df['k_best'] = list(fit.scores_)
test_mut_clas = mutual_info_classif(df_train_level,my_df_level['status'].values)
print(test_mut_clas)
scores_df['mut_clas'] = list(test_mut_clas)

test_mut_regress = mutual_info_regression(df_train_level,my_df_level['salary_value'].values)
print(test_mut_regress)
scores_df['mut_regres'] = list(test_mut_regress)
from sklearn import preprocessing
scores_df['lgb'] = test_lgb

x = scores_df[['k_best','mut_clas','mut_regres','lgb']].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scores_df_scaled = pd.DataFrame(x_scaled)
scores_df_scaled['features'] = scores_df['features']
scores_df_scaled.columns = ['SelectKBest','mutual_info_classif','mutual_info_regression','LGB','features']
scores_df_scaled.head(11)
ax = sn.heatmap(scores_df_scaled[['SelectKBest','mutual_info_classif','mutual_info_regression','LGB']],linewidths=0.1,center= 0.4 , linecolor='white', cmap='bwr', yticklabels =scores_df_scaled['features'] )
summary_df = my_df_level[['age','country','role','employer_industry','years_experience','status','salary_value']]


age = summary_df[['age','status','country']]
age['count'] = age.groupby('age').transform('count')['country'].dropna()
x=age.groupby(('age','status')).count().reset_index()[['age','status','country']].dropna()
age.columns = ['age1','status1','country1','count1'] 
x.columns = ['age','status','count']

def getRatioAge(y):
    x['new_count'] = x['count'].astype(float)
    for indx , row in y.iterrows():
        x['new_count'][indx] = float(float(y['count'][indx])/float(age[( age['age1'] == y['age'][indx] ) & ( age['status1'] == y['status'][indx] )]['count1'].values[0]))
getRatioAge(x)


country_count_total =  summary_df[['salary_value','status','country']]
country_count_total['count'] = country_count_total.groupby('country').transform('count')['salary_value']
country_count_total = country_count_total[country_count_total['count']>100]
country = country_count_total.groupby(('country','status')).count().reset_index()#['country']
def getRatioCountry(y,age,s):
    country['new_count'] = country['count'].astype(float)
    for indx , row in y.iterrows():
        country['new_count'][indx] = float(float(y['count'][indx])/float(age[( age[s] == y[s][indx] ) & ( age['status'] == y['status'][indx] )]['count'].values[0]))
getRatioCountry(country,country_count_total,'country')
country = country.sort_values(by='new_count',ascending = False)

role_count_total =  summary_df[['salary_value','status','role']]
role_count_total['count'] = role_count_total.groupby('role').transform('count')['salary_value']
role = role_count_total.groupby(('role','status')).count().reset_index()#['country']
def getRatioRole(y,age,s):
    role['new_count'] = role['count'].astype(float)
    for indx , row in y.iterrows():
        role['new_count'][indx] = float(float(y['count'][indx])/float(age[( age[s] == y[s][indx] ) & ( age['status'] == y['status'][indx] )]['count'].values[0]))
getRatioRole(role,role_count_total,'role')
role = role.sort_values(by='new_count',ascending = False)


indus_count_total =  summary_df[['salary_value','status','employer_industry']]
indus_count_total['count'] = indus_count_total.groupby('employer_industry').transform('count')['salary_value']
indus = indus_count_total.groupby(('employer_industry','status')).count().reset_index()#['country']
def getRatioIndus(y,age,s):
    indus['new_count'] = indus['count'].astype(float)
    for indx , row in y.iterrows():
        indus['new_count'][indx] = float(float(y['count'][indx])/float(age[( age[s] == y[s][indx] ) & ( age['status'] == y['status'][indx] )]['count'].values[0]))
getRatioIndus(indus,indus_count_total,'employer_industry')
indus = indus.sort_values(by='new_count',ascending = False)

years_count_total =  summary_df[['salary_value','status','years_experience']]
years_count_total['count'] = years_count_total.groupby('years_experience').transform('count')['salary_value'].dropna()
years = years_count_total.groupby(('years_experience','status')).count().reset_index().dropna()#['country']
def getRatioYears(y,age,s):
    years['new_count'] = years['count'].astype(float)
    for indx , row in y.iterrows():
        years['new_count'][indx] = float(float(y['count'][indx])/float(age[( age[s] == y[s][indx] ) & ( age['status'] == y['status'][indx] )]['count'].values[0]))
getRatioYears(years,years_count_total,'years_experience')


sal_count_total =  summary_df[['salary_value','status','role']]
sal_count_total['count'] = sal_count_total.groupby('salary_value').transform('count')['role'].dropna()
sal = sal_count_total.groupby(('salary_value','status')).count().reset_index().dropna()#['country']
def getRatioSal(y,age,s):
    sal['new_count'] = sal['count'].astype(float)
    for indx , row in y.iterrows():
        sal['new_count'][indx] = float(float(y['count'][indx])/float(age[( age[s] == y[s][indx] ) & ( age['status'] == y['status'][indx] )]['count'].values[0]))
getRatioSal(sal,sal_count_total,'salary_value')


fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6)
fig.set_size_inches(20, 60)
hue_order = ['Rich','Average','Poor']
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=0, hspace=1)
pal = ['#ED07D7','#24EA02','#0F01F4']
g = sn.barplot(x = 'age',y ='new_count',palette = pal ,hue = 'status',data = x,hue_order=hue_order,orient='v',ax = ax1)
g.set_xticklabels(x.age.unique(),rotation=90)
g.set_title( "Age vs Wealth Group ")
g.set_xlabel('Age')
g.set_ylabel('% of each Group')


# for p in ax1[0].patches:
#     x=p.get_bbox().get_points()[:,0]
#     y=p.get_bbox().get_points()[1,1]
#     ax1[0].annotate('{:.1f}%'.format(100*y), (x.mean(), y),ha='center', va='bottom')

g1 = sn.barplot(x = 'country',y ='new_count',palette = pal,hue = 'status' ,hue_order=hue_order,data = country,orient='v',ax = ax2)
g1.set_xticklabels(country.country.unique(),rotation=90)
g1.set_title( "Country vs Wealth Group")
g1.set_xlabel('Country')
g1.set_ylabel('% of each Country')

    
# for p in ax1[1].patches:
#     x=p.get_bbox().get_points()[:,0]
#     y=p.get_bbox().get_points()[1,1]
#     ax1[1].annotate('{:.1f}%'.format(100*y/totalFineArts), (x.mean(), y),ha='center', va='bottom')
    
g2 = sn.barplot(x = 'role',y ='new_count',palette = pal ,hue = 'status',data = role,hue_order=hue_order,orient='v',ax = ax3)
g2.set_xticklabels(role.role.unique(),rotation=90)
g2.set_title( "Occupational Role vs Wealth Group")
g2.set_xlabel('Role')
g2.set_ylabel('% of each Role')


# for p in ax2[0].patches:
#     x=p.get_bbox().get_points()[:,0]
#     y=p.get_bbox().get_points()[1,1]
#     ax2[0].annotate('{:.1f}%'.format(100*y/totalCompSci), (x.mean(), y),ha='center', va='bottom')

g3 = sn.barplot(x = 'employer_industry',y ='new_count',palette = pal ,hue = 'status',hue_order=hue_order,data = indus,orient='v',ax = ax4)
g3.set_xticklabels(indus.employer_industry.unique(),rotation=90)
g3.set_title( "Industry vs Wealth Group")
g3.set_xlabel('Industry')
g3.set_ylabel('% of each Industry')

    
# for p in ax2[1].patches:
#     x=p.get_bbox().get_points()[:,0]
#     y=p.get_bbox().get_points()[1,1]
#     ax2[1].annotate('{:.1f}%'.format(100*y/totalFineArts), (x.mean(), y),ha='center', va='bottom')

g4 = sn.barplot(x = 'years_experience',y ='new_count',palette = pal,hue = 'status' ,hue_order=hue_order,data = years,orient='v',ax = ax5)
g4.set_xticklabels(years.years_experience.unique(),rotation=90)
g4.set_title( "Experience vs Wealth Group")
g4.set_xlabel('Experience')
g4.set_ylabel('% of each Experience Interval')


# for p in ax3[0].patches:
#     x=p.get_bbox().get_points()[:,0]
#     y=p.get_bbox().get_points()[1,1]
#     ax3[0].annotate('{:.1f}%'.format(100*y/totalCompSci), (x.mean(), y),ha='center', va='bottom')

g5 = sn.barplot(x = 'salary_value',y ='new_count',palette = pal ,hue ='status',hue_order=hue_order,data = sal,orient='v',ax = ax6)
g5.set_xticklabels(sal.salary_value.unique(),rotation=90)
g5.set_title( "Income in USD vs Wealth Group")
g5.set_xlabel('Income in USD')
g5.set_ylabel('% of each Income Interval')

    
# for p in ax3[1].patches:
#     x=p.get_bbox().get_points()[:,0]
#     y=p.get_bbox().get_points()[1,1]
#     ax3[1].annotate('{:.1f}%'.format(100*y/totalFineArts), (x.mean(), y),ha='center', va='bottom')
    
    




