# Importing packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



cols = ['CASE_STATUS','EMPLOYMENT_START_DATE','EMPLOYER_NAME',

        'EMPLOYER_STATE','JOB_TITLE','SOC_NAME','PREVAILING_WAGE','PW_UNIT_OF_PAY']



# Importing the Dataset

df18 = pd.read_csv('../input/h1b18.csv', usecols = cols)

df17 = pd.read_csv('../input/h1b17.csv', usecols = cols)

df16 = pd.read_csv('../input/h1b16.csv')
# Removing columns that are not necessary from data of df18 and df17

df17 = df17[['CASE_STATUS','EMPLOYMENT_START_DATE','EMPLOYER_NAME',

        'EMPLOYER_STATE','JOB_TITLE','SOC_NAME','PREVAILING_WAGE','PW_UNIT_OF_PAY']]



df18 = df18[['CASE_STATUS','EMPLOYMENT_START_DATE','EMPLOYER_NAME',

        'EMPLOYER_STATE','JOB_TITLE','SOC_NAME','PREVAILING_WAGE','PW_UNIT_OF_PAY']]



# Changing the EMPLOYMENT_START_DATE into a year to comply with 2016 data

df18['EMPLOYMENT_START_DATE'] = pd.to_datetime(df18['EMPLOYMENT_START_DATE'], format = '%m/%d/%y')

df18['EMPLOYMENT_START_DATE'] = df18['EMPLOYMENT_START_DATE'].apply(lambda x: x.year)



df17['EMPLOYMENT_START_DATE'] = pd.to_datetime(df17['EMPLOYMENT_START_DATE'], format = '%Y-%m-%d')

df17['EMPLOYMENT_START_DATE'] = df17['EMPLOYMENT_START_DATE'].apply(lambda x: x.year)
# Only take years 2017 from df17 and years 2018 from df18

df17 = df17.loc[df17['EMPLOYMENT_START_DATE'] == 2017, :]

df18 = df18.loc[df18['EMPLOYMENT_START_DATE'] == 2018, :]
# Change PREVAILING WAGE into yearly wages (Split into Year, Week, Month, Hourly, Bi-weekly)

df18['PREVAILING_WAGE'] = df18['PREVAILING_WAGE'].apply(lambda x: str(x).replace(',',''))

df18.loc[df18.PREVAILING_WAGE == 0, 'PREVAILING_WAGE'] = np.nan

df18.loc[df18.PW_UNIT_OF_PAY == 'Hour', 'PREVAILING_WAGE'] = df18.loc[df18.PW_UNIT_OF_PAY == 'Hour', 'PREVAILING_WAGE'].apply(lambda x:float(x)*1638)

df18.loc[df18.PW_UNIT_OF_PAY == 'Week', 'PREVAILING_WAGE'] = df18.loc[df18.PW_UNIT_OF_PAY == 'Week', 'PREVAILING_WAGE'].apply(lambda x:float(x)*52)

df18.loc[df18.PW_UNIT_OF_PAY == 'Bi-Weekly', 'PREVAILING_WAGE'] = df18.loc[df18.PW_UNIT_OF_PAY == 'Bi-Weekly', 'PREVAILING_WAGE'].apply(lambda x:float(x)*26)

df18.loc[df18.PW_UNIT_OF_PAY == 'Month', 'PREVAILING_WAGE'] = df18.loc[df18.PW_UNIT_OF_PAY == 'Month', 'PREVAILING_WAGE'].apply(lambda x:float(x)*12)

df17['PREVAILING_WAGE'] = df17['PREVAILING_WAGE'].apply(lambda x: str(x).replace(',',''))

df17.loc[df17.PREVAILING_WAGE == 0, 'PREVAILING_WAGE'] = np.nan

df17.loc[df17.PW_UNIT_OF_PAY == 'Hour', 'PREVAILING_WAGE'] = df17.loc[df17.PW_UNIT_OF_PAY == 'Hour', 'PREVAILING_WAGE'].apply(lambda x:float(x)*1638)

df17.loc[df17.PW_UNIT_OF_PAY == 'Week', 'PREVAILING_WAGE'] = df17.loc[df17.PW_UNIT_OF_PAY == 'Week', 'PREVAILING_WAGE'].apply(lambda x:float(x)*52)

df17.loc[df17.PW_UNIT_OF_PAY == 'Bi-Weekly', 'PREVAILING_WAGE'] = df17.loc[df17.PW_UNIT_OF_PAY == 'Bi-Weekly', 'PREVAILING_WAGE'].apply(lambda x:float(x)*26)

df17.loc[df17.PW_UNIT_OF_PAY == 'Month', 'PREVAILING_WAGE'] = df17.loc[df17.PW_UNIT_OF_PAY == 'Month', 'PREVAILING_WAGE'].apply(lambda x:float(x)*12)

# Remove PW_UNIT_OF_PAY 

df17 = df17[['CASE_STATUS','EMPLOYMENT_START_DATE','EMPLOYER_NAME',

        'EMPLOYER_STATE','JOB_TITLE','SOC_NAME','PREVAILING_WAGE']]



df18 = df18[['CASE_STATUS','EMPLOYMENT_START_DATE','EMPLOYER_NAME',

        'EMPLOYER_STATE','JOB_TITLE','SOC_NAME','PREVAILING_WAGE']]
# Dictionary of all States and Shortened State

states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado",

          "CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho",

          "IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana",

          "ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",

          "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey",

          "NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",

          "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota",

          "TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",

          "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}

states = dict((v.upper(), k.upper()) for k, v in states.items())



# Remove the city from the WORKSITE 

df16['WORKSITE'] = df16['WORKSITE'].apply(lambda x: x.split(',')[1])

# Removing the space infront of the states from WORKSITE

df16['WORKSITE'] = df16['WORKSITE'].apply(lambda x: x[1:])

# Replace Worksite with shortened States

df16['WORKSITE'].replace(states, inplace = True)

# Replace WORKSITE with EMPLOYER_STATE

df16.rename(columns={'YEAR':'EMPLOYMENT_START_DATE','WORKSITE':'EMPLOYER_STATE'}, inplace=True)





# Removing unnecessary columns of df16

df16 = df16[['CASE_STATUS','EMPLOYMENT_START_DATE','EMPLOYER_NAME',

        'EMPLOYER_STATE','JOB_TITLE','SOC_NAME','PREVAILING_WAGE']]
# Combining df16, df17, df18 into one dataframe

df = pd.concat([df16,df17,df18])

print(df.head())
# Cleaning the SOC_NAME 

df['SOC_NAME'] = df['SOC_NAME'].apply(lambda x: str(x).upper())

df['SOC_NAME'] = df['SOC_NAME'].apply(lambda x:str(x).replace('COMPUTER SYSTEMS ANALYSTS','COMPUTER SYSTEMS ANALYST'))

# Looking at the number of applications per year 

year_app = df['EMPLOYMENT_START_DATE'].value_counts()

print(year_app)
# Bar Chart of the number of applications

year_app = year_app.sort_index()

obama = pd.Series(year_app.values[0:6], index = year_app.index[0:6])

trump = pd.Series(year_app.values[6:], index = year_app.index[6:])



fig, ax = plt.subplots()

plt.bar(obama.index, obama.values, align = 'center', alpha = 0.7,  linewidth=0)

plt.bar(trump.index, trump.values, align = 'center', alpha = 0.7,  linewidth=0, color = '#FF0000')

plt.legend(['Obama\'s Presidency','Trump\'s Presidency'],loc=2)

plt.title('The Number of Applications from 2011 to 2018')

plt.savefig('no of applications.png')
# Heatmap of U.S. H1B visa applicants

# Import statements 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import plotly as plotly

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.io as pio



# Organizing by state 

df_heatmap = df.groupby('EMPLOYER_STATE')['CASE_STATUS'].count()



scl = [

    [0.0, 'rgb(242,240,247)'],

    [0.2, 'rgb(218,218,235)'],

    [0.4, 'rgb(188,189,220)'],

    [0.6, 'rgb(158,154,200)'],

    [0.8, 'rgb(117,107,177)'],

    [1.0, 'rgb(84,39,143)']

]



data = [go.Choropleth(

    colorscale = scl,

    autocolorscale = False,

    locations = df_heatmap.index,

    z = df_heatmap.values.astype(float),

    locationmode = 'USA-states',

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(255,255,255)',

            width = 2

        )),

    colorbar = go.choropleth.ColorBar(

        title = "")

)]



layout = go.Layout(

    title = go.layout.Title(

        text = 'H1B Visa Applicants'

    ),

    geo = go.layout.Geo(

        scope = 'usa',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

)



fig = go.Figure(data = data, layout = layout)

plotly.tools.set_credentials_file(username='jpar746', api_key='z4DayxsEIqnFEGvEUhPO')

plotly.offline.iplot(fig, filename = 'd3-cloropleth-map')
# Comparing the number of applications that are accepted or denied (Ignore withdrawn)

accept_arr = df['CASE_STATUS'].value_counts()

print(accept_arr)

accept_val = list(accept_arr)[0:4]

accept_label = list(accept_arr.index)[0:4]



# Out of all the applications we can see that only a small percentage are denied (Exactly 2.67%)

denied = (accept_arr['DENIED'] / df.shape[0]).round(4)

print(str(denied*100) + '%')
# Pie chart of accepted and denied applications 

fig, ax = plt.subplots()

ax.pie(accept_val, explode = (0,0,0,0.25), labels = accept_label, autopct = '%1.1f%%',

       pctdistance=1.2, labeldistance=1.4, wedgeprops={'alpha':0.7})

ax.axis('equal')

plt.title('Case Status')

plt.show()
# Looking at the percentage denied over the years 

accept_per = df.groupby('EMPLOYMENT_START_DATE')['CASE_STATUS'].value_counts()

year_per = []

year = []

for i in range(2011,2019):

    year_per.append((accept_per.loc[[i,'DENIED'],'DENIED'].values[0] / accept_per.loc[[i]].sum())*100)

    year.append(i)

denied_per = pd.Series(year_per, index = year)

print(denied_per)
# Scatter plot to look at the increase and decrease in denied applications

plt.plot(denied_per.index, denied_per.values, 'bo', linestyle='dashed')

plt.title('Percentage of Denied Applicants')

plt.xlabel('Year')

plt.ylabel('% of Denied Applicants')

plt.show()
uscis = pd.Series([95.7,93.9,92.6,84.5], index=[2015,2016,2017,2018])

fig, ax = plt.subplots()

bars = plt.bar(uscis.index, uscis.values, align = 'center', alpha = 0.7)

plt.xticks([2015,2016,2017,2018])

# Removing the frame

for spine in plt.gca().spines.values():

    spine.set_visible(False)

# Removing the small ticks 

ax.tick_params(axis=u'both', which=u'both',length=0)

# Removing Y ticks 

ax.set_yticklabels([])

# Placing the values above the bar 

for rect in bars:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 0.9*height, str(height)+'%',

                ha='center', va='bottom', color = 'w')

bars[2].set_color('#FF0000')

bars[3].set_color('#FF0000')

plt.title('Percentages of H-1B Accepted')

plt.savefig('perc of applications.png')

plt.show()
# Looking at 20 companies that provide the most H1B visas

common_co = df['EMPLOYER_NAME'].value_counts()

print(common_co[:20])
# Bar Chart of companies that provide the most H1B visas

common_co_name = list(common_co[0:10].index)

common_co_freq = list(common_co[0:10])



y_pos = np.arange(len(common_co_name))



# Plotting a horizontal bar chart

fig, ax = plt.subplots()

bars = plt.barh(y_pos, common_co_freq, align = 'center', alpha = 0.7)



# Labeling the bar chart

plt.yticks(y_pos, common_co_name)



# Removing the frame for a cleaner visual

for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

# Removing the small y ticks

ax.tick_params(axis=u'both', which=u'both',length=0)



# Directly labeling the values of the bar chart 

i = 0

for bar in bars:

    plt.gca().text(bar.get_x() + bar.get_width()-21000, i-0.15, str(int(bar.get_width())), 

                 ha='left', color='w', fontsize=11)

    i = i+1

    

# Removing the x-ticks

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)



# Setting the title 

plt.title('Companies that provide the most H1B Visas')

plt.savefig('co of applications.png', bbox_inches='tight')
# Number of H-1B Visa Applicants hired by each company in 2015

df15 = df[df['EMPLOYMENT_START_DATE']==2015]

hire15 = df15['EMPLOYER_NAME'].value_counts()

print(hire15[0:20])
# Bar Chart of companies that provide the most H1B visas in 2015

hire15_name = list(hire15[0:10].index)

hire15_freq = list(hire15[0:10])



y_pos = np.arange(len(hire15_name))



# Plotting a horizontal bar chart

fig, ax = plt.subplots()

bars = plt.barh(y_pos, hire15_freq, align = 'center', alpha = 0.7)



# Labeling the bar chart

plt.yticks(y_pos, hire15_name)



# Removing the frame for a cleaner visual

for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

# Removing the small y ticks

ax.tick_params(axis=u'both', which=u'both',length=0)



# Directly labeling the values of the bar chart 

i = 0

for bar in bars:

    plt.gca().text(bar.get_x() + bar.get_width()-3650, i-0.15, str(int(bar.get_width())), 

                 ha='left', color='w', fontsize=11)

    i = i+1

    

# Removing the x-ticks

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)



# Setting the title 

plt.title('Number of H-1B Visa Applicants by each company in 2015')

plt.savefig('hire15 applications_fi.png', bbox_inches='tight')
# Number of H-1B Visa Applicants hired by each company in 2018

df18 = df[df['EMPLOYMENT_START_DATE']==2018]

hire18 = df18['EMPLOYER_NAME'].value_counts()

print(hire18[0:20])
# Bar Chart of companies that provide the most H1B visas in 2018

hire18_name = list(hire18[0:10].index)

hire18_freq = list(hire18[0:10])



y_pos = np.arange(len(hire18_name))



# Plotting a horizontal bar chart

fig, ax = plt.subplots()

bars = plt.barh(y_pos, hire18_freq, align = 'center', alpha = 0.7)



# Labeling the bar chart

plt.yticks(y_pos, hire18_name)



# Removing the frame for a cleaner visual

for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

# Removing the small y ticks

ax.tick_params(axis=u'both', which=u'both',length=0)



# Directly labeling the values of the bar chart 

i = 0

for bar in bars:

    plt.gca().text(bar.get_x() + bar.get_width()-1700, i-0.15, str(int(bar.get_width())), 

                 ha='left', color='w', fontsize=11)

    i = i+1

    

# Removing the x-ticks

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)



# Setting the title 

plt.title('Number of H-1B Visa Applicants by each company in 2018')

plt.savefig('hire18 applications_f.png', bbox_inches='tight')
# The 10 most common types of jobs that H1B visa are granted for

common_type = df['SOC_NAME'].value_counts()

print(list(common_type[:10].index))

# From the first 10 jobs, we can see that majority of these jobs are related to technology (computers)
# The 10 most common job titles 

common_job = df['JOB_TITLE'].value_counts()

print(list(common_job[:10].index))
# Looking at data science roles (related like business analyst, data analyst, data engineer) for H1B Visas

data_science_only = df.loc[df['JOB_TITLE']=='DATA SCIENTIST']

bus_analyst = df.loc[df['JOB_TITLE']=='BUSINESS ANALYST']

data_analyst = df.loc[df['JOB_TITLE']=='DATA ANALYST']

data_eng = df.loc[df['JOB_TITLE']=='DATA ENGINEER']

data_science = pd.concat([data_science_only,bus_analyst,data_analyst,data_eng])



# Number of data science and similar roles h1b visas granted

print(data_science.shape[0])
# Looking at the prevailing wages of data science jobs vs other h1b visa jobs

data_science['PREVAILING_WAGE'].describe()



# Non-data science jobs' salaries

not_data_science = df[(df['JOB_TITLE'] != 'DATA SCIENTIST | BUSINESS ANALYST | DATA ANALYST | DATA ENGINEER')]

not_data_science['PREVAILING_WAGE'].describe()
# Boxplot for data science and non data science wages

# Turning values into lists for plotting

data_science = data_science.reset_index(drop=True)

not_data_science = not_data_science.reset_index(drop=True)

plot_box_df = pd.concat([not_data_science['PREVAILING_WAGE'],data_science['PREVAILING_WAGE']],

                        keys=['Not Data Science','Data Science'],axis=1)



sns.boxplot( data = plot_box_df, palette="Blues", showfliers=False)

axes = plt.gca()

axes.set_ylim([0,150000])

plt.title('Wages for Non Data Science and Data Science Jobs')

plt.show()
# Location for Data Science jobs (H1B Applicants)

ds_heatmap = data_science.groupby('EMPLOYER_STATE')['CASE_STATUS'].count()



scl = [

    [0.0, 'rgb(242,240,247)'],

    [0.2, 'rgb(218,218,235)'],

    [0.4, 'rgb(188,189,220)'],

    [0.6, 'rgb(158,154,200)'],

    [0.8, 'rgb(117,107,177)'],

    [1.0, 'rgb(84,39,143)']

]



data = [go.Choropleth(

    colorscale = scl,

    autocolorscale = False,

    locations = ds_heatmap.index,

    z = ds_heatmap.values.astype(float),

    locationmode = 'USA-states',

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(255,255,255)',

            width = 2

        )),

    colorbar = go.choropleth.ColorBar(

        title = "")

)]



layout = go.Layout(

    title = go.layout.Title(

        text = 'H1B Data Science Applicants'

    ),

    geo = go.layout.Geo(

        scope = 'usa',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

)



fig = go.Figure(data = data, layout = layout)

plotly.tools.set_credentials_file(username='jpar746', api_key='z4DayxsEIqnFEGvEUhPO')

plotly.offline.iplot(fig, filename = 'd3-cloropleth-map')