import os

import re

import warnings

import pandas as pd

import numpy as np

import seaborn as sns 

import plotly.graph_objs as go

import plotly.tools as tools

import plotly.offline as ply

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

from mpl_toolkits.mplot3d import axes3d

from wordcloud import WordCloud, STOPWORDS

from plotnine import *

from os import path

from PIL import Image

##### configurations #####

ply.init_notebook_mode(connected=True)

color = sns.color_palette()

%matplotlib inline

warnings.filterwarnings('ignore')
print(os.listdir("../input/data-science-for-good-careervillage/"))
PATH = '../input/data-science-for-good-careervillage/'

students = pd.read_csv(PATH + 'students.csv')

professionals = pd.read_csv(PATH + 'professionals.csv')

matches = pd.read_csv(PATH + 'matches.csv')

questions = pd.read_csv(PATH + 'questions.csv')

questions_score = pd.read_csv(PATH + 'question_scores.csv')

answers = pd.read_csv(PATH + 'answers.csv')

answers_score = pd.read_csv(PATH + 'answer_scores.csv')

comments = pd.read_csv(PATH + 'comments.csv')

emails = pd.read_csv(PATH + 'emails.csv')

groups = pd.read_csv(PATH + 'groups.csv')

group_memberships = pd.read_csv(PATH + 'group_memberships.csv')

school_memberships = pd.read_csv(PATH + 'school_memberships.csv')

tags = pd.read_csv(PATH + 'tags.csv')

tag_questions = pd.read_csv(PATH + 'tag_questions.csv')

tag_users = pd.read_csv(PATH + 'tag_users.csv')
students.head()
students['students_daily_joined'] = pd.to_datetime(students['students_date_joined']).dt.to_period('D')

students['students_month_joined'] = pd.to_datetime(students['students_date_joined']).dt.to_period('M')

students['students_year_joined'] = pd.to_datetime(students['students_date_joined']).dt.to_period('Y')

students['students_weekly_joined'] = pd.to_datetime(students['students_date_joined']).dt.to_period('W')
plt.style.use('ggplot')

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



## Daily Registrations

students_register = students[['students_daily_joined', 

                              'students_id']].groupby(['students_daily_joined']).count().reset_index()

students_register = students_register.set_index('students_daily_joined')

plt.rcParams['figure.figsize'] = (8, 6)

max_date = max(students_register.index)

min_date = min(students_register.index)

ax = students_register.plot(color='blue', ax=ax1, label='Registrations')

ax.axvspan(min_date, '2011-12-31', color='green', alpha=0.3)

ax.axvspan('2019-01-01', max_date, color='green', alpha=0.3)

ax.plot([min_date], [students_register.values[0]], '>r', markersize=10)

ax.plot([max_date], [students_register.values[-1]], 'or', markersize=10)

ax.annotate('Highest # of Registrations', xy=('2017-08-01', 400),  xycoords='data',

             xytext=(0, 100), textcoords='offset points',

             size=13, ha='right', va="center",

             bbox=dict(boxstyle="round", alpha=0.1),

             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

ax.set_xlabel('Date in Days')

ax.set_ylabel('Number of Students Registered')

ax.set_title('All time Students Registration by Day')



## Weekly Registrations

students_register = students[['students_weekly_joined', 

                              'students_id']].groupby(['students_weekly_joined']).count().reset_index()

students_register = students_register.set_index('students_weekly_joined')

max_date = max(students_register.index)

min_date = min(students_register.index)

plt.rcParams['figure.figsize'] = (8, 6)

ax = students_register.plot(color='blue', ax=ax2)

ax.axvspan(min_date, '2011-12-31', color='green', alpha=0.3)

ax.axvspan('2019-01-01', max_date, color='green', alpha=0.3)

ax.set_xlabel('Date in weeks')

ax.set_ylabel('Number of Students Registered')

ax.set_title('All time Students Registration by Weekly')



## Monthly Registrations

students_register = students[['students_month_joined', 

                              'students_id']].groupby(['students_month_joined']).count().reset_index()

students_register = students_register.set_index('students_month_joined')

plt.rcParams['figure.figsize'] = (8, 6)

ax = students_register.plot(color='blue', ax=ax3)

max_date = max(students_register.index)

min_date = min(students_register.index)

ax.axvspan(min_date, '2011-12', color='green', alpha=0.3)

ax.axvspan('2019-01', max_date, color='green', alpha=0.3)

ax.axvline('2011-12-31', color='green', linestyle='--')

ax.set_xlabel('Date in Months')

ax.set_ylabel('Number of Students Registered')

ax.set_title('All time Students Registration by Month')



## Yearly Registrations

students_register = students[['students_year_joined', 

                              'students_id']].groupby(['students_year_joined']).count().reset_index()

students_register = students_register.set_index('students_year_joined')

plt.rcParams['figure.figsize'] = (8, 6)

ax = students_register.plot(color='blue', ax=ax4)

max_date = max(students_register.index)

min_date = min(students_register.index)

ax.axvspan(min_date, '2011-12-31', color='green', alpha=0.3)

ax.axvspan('2019-01-01', max_date, color='green', alpha=0.3)

ax.set_xlabel('Date in Years')

ax.set_ylabel('Number of Students')

ax.set_title('All time Students Registration by Year')



plt.show()
professionals.head()
professionals['professionals_daily_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.to_period('D')

professionals['professionals_month_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.to_period('M')

professionals['professionals_year_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.to_period('Y')

professionals['professionals_weekly_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.to_period('W')
plt.style.use('ggplot')

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



## Daily Registrations

professionals_register = professionals[['professionals_daily_joined', 

                              'professionals_id']].groupby(['professionals_daily_joined']).count().reset_index()

professionals_register = professionals_register.set_index('professionals_daily_joined')

max_date = max(professionals_register.index)

min_date = min(professionals_register.index)

plt.rcParams['figure.figsize'] = (8, 6)

ax = professionals_register.plot(color='blue', ax=ax1)

max_date = max(professionals_register.index)

min_date = min(professionals_register.index)

ax.axvspan(min_date, '2011-12-31', color='green', alpha=0.3)

ax.axvspan('2019-01-01', max_date, color='green', alpha=0.3)

ax.set_xlabel('Date in Days')

ax.set_ylabel('Number of Students Registered')

ax.set_title('All time Student Registration by Day')



## Weekly Registrations

professionals_register = professionals[['professionals_weekly_joined', 

                              'professionals_id']].groupby(['professionals_weekly_joined']).count().reset_index()

professionals_register = professionals_register.set_index('professionals_weekly_joined')



plt.rcParams['figure.figsize'] = (8, 6)

ax = professionals_register.plot(color='blue', ax=ax2)

max_date = max(professionals_register.index)

min_date = min(professionals_register.index)

ax.axvspan(min_date, '2011-12-31', color='green', alpha=0.3)

ax.axvspan('2019-01-01', max_date, color='green', alpha=0.3)

ax.set_xlabel('Date in weeks')

ax.set_ylabel('Number of Students Registered')

ax.set_title('All time Student Registration by Weekly')



## Monthly Registrations

professionals_register = professionals[['professionals_month_joined', 

                              'professionals_id']].groupby(['professionals_month_joined']).count().reset_index()

professionals_register = professionals_register.set_index('professionals_month_joined')

plt.rcParams['figure.figsize'] = (8, 6)

ax = professionals_register.plot(color='blue', ax=ax3)

max_date = max(professionals_register.index)

min_date = min(professionals_register.index)

ax.axvspan(min_date, '2011-12', color='green', alpha=0.3)

ax.axvspan('2019-01', max_date, color='green', alpha=0.3)

ax.axvline('2011-12-31', color='green', linestyle='--')

ax.set_xlabel('Date in Months')

ax.set_ylabel('Number of Students Registered')

ax.set_title('All time Student Registration by Month')



## Yearly Registrations

professionals_register = professionals[['professionals_year_joined', 

                              'professionals_id']].groupby(['professionals_year_joined']).count().reset_index()

professionals_register = professionals_register.set_index('professionals_year_joined')

plt.rcParams['figure.figsize'] = (8, 6)

ax = professionals_register.plot(color='blue', ax=ax4)

max_date = max(professionals_register.index)

min_date = min(professionals_register.index)

ax.axvspan(min_date, '2011-12-31', color='green', alpha=0.3)

ax.axvspan('2019-01-01', max_date, color='green', alpha=0.3)

ax.set_xlabel('Date in Years')

ax.set_ylabel('Number of Students')

ax.set_title('All time Students Registration by Year')



plt.show()
fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



professionals_register = professionals[['professionals_month_joined', 

                              'professionals_id']].groupby(['professionals_month_joined']).count().reset_index()

students_register = students[['students_month_joined', 

                              'students_id']].groupby(['students_month_joined']).count().reset_index()

professionals_register = professionals_register.set_index('professionals_month_joined')

students_register = students_register.set_index('students_month_joined')

plt.rcParams['figure.figsize'] = (8, 6)

ax = professionals_register.plot(color='blue', ax=ax1)





plt.rcParams['figure.figsize'] = (8, 6)

ax = students_register.plot(color='red', ax=ax1)

ax.set_xlabel('Date in Months')

ax.set_ylabel('Number of Registrations')

ax.set_title('All time Registration by Month')





professionals_register = professionals[['professionals_year_joined', 

                              'professionals_id']].groupby(['professionals_year_joined']).count().reset_index()

students_register = students[['students_year_joined', 

                              'students_id']].groupby(['students_year_joined']).count().reset_index()

professionals_register = professionals_register.set_index('professionals_year_joined')

students_register = students_register.set_index('students_year_joined')

plt.rcParams['figure.figsize'] = (8, 6)

ax = professionals_register.plot(color='blue', ax=ax2)





plt.rcParams['figure.figsize'] = (8, 6)

ax = students_register.plot(color='red', ax=ax2)

ax.set_xlabel('Date in Years')

ax.set_ylabel('Number of Registrations')

ax.set_title('All time Registration by Years')



plt.show()
list_countries = ["Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",

"Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",

"Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",

"Burkina Faso", "Burundi", "Côte d'Ivoire", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",

"Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica", "Croatia", "Cuba", "Cyprus",

"Czech Republic", "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador",

"Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji", "Finland", "France", "Gabon",

"Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",

"Holy See", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",

"Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",

"Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macedonia", "Madagascar", "Malawi", "Malaysia",

"Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco",

"Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",

"Nicaragua", "Niger", "Nigeria", "North Korea", "Norway", "Oman", "Pakistan", "Palau", "Palestine State", "Panama",

"Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",

"Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe",

"Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",

"Somalia", "South Africa", "South Korea", "South Sudan",  "Spain", "Sri Lanka", "Sudan", "Suriname","Swaziland", "Sweden",

"Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago",

"Tunisia", "Turkey",  "Turkmenistan", "Tuvalu", "Uganda",  "Ukraine", "United Arab Emirates", "United Kingdom", 

"United States", "Uruguay", "Uzbekistan", "Vanuatu",  "Venezuela", "Viet Nam", "Yemen",  "Zambia", "Zimbabwe"]



usa_codes = [['AL', 'Alabama'], ['AK', 'Alaska'],  ['AZ', 'Arizona'], ['AR', 'Arkansas'],

       ['CA', 'California'], ['CO', 'Colorado'], ['CT', 'Connecticut'], ['DE', 'Delaware'],

       ['FL', 'Florida'], ['GA', 'Georgia'], ['HI', 'Hawaii'], ['ID', 'Idaho'], ['IL', 'Illinois'],

       ['IN', 'Indiana'], ['IA', 'Iowa'], ['KS', 'Kansas'], ['KY', 'Kentucky'],['LA', 'Louisiana'], 

       ['ME', 'Maine'], ['MD', 'Maryland'], ['MA', 'Massachusetts'], ['MI', 'Michigan'], ['MN', 'Minnesota'],

       ['MS', 'Mississippi'], ['MO', 'Missouri'], ['MT', 'Montana'], ['NE', 'Nebraska'], ['NV', 'Nevada'], 

       ['NH', 'New Hampshire'], ['NJ', 'New Jersey'], ['NM', 'New Mexico'], ['NY', 'New York'], ['NC', 'North Carolina'], 

       ['ND', 'North Dakota'], ['OH', 'Ohio'], ['OK', 'Oklahoma'], ['OR', 'Oregon'], ['PA', 'Pennsylvania'],

       ['RI', 'Rhode Island'], ['SC', 'South Carolina'], ['SD', 'South Dakota'], ['TN', 'Tennessee'],

       ['TX', 'Texas'], ['UT', 'Utah'], ['VT', 'Vermont'], ['VA', 'Virginia'], ['WA', 'Washington'], 

       ['WV', 'West Virginia'], ['WI', 'Wisconsin'], ['WY', 'Wyoming']]

us_states = pd.DataFrame(data=usa_codes, columns=['Code', 'State'])





georgia_cities = ["Tbilisi", "Batumi", "Kutaisi", "Rustavi", "Gori", "Zugdidi", "Poti", "Khashuri", "Samtredia", "Senaki", 

"Zestafoni", "Marneuli", "Telavi", "Akhaltsikhe", "Kobuleti", "Ozurgeti", "Kaspi", "Chiatura", "Tsqaltubo"

,"Sagarejo", "Gardabani", "Borjomi", "Tqibuli", "Khoni", "Bolnisi", "Akhalkalaki", "Gurjaani", "Mtskheta"

,"Qvareli", "Akhmeta", "Kareli", "Lanchkhuti", "Tsalenjikha", "Dusheti", "Sachkhere", "Dedoplistsqaro",

"Lagodekhi", "Ninotsminda", "Abasha","Tsnori", "Terjola", "Martvili", "Jvari", "Khobi", "Vani", "Baghdati"

,"Vale", "Tetritsqaro", "Tsalka", "Dmanisi", "Oni", "Ambrolauri", "Sighnaghi", "Tsageri"]
def get_country(location, countries, states):

    ''' This module returns coutry for any given location '''

    address = ''

    for country in countries:

        if country in location:

            address = country

    

    if address == '':

        for state in states:

            if state in location:

                address = 'United States'

                

    return address
students.students_location = students.students_location.astype(str)

students['country'] = students.apply(lambda row: get_country(row['students_location'], 

                                                             list_countries, 

                                                             us_states['State']), axis=1)

students.loc[students.students_location.str.contains('Indiana'), 'country'] = 'United States'

students.loc[students.students_location.str.contains('North Carolina'), 'country'] = 'United States'

students.loc[students.students_location.str.contains('Georgia'), 'country'] = 'United States'

for city in georgia_cities:

    students.loc[students.students_location.str.contains(city), 'country'] = 'Georgia'
students_maps = pd.DataFrame(students.country.value_counts()).reset_index()

students_maps.columns=['country', 'total']

students_maps = students_maps.reset_index().drop('index', axis=1)
data = [ dict(

        type = 'choropleth',

        locations = students_maps['country'],

        locationmode = 'country names',

        z = students_maps['total'],

        text = students_maps['country'],

        colorscale =

            [[0,"rgb(5, 50, 172)"],[0.85,"rgb(40, 100, 190)"],[0.9,"rgb(70, 140, 245)"],

            [0.94,"rgb(90, 160, 245)"],[0.97,"rgb(106, 177, 247)"],[1,"rgb(220, 250, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Number of Students'),

      ) ]



layout = dict(

    title = 'Number of students Per Country',

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



figure = dict( data=data, layout=layout )

ply.iplot(figure, validate=False, filename='students')
#students_usa

def usa_state(location, usa_states):

    ''' This modules fix States Location for United States '''

    address = ''

    for index, state in usa_states.iterrows():

        if (state.Code in location) or (state.State in location):

            address = state.Code

    return address
students_usa = students[students.country == 'United States']

students_usa['States'] = students_usa.apply(lambda x: usa_state(x['students_location'], us_states), axis=1)

students_usa.loc[students_usa['States'] == '', 'States'] = 'NY'

counts = pd.DataFrame({'Code': students_usa.States.value_counts().index, 

                       'Total': students_usa.States.value_counts().values})

maps_df = counts.merge(us_states, on='Code', how='inner')
maps_df['text'] = maps_df['State'] + '<br>  ' + (maps_df['Total']).astype(str)+' Students'

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = [ dict(

        type='choropleth',

        colorscale = 'YlGnBu',

        autocolorscale = False,

        locations = maps_df['Code'],

        z = maps_df['Total'].astype(float),

        locationmode = 'USA-states',

        text = maps_df['text'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Number of Students")

        ) ]



layout = dict(

        title = 'CareerVillage.org Students <br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

ply.iplot( fig, filename='d3-cloropleth-map' )
professionals.professionals_location = professionals.professionals_location.astype(str)

professionals['country'] = professionals.apply(lambda row: get_country(row['professionals_location'], 

                                                             list_countries, 

                                                             us_states['State']), axis=1)

professionals.loc[professionals.professionals_location.str.contains('Indiana'), 'country'] = 'United States'

professionals.loc[professionals.professionals_location.str.contains('North Carolina'), 'country'] = 'United States'

professionals.loc[professionals.professionals_location.str.contains('Georgia'), 'country'] = 'United States'

for city in georgia_cities:

    professionals.loc[professionals.professionals_location.str.contains(city), 'country'] = 'Georgia'
professionals_maps = pd.DataFrame(professionals.country.value_counts()).reset_index()

professionals_maps.columns=['country', 'total']

professionals_maps = professionals_maps.reset_index().drop('index', axis=1)
data = [ dict(

        type = 'choropleth',

        locations = professionals_maps['country'],

        locationmode = 'country names',

        z = professionals_maps['total'],

        text = professionals_maps['country'],

        colorscale = [[0,"rgb(5, 50, 172)"],[0.85,"rgb(40, 100, 190)"],[0.9,"rgb(70, 140, 245)"],

            [0.94,"rgb(90, 160, 245)"],[0.97,"rgb(106, 177, 247)"],[1,"rgb(220, 250, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Number of Professionals'),

      ) ]



layout = dict(

    title = 'Number of Professionals Per Country',

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



figure = dict( data=data, layout=layout )

ply.iplot(figure, validate=False, filename='Professionals')
professionals_usa = professionals[professionals.country == 'United States']

professionals_usa['States'] = professionals_usa.apply(lambda x: usa_state(x['professionals_location'], us_states), axis=1)

professionals_usa.loc[professionals_usa['States'] == '', 'States'] = 'NY'

counts = pd.DataFrame({'Code': professionals_usa.States.value_counts().index, 

                       'Total': professionals_usa.States.value_counts().values})

maps_df = counts.merge(us_states, on='Code', how='inner')
maps_df['text'] = maps_df['State'] + '<br>  ' + (maps_df['Total']).astype(str)+' Students'

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = maps_df['Code'],

        z = maps_df['Total'].astype(float),

        locationmode = 'USA-states',

        #text = maps_df['text'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Number of Professionals")

        ) ]



layout = dict(

        title = 'CareerVillage.org Professionals <br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

ply.iplot( fig, filename='d3-cloropleth-map' )
plt.figure(figsize=(12, 8))

professionals.professionals_industry.value_counts()[:20][::-1].plot(kind='barh')

plt.show()
def find_company(headline):

    ''' This Function finds company at which Professionals work '''

    value = ''

    if ' at ' in str(headline):

        value = headline.split(' at ')[1]

    else:

        value = str(headline)

    return value
professionals['company'] = professionals.professionals_headline.apply(lambda x:find_company(x))
plt.figure(figsize=(12, 8))

professionals[(professionals.company!='nan')&(professionals.company!='--')].company.value_counts()[:20][::-1].plot(kind='barh')

plt.show()
clouds_company = professionals[(professionals.company!='nan')&(professionals.company!='--')]

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

mask = np.array(Image.open(path.join(d, "../input/careerimages/Job.png")))

stopwords = set(STOPWORDS)

wc = WordCloud(background_color="white", max_words=2000, mask=mask,

               stopwords=stopwords, contour_width=3, contour_color='white')

wc.generate(' '.join(clouds_company['company'].astype(str)))

plt.figure(figsize=(16, 12))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
clouds_company.head()
# professionals[(professionals.company!='nan')&(professionals.company!='--')].company.value_counts()

# sns.set(style="whitegrid")

# fig, ax = plt.subplots(figsize=(13, 7))

# sns.set_color_codes("pastel")

# sns.barplot(x="Total Members", y='Group Name', data=groups_data,

#             label="Members", color="b")



# sns.set_color_codes("muted")

# sns.barplot(x="Total Groups", y="Group Name", data=groups_data,

#             label="Total Groups", color="b")



# ax.legend(ncol=2, loc="lower right", frameon=True)

# ax.set(xlim=(0, 411), ylabel="Group Name", title='Number of Members In Groups',

#        xlabel="Number of Groups")

# sns.despine(left=True, bottom=True)
school_memberships.head()
len(school_memberships.school_memberships_school_id.unique())
questions.head()
structured_patterns = [

 (r'won\'t', 'will not'),

 (r'can\'t', 'cannot'),

 (r'i\'m', 'i am'),

 (r'ain\'t', 'is not'),

 (r'(\w+)\'ll', '\g<1> will'),

 (r'(\w+)n\'t', '\g<1> not'),

 (r'(\w+)\'ve', '\g<1> have'),

 (r'(\w+)\'s', '\g<1> is'),

 (r'(\w+)\'re', '\g<1> are'),

 (r'(\w+)\'d', '\g<1> would')

]



class RegexpReplacer(object):

    def __init__(self, patterns=structured_patterns):

         self.patterns = [(re.compile(regex), repl) for (regex, repl) in

         patterns]

            

    def replace(self, text):

        s = text

        for (pattern, repl) in self.patterns:

             s = re.sub(pattern, repl, s)

        return s
def strip_symbols(text):

    return ' '.join(re.compile(r'\W+', re.UNICODE).split(text))



def clean_text(df, column):

    

    df[column] = df[column].str.lower()

    df[column] = df[column].str.replace('\n',' ')

    replacer = RegexpReplacer()

    try:

        df[column] = df[column].apply(lambda x:replacer.replace(x))

        df[column] = df[column].apply(lambda x:strip_symbols(x))

    except:

        pass

    return df



def plot_wordcloud(df, column, name):

    

    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    mask = np.array(Image.open(path.join(d, '../input/careerimages/' + name + ".png")))

    stopwords = set(STOPWORDS)

    wc = WordCloud(background_color="white", max_words=2000, mask=mask,

                   stopwords=stopwords, contour_width=3, contour_color='white')

    wc.generate(' '.join(df[column]))

    plt.figure(figsize=(16, 12))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    plt.show()
questions = clean_text(questions, 'questions_title')

questions = clean_text(questions, 'questions_body')
questions.head()
#questions.questions_author_id.value_counts()
plot_wordcloud(questions, 'questions_title', 'WordArt')
plot_wordcloud(questions, 'questions_body', 'Quiz')
answers.head()
answers.answers_body = answers.answers_body.apply(lambda x: re.sub(re.compile('<.*?>'), '', str(x)))
answers.answers_body = answers.answers_body.astype(str)

answers = clean_text(answers, 'answers_body')
plot_wordcloud(answers, 'answers_body', 'Reply')
comments.head()
comments = clean_text(comments, 'comments_body')
comments.comments_body = comments.comments_body.astype(str)

plot_wordcloud(comments, 'comments_body', 'Comment')
fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(111)

questions.questions_date_added = pd.to_datetime(questions.questions_date_added).dt.to_period('M')

answers.answers_date_added = pd.to_datetime(answers.answers_date_added).dt.to_period('M')

comments.comments_date_added = pd.to_datetime(comments.comments_date_added).dt.to_period('M')

questions_added = questions[['questions_date_added', 

                              'questions_id']].groupby(['questions_date_added']).count().reset_index()

answers_added = answers[['answers_date_added', 

                              'answers_id']].groupby(['answers_date_added']).count().reset_index()

comments_added = comments[['comments_date_added', 

                              'comments_id']].groupby(['comments_date_added']).count().reset_index()

questions_added = questions_added.set_index('questions_date_added')

answers_added = answers_added.set_index('answers_date_added')

comments_added = comments_added.set_index('comments_date_added')

plt.rcParams['figure.figsize'] = (16, 8)

ax = questions_added.plot(color='blue', ax=ax1)

ax = answers_added.plot(color='red', ax=ax1)

ax = comments_added.plot(color='green', ax=ax1)

ax.set_xlabel('Date in Month')

ax.set_ylabel('Number of Interactions')

ax.set_title('Conversations On Questions, Answers and Comments')



plt.show()
questions_score.head()
sns.kdeplot(questions_score.score)

plt.show()
sns.kdeplot(questions_score.score, cumulative=True)

plt.show()
answers_score.head()
sns.kdeplot(answers_score.score)

plt.show()
sns.kdeplot(answers_score.score, cumulative=True)

plt.show()
matches.head()
emails.head()
emails.emails_frequency_level.value_counts()
X = emails.emails_frequency_level.value_counts()

colors = ['#F08080', '#1E90FF', '#FFFF99']



plt.pie(X.values, labels=X.index, colors=colors,

        startangle=90,

        explode = (0, 0, 0),

        autopct = '%1.2f%%')

plt.axis('equal')

plt.show()
groups.head()
groups_type = pd.DataFrame(groups.groups_group_type.value_counts(dropna=False))

groups_type = groups_type.reset_index().rename(columns={'index':'Group Name', 'groups_group_type': 'Total Groups'})

groups_type
group_memberships.head()
groups_data = pd.DataFrame(group_memberships.merge(groups.rename(columns={'groups_id':'group_memberships_group_id'}), 

                        on='group_memberships_group_id').groups_group_type.value_counts().reset_index()).rename(

columns={'index': 'Group Name', 'groups_group_type':'Total Members'}).merge(groups_type, on='Group Name')

groups_data
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(13, 7))

sns.set_color_codes("pastel")

sns.barplot(x="Total Members", y='Group Name', data=groups_data,

            label="Members", color="b")



sns.set_color_codes("muted")

sns.barplot(x="Total Groups", y="Group Name", data=groups_data,

            label="Total Groups", color="b")



ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 411), ylabel="Group Name", title='Number of Members In Groups',

       xlabel="Number of Groups")

sns.despine(left=True, bottom=True)
tags.head()
tag_users.head()
tag_questions.head()
users_tags = tag_users.rename(columns={'tag_users_tag_id': 'tags_tag_id'}).merge(tags, on='tags_tag_id')
users_tags.head()
users_tags.groupby(['tag_users_user_id', 

                    'tags_tag_name']).agg({'tags_tag_id':

                                           'count'}).reset_index().sort_values(by=['tags_tag_id'], ascending=False).head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
tfidf = TfidfVectorizer(stop_words='english')

## For making the questions asked we can add even tags for extensions [questions.questions_body, tags.name, ] this has to be merged with 

# user tags and questions tags

questions.questions_body = pd.concat([questions.questions_body, questions.questions_title], axis=1)

tfidf_matrix = tfidf.fit_transform(questions.questions_body)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(questions.index, index=questions['questions_title']).drop_duplicates()
def get_recommendations(questions_title, cosine_sim=cosine_sim):

    idx = indices[questions_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    question_indices = [i[0] for i in sim_scores]

    return questions.iloc[question_indices].merge(answers.rename(columns={'answers_question_id':

                                                                         'questions_id'}), on='questions_id')[['answers_body',

                                                                                                             'answers_author_id',

                                                                                                             'questions_body']]
answers.head()
get_recommendations('teacher career question')