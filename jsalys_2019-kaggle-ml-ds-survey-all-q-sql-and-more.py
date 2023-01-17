# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print('--------------------------------------------------------')

print('------+--+--------------+------+++--+++-+-+++-----------')

print('------+-+---------------+--------+--+-+-+-+-+-----------')

print('------++---+++--+++-+++-+-+++----+--+-+-+-+++-----------')

print('------+-+--+-+--+-+-+-+-+-+-+----+--+-+-+---+-----------')

print('------+--+-++++-+++-+++-+-+++---+++++++-+-+++-----------')

print('------------------+---+---------------------------------')

print('------------------+---+---------------------------------')

print('------------------+---+---------------------------------')

print('--------------------------------------------------------')
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import plotly.graph_objects as go

import numpy as np





%matplotlib inline
multiple_choice_responses_2019 = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
multiple_choice_responses_2018 = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')
multiple_choice_responses_2017 = pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding='ISO-8859-1')
# Building a dictionary for age of paricipants in 2017 survey. 



age_dic = {}



for age_range in multiple_choice_responses_2018.Q2.value_counts().index[:-2]:



    

    age_range_list = list(range(int(age_range.split('-')[0]),int(age_range.split('-')[1]) +1))

    

    age_count_list = []

    

   

    for age in age_range_list:

        

        age_count = sum(multiple_choice_responses_2017.Age == age)

        

        age_count_list.append(age_count)

        

    age_dic[age_range] = sum(age_count_list)

age_dic['80+'] = sum(multiple_choice_responses_2017.Age >= 80)

age_df_2017 = pd.DataFrame(age_dic.values(),age_dic.keys()) # Corrected df for age of respondens.


age_order_2017 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']

age_values_2017 = age_df_2017.T[age_order_2017].values[0]



age_order_2018 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']

age_values_2018 = multiple_choice_responses_2018.Q2.value_counts()[age_order_2018].values



age_order_2019 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']

age_values_2019 = multiple_choice_responses_2019.Q1.value_counts()[age_order_2019].values



age_order_2019_SQL = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']

age_values_2019_SQL = multiple_choice_responses_2019.Q1[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[age_order_2019].values



age_vals_2017_prc = age_values_2017/sum(age_values_2017)

age_vals_2018_prc = age_values_2018/sum(age_values_2018)

age_vals_2019_prc = age_values_2019/sum(age_values_2019)

age_vals_2019_prc_SQL = age_values_2019_SQL/sum(age_values_2019_SQL)



fig = go.Figure(data=[

    go.Bar(name='2017 data', x=age_order_2017, y=age_vals_2017_prc,),

    go.Bar(name='2018 data', x=age_order_2018, y=age_vals_2018_prc),

    go.Bar(name='2019 data', x=age_order_2019, y=age_vals_2019_prc),

    go.Bar(name='2019 data SQL', x=age_order_2019_SQL, y=age_vals_2019_prc_SQL),

    

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text='Respodents age',yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
# Respondents gender.



plt.figure(3, figsize=(20,15))

the_grid = GridSpec(1, 3)



# Q GennderSelect.

# Year 2017.



gender_values_2017 = multiple_choice_responses_2017.GenderSelect.value_counts().values

gender_2017 = multiple_choice_responses_2017.GenderSelect.value_counts().index



plt.subplot(the_grid[0, 0])



my_circle=plt.Circle((0,0), 0.9, color='white')

plt.pie(gender_values_2017,  autopct='%1.1f%%', labels=gender_2017, colors=['skyblue','pink','green','brown'])

p=plt.gcf()

plt.title("Gender distribution in 2017.")

p.gca().add_artist(my_circle)



# Q1.

# Year 2018.



gender_values_2018 = multiple_choice_responses_2018.Q1.value_counts().values[:4]

gender_2018 = multiple_choice_responses_2018.Q1.value_counts().index[:4]



plt.subplot(the_grid[0, 1])



my_circle=plt.Circle((0,0), 0.9, color='white')

plt.pie(gender_values_2018, autopct='%1.1f%%', labels=gender_2018, colors=['skyblue','pink','green','brown'])

p=plt.gcf()

plt.title("Gender distribution in 2018.")

p.gca().add_artist(my_circle)



# Q2.

# Year 2019.



plt.subplot(the_grid[0, 2])



gender_values_2019 = multiple_choice_responses_2019.Q2.value_counts().values[:4]

gender_2019 = multiple_choice_responses_2019.Q2.value_counts().index[:4]





my_circle=plt.Circle((0,0), 0.9, color='white')

plt.pie(gender_values_2019,  autopct='%1.1f%%',labels=gender_2019, colors=['skyblue','pink','green','brown'])

p=plt.gcf()

plt.title("Gender distribution in 2019.")

p.gca().add_artist(my_circle)



plt.show()
gender_values_2019 = multiple_choice_responses_2019.Q2[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().values[:4]

gender_2019 = multiple_choice_responses_2019.Q2[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().index[:4]





my_circle=plt.Circle((0,0), 0.9, color='white')

plt.pie(gender_values_2019,  autopct='%1.1f%%',labels=gender_2019, colors=['skyblue','pink','green','brown'])

p=plt.gcf()

plt.title("Gender distribution in 2019.")

p.gca().add_artist(my_circle)



plt.show()
import pycountry



country = multiple_choice_responses_2019.Q3[1:].value_counts().index

country = pd.Series(country)

country = country.replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom')

country = country.replace('United States of America','United States')

country = country.replace('Iran, Islamic Republic of...','Iran')

country = country.replace('Republic of Korea','Other')

country = country.replace('Hong Kong (S.A.R.)','Hong Kong')

country = country.replace('South Korea', 'Korea')



country_values = multiple_choice_responses_2019.Q3[1:].value_counts().values





countries_2019 = []

iso_alpha_2019 = []

countries_vals_2019 = []



for c,v in zip(country,multiple_choice_responses_2019.Q3[1:].value_counts().values):

    

    iso = pycountry.countries.search_fuzzy(c)[0].alpha_3

    pop = v*232009

 

    if c !="Other":

        countries_2019.append(c)

        iso_alpha_2019.append(iso)

        countries_vals_2019.append(pop)
df_countries_2019 = pd.DataFrame()

df_countries_2019['country'] = countries_2019

df_countries_2019['iso_alpha'] = iso_alpha_2019

df_countries_2019['pop'] = countries_vals_2019

df_countries_2019['year'] = '2019'
country_2017 = multiple_choice_responses_2017.Country.value_counts().index

country_2017 = pd.DataFrame(country_2017, columns=['Country'])

country_2017 = country_2017.replace('People \'s Republic of China', 'China')

country_2017 = country_2017.replace('Republic of China', 'China')

country_2017 = country_2017.replace('South Korea', 'Korea')



country_2017_values = multiple_choice_responses_2017.Country.value_counts().values



countries_2017 = []

iso_alpha_2017 = []

countries_vals_2017 = []



for c,v in zip(country_2017['Country'].values,country_2017_values):



    iso = pycountry.countries.search_fuzzy(c)[0].alpha_3

    pop = v*232009



    if c !="Other":

        countries_2017.append(c)

        iso_alpha_2017.append(iso)

        countries_vals_2017.append(pop)



    

df_countries_2017 = pd.DataFrame()

df_countries_2017['country'] = countries_2017

df_countries_2017['iso_alpha'] = iso_alpha_2017

df_countries_2017['pop'] = countries_vals_2017

df_countries_2017['year'] = '2017'
country_2018 = multiple_choice_responses_2018.Q3.value_counts().index

country_2018 = pd.DataFrame(country_2018, columns=['Country'])

country_2018 = country_2018.replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom')

country_2018 = country_2018.replace('I do not wish to disclose my location', 'Other')

country_2018 = country_2018.replace('South Korea', 'Korea')

country_2018 = country_2018.replace('United States of America','United States')

country_2018 = country_2018.replace('Iran, Islamic Republic of...','Iran')

country_2018 = country_2018.replace('Hong Kong (S.A.R.)','Hong Kong')

country_2018 = country_2018.replace('Republic of Korea','Other')





country_2018_values = multiple_choice_responses_2018.Q3.value_counts().values



countries_2018 = []

iso_alpha_2018 = []

countries_vals_2018 = []



for c,v in zip(country_2018['Country'][:-1].values,country_2018_values):

    

    iso = pycountry.countries.search_fuzzy(c)[0].alpha_3

    pop = v*232009

    

    if c !="Other":

        countries_2018.append(c)

        iso_alpha_2018.append(iso)

        countries_vals_2018.append(pop)



    

df_countries_2018 = pd.DataFrame()

df_countries_2018['country'] = countries_2018

df_countries_2018['iso_alpha'] = iso_alpha_2018

df_countries_2018['pop'] = countries_vals_2018

df_countries_2018['year'] = '2018'

# Combining 2017, 2018, 2019 years in one dataframe.



frames = [df_countries_2017, df_countries_2018, df_countries_2019]



df_countries_2017_2018_2019 = pd.concat(frames)
import plotly.express as px

fig = px.scatter_geo(df_countries_2017_2018_2019, locations="iso_alpha",

                     hover_name="country", size="pop",

                     animation_frame="year",

                     projection="natural earth")

fig.show()
# Q3.

# Country do you currently reside.



country = multiple_choice_responses_2019.Q3[1:].value_counts().index

country = pd.Series(country)

country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')

country = country.replace('United States of America','USA')

country_values = multiple_choice_responses_2019.Q3[1:].value_counts().values



plt.figure(figsize=(16, 6))

g = sns.barplot(x=country, y=country_values)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_ylabel("Number of respondens.")



for ix, x in zip(range(len(country_values)+1),country_values):

    g.text(ix,x,x, horizontalalignment='center')

    

plt.title("Country distribution in 2019 survey.")

plt.show()
# Q3.

# Country do you currently reside.



country =  multiple_choice_responses_2019.Q3[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().index

country = pd.Series(country)

country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')

country = country.replace('United States of America','USA')

country_values = multiple_choice_responses_2019.Q3[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().values







plt.figure(figsize=(16, 6))

g = sns.barplot(x=country, y=country_values)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_ylabel("Number of respondens.")



for ix, x in zip(range(len(country_values)+1),country_values):

    g.text(ix,x,x, horizontalalignment='center')

    

plt.title("Country distribution in 2019 survey.")

plt.show()
education = multiple_choice_responses_2019.Q4.value_counts()[:-1].index

education_values = multiple_choice_responses_2019.Q4.value_counts()[:-1].values



sns.set({'figure.figsize':(6,6)})

my_circle=plt.Circle( (0,0), 0.9, color='white')

plt.pie(education_values, autopct='%1.1f%%', labels=education)

p=plt.gcf()

plt.title("Education distribution in 2019.")

p.gca().add_artist(my_circle)

plt.show()
education = multiple_choice_responses_2019.Q4[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].index

education_values = multiple_choice_responses_2019.Q4[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].values



sns.set({'figure.figsize':(6,6)})

my_circle=plt.Circle( (0,0), 0.9, color='white')

plt.pie(education_values, autopct='%1.1f%%', labels=education)

p=plt.gcf()

plt.title("Education distribution in 2019.")

p.gca().add_artist(my_circle)

plt.show()
degree = multiple_choice_responses_2019.Q5.value_counts()[:-1].index

degree_values = multiple_choice_responses_2019.Q5.value_counts()[:-1].values



sns.set({'figure.figsize':(6,6)})

my_circle=plt.Circle( (0,0), 0.9, color='white')

plt.pie(degree_values, autopct='%1.1f%%', labels=degree)

p=plt.gcf()

plt.title("Degree distribution in 2019.")

p.gca().add_artist(my_circle)

plt.show()
degree = multiple_choice_responses_2019.Q5[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].index

degree_values = multiple_choice_responses_2019.Q5[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[:-1].values



sns.set({'figure.figsize':(6,6)})

my_circle=plt.Circle( (0,0), 0.9, color='white')

plt.pie(degree_values, autopct='%1.1f%%', labels=degree)

p=plt.gcf()

plt.title("Degree distribution in 2019.")

p.gca().add_artist(my_circle)

plt.show()
company_order = ['0-49 employees','50-249 employees','250-999 employees','1000-9,999 employees','> 10,000 employees']

company_index = multiple_choice_responses_2019.Q6.value_counts()[company_order].index

company_values = multiple_choice_responses_2019.Q6.value_counts()[company_order].values

company_values_SQL = multiple_choice_responses_2019.Q6[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[company_order].values



company_vals_prc = company_values/sum(company_values)

company_vals_SQL_prc = company_values_SQL/sum(company_values_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=company_order, y=company_vals_prc),

    go.Bar(name='2019 data SQL', x=company_order, y=company_vals_SQL_prc),

    

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text='The size of the company where you work',yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
numb_of_people_order = ['0','1-2','3-4','5-9','10-14','15-19','20+']

numb_of_people_index = multiple_choice_responses_2019.Q7.value_counts()[numb_of_people_order].index

numb_of_people_values = multiple_choice_responses_2019.Q7.value_counts()[numb_of_people_order].values

numb_of_people_values_SQL = multiple_choice_responses_2019.Q7[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[numb_of_people_order].values



numb_of_people_vals_prc = numb_of_people_values/sum(numb_of_people_values)

numb_of_people_vals_SQL_prc = numb_of_people_values_SQL/sum(numb_of_people_values_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=numb_of_people_order, y=numb_of_people_vals_prc),

    go.Bar(name='2019 data SQL', x=numb_of_people_order, y=numb_of_people_vals_SQL_prc),

    

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text='Aproximate number of individuals are responsible for data science workloads',yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
labeling_current_ML_incorporation = multiple_choice_responses_2019.Q8.value_counts()[:-1].index

values_current_ML_incorporation = multiple_choice_responses_2019.Q8.value_counts().values

values_current_ML_incorporation_SQL =  multiple_choice_responses_2019.Q8[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts().values

values_current_ML_incorporation_prc = values_current_ML_incorporation/sum(values_current_ML_incorporation)

values_current_ML_incorporation_SQL_prc = values_current_ML_incorporation_SQL/sum(values_current_ML_incorporation_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=labeling_current_ML_incorporation, y=values_current_ML_incorporation_prc),

    go.Bar(name='2019 data SQL', x=labeling_current_ML_incorporation, y=values_current_ML_incorporation_SQL_prc),

    

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text='Employer incorporate machine learning methods into their business',yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
Q9cols = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']



list_number_important_activities = []

list_important_activities = []



for col in Q9cols:

    important_activities = multiple_choice_responses_2019[col].value_counts().index[0] 

    number_important_activities = multiple_choice_responses_2019[col].value_counts()[0] 

    list_number_important_activities.append(number_important_activities)

    list_important_activities.append(important_activities)



df = pd.DataFrame(list_number_important_activities)

df = df.T

df.columns = list_important_activities



labels = list_important_activities

sizes = list_number_important_activities

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=False, startangle=140)

ax1.axis('equal')

plt.title("Your role at work in 2019 survey.")

plt.show()
Q9cols = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']



list_number_important_activities_SQL = []

list_important_activities = []



for col in Q9cols:

    important_activities = multiple_choice_responses_2019[col].value_counts().index[0] 

    number_important_activities_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    list_number_important_activities_SQL.append(number_important_activities_SQL)

    list_important_activities.append(important_activities)



df = pd.DataFrame(list_number_important_activities_SQL)

df = df.T

df.columns = list_important_activities



labels = list_important_activities

sizes = list_number_important_activities_SQL

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=False, startangle=140)

ax1.axis('equal')

plt.title("Your role at for SQL user")

plt.show()


multiple_choice_responses_2019.Q10.value_counts().index



compensation_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',

                      '5,000-7,499', '7,500-9,999','10,000-14,999','15,000-19,999',  

                      '20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999', 

                      '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999',

                      '90,000-99,999','100,000-124,999','125,000-149,999', 

                      '150,000-199,999','200,000-249,999','250,000-299,999', '300,000-500,000', '> $500,000']



values_compensation = multiple_choice_responses_2019.Q10.value_counts()[compensation_order].values

values_compensation_SQL = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[compensation_order].values





values_compensation_prc = values_compensation/sum(values_compensation)

values_compensation_SQL_prc = values_compensation_SQL/sum(values_compensation_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=compensation_order, y=values_compensation_prc),

    go.Bar(name='2019 data SQL', x=compensation_order, y=values_compensation_SQL_prc),

    

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Yearly compensation (approximate $USD) in 2019",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()



money_spent_order = ['$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999','> $100,000 ($USD)']

money_spent_order_formated_x = ['$0 (USD)', '$1-99', '$100-999', '$1000-9,999', '$10,000-99,999','> $100,000']

values_money_spent = multiple_choice_responses_2019.Q11.value_counts()[money_spent_order].values

values_money_spent_SQL = multiple_choice_responses_2019.Q11[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[money_spent_order].values



values_money_spent_prc = values_money_spent/sum(values_money_spent)

values_money_spent_SQL_prc = values_money_spent_SQL/sum(values_money_spent_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=money_spent_order_formated_x, y=values_money_spent_prc),

    go.Bar(name='2019 data SQL', x=money_spent_order_formated_x, y=values_money_spent_SQL_prc),

    

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text=" Approximately money have you spent on machine learning and/or cloud computing products at your work",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()

Q12cols = ['Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']



list_number_of_media_sources = []

list_of_media_sources = []

list_number_of_media_sources_SQL = []



for col in Q12cols:

    media_source = multiple_choice_responses_2019[col].value_counts().index[0] 

    number_media_sources = multiple_choice_responses_2019[col].value_counts()[0]

    number_media_sources_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]

    

    list_number_of_media_sources.append(number_media_sources)

    list_of_media_sources.append(media_source)

    list_number_of_media_sources_SQL.append(number_media_sources_SQL)

    

    

list_number_of_media_sources_prc = list_number_of_media_sources/sum(list_number_of_media_sources)

list_number_of_media_sources_SQL_prc =list_number_of_media_sources_SQL/sum(list_number_of_media_sources_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_of_media_sources, y=list_number_of_media_sources_prc),

    go.Bar(name='2019 data SQL', x=list_of_media_sources, y=list_number_of_media_sources_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Your favorite media sources that report on data science topics",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

Q13cols = ['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']



list_science_courses = []

list_number_science_courses = []

list_number_science_courses_SQL = []



for col in Q13cols:

    science_course = multiple_choice_responses_2019[col].value_counts().index[0] 

    number_science_courses = multiple_choice_responses_2019[col].value_counts()[0] 

    number_science_courses_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]

    list_number_science_courses.append(number_science_courses)

    list_science_courses.append(science_course)

    list_number_science_courses_SQL.append(number_science_courses_SQL)

    

    

list_number_science_courses_prc = list_number_science_courses/sum(list_number_science_courses)

list_number_science_courses_SQL_prc = list_number_science_courses_SQL/sum(list_number_science_courses_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_science_courses, y=list_number_science_courses_prc),

    go.Bar(name='2019 data SQL', x=list_science_courses, y=list_number_science_courses_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Platforms that you have begun or completed data science courses",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

tools = multiple_choice_responses_2019.Q14.value_counts().index[:-1]

tools_number = multiple_choice_responses_2019.Q14.value_counts()[tools].values

tools_number_sql = multiple_choice_responses_2019.Q14[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[tools].values



tools_number_prc = tools_number/sum(tools_number)

tools_number_SQL_prc = tools_number_sql/sum(tools_number_sql)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=tools, y=tools_number_prc),

    go.Bar(name='2019 data SQL', x=tools, y=tools_number_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="The primary tool that you use at work or school to analyze data",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

code_writing_time = ['I have never written code','< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']

numb_code_writing_time = multiple_choice_responses_2019.Q15.value_counts()[code_writing_time].values

numb_code_writing_time_SQL = multiple_choice_responses_2019.Q15[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[code_writing_time].values



#print(numb_code_writing_time_SQL[1:])

code_writing_number_prc = numb_code_writing_time/sum(numb_code_writing_time)

code_writing_number_SQL_prc = numb_code_writing_time_SQL[1:]/sum(numb_code_writing_time_SQL[1:])



code_corrected_number_SQL = np.insert(code_writing_number_SQL_prc,0,0)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=code_writing_time, y=code_writing_number_prc),

    go.Bar(name='2019 data SQL', x=code_writing_time, y=code_corrected_number_SQL),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Time spent writing code to analyze data",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q16cols = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']



list_of_IDE_numbers = []

list_of_IDE = []

list_of_IDE_numbers_SQL = []





for col in Q16cols:

    IDE = multiple_choice_responses_2019[col].value_counts().index[0]

    number_of_IDE = multiple_choice_responses_2019[col].value_counts()[0]

    number_of_IDE_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]

  

    

    list_of_IDE_numbers.append(number_of_IDE)

    list_of_IDE.append(IDE)

    list_of_IDE_numbers_SQL.append(number_of_IDE_SQL)

    

list_of_IDE_numbers_prc = list_of_IDE_numbers/sum(list_of_IDE_numbers)

list_of_IDE_numbers_SQL_prc = list_of_IDE_numbers_SQL / sum(list_of_IDE_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_of_IDE, y=list_of_IDE_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_of_IDE, y=list_of_IDE_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="IDE you are using for regural basis",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q17cols = ['Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12']



list_of_hnotebooks_numbers = []

list_of_hnotebooks = []

list_of_hnotebooks_numbers_SQL = []



for col in Q17cols:

    hnotebooks = multiple_choice_responses_2019[col].value_counts().index[0] 

    number_hnotebooks = multiple_choice_responses_2019[col].value_counts()[0] 

    number_hnotebooks_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_of_hnotebooks_numbers.append(number_hnotebooks)

    list_of_hnotebooks.append(hnotebooks)

    list_of_hnotebooks_numbers_SQL.append(number_hnotebooks_SQL)

    

list_of_hnotebooks_numbers_prc = list_of_hnotebooks_numbers/sum(list_of_hnotebooks_numbers)

list_of_hnotebooks_numbers_SQL_prc = list_of_hnotebooks_numbers_SQL / sum(list_of_hnotebooks_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_of_hnotebooks, y=list_of_hnotebooks_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_of_hnotebooks, y=list_of_hnotebooks_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Following hosted notebook products do you use on a regular basis ",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q18cols = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']



list_number_of_users = []

list_of_programming_languages = []

for col in Q18cols:

    programming_language = multiple_choice_responses_2019[col].value_counts().index[0] 

    number_of_users = multiple_choice_responses_2019[col].value_counts()[0] 

    list_number_of_users.append(number_of_users)

    list_of_programming_languages.append(programming_language)

    

g = sns.barplot(x=list_of_programming_languages, y=list_number_of_users)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title("Popularity of programming languages in 2019 survey")

g.set_ylabel("Number of respondents.")



for ix, x in zip(range(len(list_number_of_users)+1),list_number_of_users):

    g.text(ix,x,x, horizontalalignment='center')



plt.show()
languages = multiple_choice_responses_2019.Q19.value_counts().index[:-1]

languages_numbers = multiple_choice_responses_2019.Q19.value_counts().values[:-1]



g = sns.barplot(x=languages, y=languages_numbers)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.title("Programming language would you recommend an aspiring data scientist to learn first in 2019.")

g.set_ylabel("Number of respondents.")







for ix, x in zip(range(len(languages_numbers)+1),languages_numbers):

    g.text(ix,x,x, horizontalalignment='center')



plt.show()
Q20cols = ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']



list_viz_tools = []

list_viz_tools_numbers = []

list_viz_tools_numbers_SQL = []



for col in Q20cols:

    viz_tools = multiple_choice_responses_2019[col].value_counts().index[0] 

    viz_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    viz_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    list_viz_tools.append(viz_tools)

    list_viz_tools_numbers.append(viz_tools_numbers)

    list_viz_tools_numbers_SQL.append(viz_tools_numbers_SQL)

    

    

list_of_viz_tools_numbers_prc = list_viz_tools_numbers/sum(list_viz_tools_numbers)

list_of_viz_tools_numbers_SQL_prc = list_viz_tools_numbers_SQL / sum(list_viz_tools_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_viz_tools, y=list_of_viz_tools_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_viz_tools, y=list_of_viz_tools_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Visualization libraries or tools do you use on a regular basis ",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

    

Q20cols = ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']



list_viz_tools = []

list_viz_tools_numbers = []

list_viz_tools_numbers_SQL = []



for col in Q20cols:

    viz_tools = multiple_choice_responses_2019[col].value_counts().index[0] 

    viz_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    viz_tools_numbers_SQL = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_3 == "SQL") & (multiple_choice_responses_2019.Q18_Part_1 != "Python")].value_counts()[0] 

    list_viz_tools.append(viz_tools)

    list_viz_tools_numbers.append(viz_tools_numbers)

    list_viz_tools_numbers_SQL.append(viz_tools_numbers_SQL)

    

    

list_of_viz_tools_numbers_prc = list_viz_tools_numbers/sum(list_viz_tools_numbers)

list_of_viz_tools_numbers_SQL_prc = list_viz_tools_numbers_SQL / sum(list_viz_tools_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_viz_tools, y=list_of_viz_tools_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_viz_tools, y=list_of_viz_tools_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Visualization libraries or tools do you use on a regular basis ",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q21cols = ['Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5']



list_hardware_tools = []

list_hardware_tools_numbers = []

list_hardware_tools_numbers_SQL = []





for col in Q21cols:

    

    hardware_tools = multiple_choice_responses_2019[col].value_counts().index[0] 

    hardware_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    hardware_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_hardware_tools.append(hardware_tools)

    list_hardware_tools_numbers.append(hardware_tools_numbers)

    list_hardware_tools_numbers_SQL.append(hardware_tools_numbers_SQL)



list_hardware_tools_numbers_prc = list_hardware_tools_numbers/sum(list_hardware_tools_numbers)

list_hardware_tools_numbers_SQL_prc = list_hardware_tools_numbers_SQL / sum(list_hardware_tools_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_hardware_tools, y=list_hardware_tools_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_hardware_tools, y=list_hardware_tools_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Types of specialized hardware do you use on a regular basis",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()



TPU_usage =  ['Never', 'Once', '2-5 times', '6-24 times', '> 25 times']

TPU_usage_numbers = multiple_choice_responses_2019.Q22.value_counts()[TPU_usage].values

TPU_usage_numbers_SQL = multiple_choice_responses_2019.Q22[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[TPU_usage].values





TPU_usage_numbers_prc = TPU_usage_numbers / sum(TPU_usage_numbers)

TPU_usage_numbers_SQL_prc = TPU_usage_numbers_SQL / sum(TPU_usage_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=TPU_usage, y=TPU_usage_numbers_prc),

    go.Bar(name='2019 data SQL', x=TPU_usage, y=TPU_usage_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Use of a TPU (tensor processing unit)",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

Users_years =  ['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years']

Users_years_counts = multiple_choice_responses_2019.Q23.value_counts()[Users_years].values

Users_years_counts_SQL = multiple_choice_responses_2019.Q23[multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[Users_years].values



Users_years_counts_prc = Users_years_counts / sum(Users_years_counts)

Users_years_counts_SQL_prc = Users_years_counts_SQL / sum(Users_years_counts_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=Users_years, y=Users_years_counts_prc),

    go.Bar(name='2019 data SQL', x=Users_years, y=Users_years_counts_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Use of machine learning methods",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q24cols = ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']



list_ml_tools = []

list_ml_tools_numbers = []

list_ml_tools_numbers_SQL = []



for col in Q24cols:

    

    ml_tools = multiple_choice_responses_2019[col].value_counts().index[0] 

    ml_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    ml_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]

    

    list_ml_tools.append(ml_tools)

    list_ml_tools_numbers.append(ml_tools_numbers)

    list_ml_tools_numbers_SQL.append(ml_tools_numbers_SQL)



list_ml_tools_prc = list_ml_tools_numbers / sum(list_ml_tools_numbers)

list_ml_tools_SQL_prc = list_ml_tools_numbers_SQL / sum(list_ml_tools_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_ml_tools, y=list_ml_tools_prc),

    go.Bar(name='2019 data SQL', x=list_ml_tools, y=list_ml_tools_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="ML algorithms do you use on a regular basis",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

Q25cols = ['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']



list_ml_tools_reg = []

list_ml_tools_numbers_reg = []

list_ml_tools_numbers_reg_SQL = []



for col in Q25cols:

    

    ml_tools_reg = multiple_choice_responses_2019[col].value_counts().index[0] 

    ml_tools_numbers_reg = multiple_choice_responses_2019[col].value_counts()[0] 

    ml_tools_numbers_reg_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    

    list_ml_tools_reg.append(ml_tools_reg)

    list_ml_tools_numbers_reg.append(ml_tools_numbers_reg)

    list_ml_tools_numbers_reg_SQL.append(ml_tools_numbers_reg_SQL)

    



list_ml_tools_prc = list_ml_tools_numbers_reg / sum(list_ml_tools_numbers_reg)

list_ml_tools_SQL_prc = list_ml_tools_numbers_reg_SQL / sum(list_ml_tools_numbers_reg_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_ml_tools_reg, y=list_ml_tools_prc),

    go.Bar(name='2019 data SQL', x=list_ml_tools_reg, y=list_ml_tools_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="ML algorithms do you use on a regular basis",yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q26cols = ['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']



list_cv_tools = []

list_cv_tools_numbers = []

list_cv_tools_numbers_SQL = []



for col in Q26cols:

    cv_tools= multiple_choice_responses_2019[col].value_counts().index[0] 

    cv_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    cv_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_cv_tools.append(cv_tools)

    list_cv_tools_numbers.append(cv_tools_numbers)

    list_cv_tools_numbers_SQL.append(cv_tools_numbers_SQL)



list_cv_tools_prc = list_cv_tools_numbers / sum(list_cv_tools_numbers)

list_cv_tools_SQL_prc = list_cv_tools_numbers_SQL / sum(list_cv_tools_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_cv_tools, y=list_cv_tools_prc),

    go.Bar(name='2019 data SQL', x=list_cv_tools, y=list_cv_tools_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Categories of computer vision methods do you use on a regular basis",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

Q27cols = ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']



list_nlp_tools = []

list_nlp_tools_numbers = []

list_nlp_tools_numbers_SQL = []



for col in Q27cols:

    

    nlp_tools= multiple_choice_responses_2019[col].value_counts().index[0] 

    nlp_tools_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    nlp_tools_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_nlp_tools.append(nlp_tools)

    list_nlp_tools_numbers.append(nlp_tools_numbers)

    list_nlp_tools_numbers_SQL.append(nlp_tools_numbers_SQL)





list_nlp_tools_prc = list_nlp_tools_numbers / sum(list_nlp_tools_numbers)

list_nlp_tools_SQL_prc = list_nlp_tools_numbers_SQL / sum(list_nlp_tools_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_nlp_tools, y=list_nlp_tools_prc),

    go.Bar(name='2019 data SQL', x=list_nlp_tools, y=list_nlp_tools_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="The following natural language processing (NLP) methods do you use on a regular basis",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()
Q28cols = ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']



list_ml_frs = []

list_ml_frs_numbers = []

list_ml_frs_numbers_SQL = []





for col in Q28cols:

    

    ml_frs = multiple_choice_responses_2019[col].value_counts().index[0] 

    ml_frs_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    ml_frs_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0]

    

    list_ml_frs.append(ml_frs)

    list_ml_frs_numbers.append(ml_frs_numbers)

    list_ml_frs_numbers_SQL.append(ml_frs_numbers_SQL)

    

list_frs_numbers_prc = list_ml_frs_numbers / sum(list_ml_frs_numbers)

list_frs_numbers_SQL_prc = list_ml_frs_numbers_SQL / sum(list_ml_frs_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_ml_frs, y=list_frs_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_ml_frs, y=list_frs_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Machine learning frameworks do you use on a regular basis",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

Q29cols = ['Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12']



list_cps = []

list_cps_numbers = []

list_cps_numbers_SQL = []



for col in Q29cols:

    

    cps = multiple_choice_responses_2019[col].value_counts().index[0] 

    cps_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    cps_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_cps.append(cps)

    list_cps_numbers.append(cps_numbers)

    list_cps_numbers_SQL.append(cps_numbers_SQL)

    

list_cps_numbers_prc = list_cps_numbers / sum(list_cps_numbers)

list_cps_numbers_SQL_prc = list_cps_numbers_SQL / sum(list_cps_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_cps, y=list_cps_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_cps, y=list_cps_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Computer platforms you use on regular basis ",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()

    

Q30cols = ['Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8','Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12']



list_ccps = []

list_ccps_numbers = []

list_ccps_numbers_SQL = []





for col in Q30cols:

    

    ccps = multiple_choice_responses_2019[col].value_counts().index[0] 

    ccps_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    ccps_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 



    

    list_ccps.append(ccps)

    list_ccps_numbers.append(ccps_numbers)

    list_ccps_numbers_SQL.append(ccps_numbers_SQL)

    

list_ccps_numbers_prc = list_ccps_numbers / sum(list_ccps_numbers)

list_ccps_numbers_SQL_prc = list_ccps_numbers_SQL / sum(list_ccps_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_ccps, y=list_ccps_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_ccps, y=list_ccps_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Specific cloud computing products do you use on a regular basis",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()    


Q31cols = ['Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8','Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12']



list_bds = []

list_bds_numbers = []

list_bds_numbers_SQL = []





for col in Q31cols:

    

    bds = multiple_choice_responses_2019[col].value_counts().index[0] 

    bds_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    bds_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    

    list_bds.append(bds)

    list_bds_numbers.append(bds_numbers)

    list_bds_numbers_SQL.append(bds_numbers_SQL)

    

list_bds_numbers_prc = list_bds_numbers / sum(list_bds_numbers)

list_bds_numbers_SQL_prc = list_bds_numbers_SQL / sum(list_bds_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_bds, y=list_bds_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_bds, y=list_bds_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Specific big data / analytics products do you use on a regular basis",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show()    

Q32cols = ['Q32_Part_1','Q32_Part_2','Q32_Part_3','Q32_Part_4','Q32_Part_5','Q32_Part_6','Q32_Part_7','Q32_Part_8','Q32_Part_9','Q32_Part_10','Q32_Part_11','Q32_Part_12']



list_mlps = []

list_mlps_numbers = []

list_mlps_numbers_SQL = []



for col in Q32cols:

    mlps = multiple_choice_responses_2019[col].value_counts().index[0] 

    mlps_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    mlps_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_mlps.append(mlps)

    list_mlps_numbers.append(mlps_numbers)

    list_mlps_numbers_SQL.append(mlps_numbers_SQL)



list_mlps_numbers_prc = list_mlps_numbers / sum(list_mlps_numbers)

list_mlps_numbers_SQL_prc = list_mlps_numbers_SQL / sum(list_mlps_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_mlps, y=list_mlps_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_mlps, y=list_mlps_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Following machine learning products do you use on a regular basis ",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show() 

    

Q33cols = ['Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8','Q33_Part_9','Q33_Part_10','Q33_Part_11','Q33_Part_12']



list_amlts = []

list_amlts_numbers = []

list_amlts_numbers_SQL = []





for col in Q33cols:

    

    amlts = multiple_choice_responses_2019[col].value_counts().index[0] 

    amlts_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    amlts_numbers_SQL = multiple_choice_responses_2019[col][multiple_choice_responses_2019.Q18_Part_3 == "SQL"].value_counts()[0] 

    

    list_amlts.append(amlts)

    list_amlts_numbers.append(amlts_numbers)

    list_amlts_numbers_SQL.append(amlts_numbers_SQL)



list_amlts_numbers_prc = list_amlts_numbers / sum(list_amlts_numbers)

list_amlts_numbers_SQL_prc = list_amlts_numbers_SQL / sum(list_amlts_numbers_SQL)



fig = go.Figure(data=[

    go.Bar(name='2019 data', x=list_amlts, y=list_amlts_numbers_prc),

    go.Bar(name='2019 data SQL', x=list_amlts, y=list_amlts_numbers_SQL_prc),

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_layout(title_text="Automated machine learning tools (or partial AutoML tools) do you use on a regular basis",

        yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)), 

fig.show() 

multiple_choice_responses_2019.Q34_Part_1.value_counts()



Q34cols = ['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12']



list_dbs = []

list_dbs_numbers = []



for col in Q34cols:

    

    dbs = multiple_choice_responses_2019[col].value_counts().index[0] 

    dbs_numbers = multiple_choice_responses_2019[col].value_counts()[0] 

    

    list_dbs.append(dbs)

    list_dbs_numbers.append(dbs_numbers)

    

g = sns.barplot(x=list_dbs, y=list_dbs_numbers)

g.set_xticklabels(g.get_xticklabels(), rotation=80)

plt.title("The following relational database products do you use on a regular basis in 2019")

g.set_ylabel('Number of respondents.')



for ix, x in zip(range(len(list_dbs_numbers)+1),list_dbs_numbers):

    g.text(ix,x,x, horizontalalignment='center')



plt.show()
junior_ml_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()[:5]

mid_ml_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()[:5]

senior_ml_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()[:5]



plt.figure(3, figsize=(20,5))

the_grid = GridSpec(1, 3)



# Junior Developers are from these top 5 countries.



plt.subplot(the_grid[0, 0])

g = sns.barplot(x=junior_ml_devs.index, y=junior_ml_devs.values)

g.set_xticklabels(junior_ml_devs.index, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Countries where junior ML developers come from in 2019 survey.")





# Middle developers are from these top 5 countries.



plt.subplot(the_grid[0, 1])

g = sns.barplot(x=mid_ml_devs.index, y=mid_ml_devs.values)

g.set_xticklabels(mid_ml_devs.index, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Countries where middle-level ML developers come from in 2019 survey.")





# Senior developers are from these top 5 countries.



plt.subplot(the_grid[0, 2])



g = sns.barplot(x=senior_ml_devs.index, y=senior_ml_devs.values)

g.set_xticklabels(senior_ml_devs.index, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Countries where senior ML developer come from in 2019 survey.")



plt.show()
Q18cols = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']



l_junior_ml_dev_lang = []

l_junior_ml_dev_lang_num = [] 



for col in Q18cols:

 

    junior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()

    junior_ml_dev_lang = junior_ml_devs.index[0]

    junior_ml_dev_lang_num = junior_ml_devs.values[0]

    

    l_junior_ml_dev_lang.append(junior_ml_dev_lang)

    l_junior_ml_dev_lang_num.append(junior_ml_dev_lang_num)



    

l_mid_ml_dev_lang = []

l_mid_ml_dev_lang_num = [] 



for col in Q18cols:

    

    mid_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()



    mid_ml_dev_lang = mid_ml_devs.index[0]  

    mid_ml_dev_lang_num = mid_ml_devs.values[0]

    

    l_mid_ml_dev_lang.append(mid_ml_dev_lang)

    l_mid_ml_dev_lang_num.append(mid_ml_dev_lang_num)



l_senior_ml_dev_lang = []

l_senior_ml_dev_lang_num = [] 





for col in Q18cols:



    senior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()



    senior_ml_dev_lang = senior_ml_devs.index[0]  

    senior_ml_dev_lang_num = senior_ml_devs.values[0]

    

    l_senior_ml_dev_lang.append(senior_ml_dev_lang)

    l_senior_ml_dev_lang_num.append(senior_ml_dev_lang_num)



    

plt.figure(3, figsize=(20,5))

the_grid = GridSpec(1, 3)



# Junior Developers.



plt.subplot(the_grid[0, 0])

g = sns.barplot(x=l_junior_ml_dev_lang, y=l_junior_ml_dev_lang_num)

g.set_xticklabels(l_junior_ml_dev_lang, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Programming languages junior ML developers use in 2019 survey.")





# Middle-level developers.



plt.subplot(the_grid[0, 1])

g = sns.barplot(x=l_mid_ml_dev_lang, y=l_mid_ml_dev_lang_num)

g.set_xticklabels(l_mid_ml_dev_lang, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Programming languages middle-level ML developers use in 2019 survey")





# Senior developers.



plt.subplot(the_grid[0, 2])



g = sns.barplot(x=l_senior_ml_dev_lang, y=l_senior_ml_dev_lang_num)

g.set_xticklabels(l_senior_ml_dev_lang, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Programming languages senior ML developers use in 2019 survey")



plt.show()
Q16cols = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']



l_junior_ml_dev_ide = []

l_junior_ml_dev_ide_num = [] 



for col in Q16cols:

 

    junior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()

    junior_ml_dev_ide = junior_ml_devs.index[0]

    junior_ml_dev_ide_num = junior_ml_devs.values[0]

    

    l_junior_ml_dev_ide.append(junior_ml_dev_ide)

    l_junior_ml_dev_ide_num.append(junior_ml_dev_ide_num)



    

l_mid_ml_dev_ide = []

l_mid_ml_dev_ide_num = [] 



for col in Q16cols:

    

    mid_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()



    mid_ml_dev_ide = mid_ml_devs.index[0]  

    mid_ml_dev_ide_num = mid_ml_devs.values[0]

    

    l_mid_ml_dev_ide.append(mid_ml_dev_ide)

    l_mid_ml_dev_ide_num.append(mid_ml_dev_ide_num)



l_senior_ml_dev_ide = []

l_senior_ml_dev_ide_num = [] 





for col in Q16cols:



    senior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()



    senior_ml_dev_ide = senior_ml_devs.index[0]  

    senior_ml_dev_ide_num = senior_ml_devs.values[0]

    

    l_senior_ml_dev_ide.append(senior_ml_dev_ide)

    l_senior_ml_dev_ide_num.append(senior_ml_dev_ide_num)



    

plt.figure(3, figsize=(20,5))

the_grid = GridSpec(1, 3)



# Junior Developers.



plt.subplot(the_grid[0, 0])

g = sns.barplot(x=l_junior_ml_dev_ide, y=l_junior_ml_dev_ide_num)

g.set_xticklabels(l_junior_ml_dev_ide, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("IDEs junior ML developers use in 2019 survey.")





# Middle-level developers.



plt.subplot(the_grid[0, 1])

g = sns.barplot(x=l_mid_ml_dev_ide, y=l_mid_ml_dev_ide_num)

g.set_xticklabels(l_mid_ml_dev_ide, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("IDEs middle-level ML developers use in 2019 survey")





# Senior developers.



plt.subplot(the_grid[0, 2])



g = sns.barplot(x=l_senior_ml_dev_ide, y=l_senior_ml_dev_ide_num)

g.set_xticklabels(l_senior_ml_dev_ide, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("IDEs senior ML developers use in 2019 survey")



plt.show()

Q34cols = ['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12']



l_junior_ml_dev_db = []

l_junior_ml_dev_db_num = [] 



for col in Q34cols:

 

    junior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()

    junior_ml_dev_db = junior_ml_devs.index[0]

    junior_ml_dev_db_num = junior_ml_devs.values[0]

    

    l_junior_ml_dev_db.append(junior_ml_dev_db)

    l_junior_ml_dev_db_num.append(junior_ml_dev_db_num)



    

l_mid_ml_dev_db = []

l_mid_ml_dev_db_num = [] 



for col in Q34cols:

    

    mid_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()



    mid_ml_dev_db = mid_ml_devs.index[0]  

    mid_ml_dev_db_num = mid_ml_devs.values[0]

    

    l_mid_ml_dev_db.append(mid_ml_dev_db)

    l_mid_ml_dev_db_num.append(mid_ml_dev_db_num)



l_senior_ml_dev_db = []

l_senior_ml_dev_db_num = [] 





for col in Q34cols:



    senior_ml_devs = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()



    senior_ml_dev_db = senior_ml_devs.index[0]  

    senior_ml_dev_db_num = senior_ml_devs.values[0]

    

    l_senior_ml_dev_db.append(senior_ml_dev_db)

    l_senior_ml_dev_db_num.append(senior_ml_dev_db_num)



    

plt.figure(3, figsize=(20,5))

the_grid = GridSpec(1, 3)



# Junior Developers.



plt.subplot(the_grid[0, 0])

g = sns.barplot(x=l_junior_ml_dev_db, y=l_junior_ml_dev_db_num)

g.set_xticklabels(l_junior_ml_dev_db, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Relational database popular among junior ML developers")





# Middle-level developers.



plt.subplot(the_grid[0, 1])

g = sns.barplot(x=l_mid_ml_dev_db, y=l_mid_ml_dev_db_num)

g.set_xticklabels(l_mid_ml_dev_db, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Relational database popular among middle-level ML developers")





# Senior developers.



plt.subplot(the_grid[0, 2])



g = sns.barplot(x=l_senior_ml_dev_db, y=l_senior_ml_dev_db_num)

g.set_xticklabels(l_senior_ml_dev_db, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Relational database popular among senior ML developers")



plt.show()

compensation_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',

                      '5,000-7,499', '7,500-9,999','10,000-14,999','15,000-19,999',  

                      '20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999', 

                      '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999',

                      '90,000-99,999','100,000-124,999','125,000-149,999', 

                      '150,000-199,999','200,000-249,999','250,000-299,999', '300,000-500,000', '> $500,000']

values_compensation = multiple_choice_responses_2019.Q10.value_counts()[compensation_order].values



 

junior_ml_devs = multiple_choice_responses_2019.Q10[(multiple_choice_responses_2019.Q23 == '< 1 years') | (multiple_choice_responses_2019.Q23 == '1-2 years')| (multiple_choice_responses_2019.Q23 ==  '2-3 years')].value_counts()[compensation_order].values

mid_ml_devs = multiple_choice_responses_2019.Q10[(multiple_choice_responses_2019.Q23 == '3-4 years') | (multiple_choice_responses_2019.Q23 == '4-5 years')].value_counts()[compensation_order].values

senior_ml_devs = multiple_choice_responses_2019.Q10[(multiple_choice_responses_2019.Q23 == '5-10 years') | (multiple_choice_responses_2019.Q23 == '10-15 years') | (multiple_choice_responses_2019.Q23 == '20+ years') ].value_counts()[compensation_order].values



    

plt.figure(3, figsize=(20,5))

the_grid = GridSpec(1, 3)



# Junior Developers.



plt.subplot(the_grid[0, 0])

g = sns.barplot(x=compensation_order, y=junior_ml_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for junior developers in 2019 survey.")





# Middle-level developers.



plt.subplot(the_grid[0, 1])

g = sns.barplot(x=compensation_order, y=mid_ml_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for middle-level developers in 2019 survey")





# Senior developers.



plt.subplot(the_grid[0, 2])



g = sns.barplot(x=compensation_order, y=senior_ml_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for senior developers in 2019 survey")



plt.show()

compensation_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',

                      '5,000-7,499', '7,500-9,999','10,000-14,999','15,000-19,999',  

                      '20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999', 

                      '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999',

                      '90,000-99,999','100,000-124,999','125,000-149,999', 

                      '150,000-199,999','200,000-249,999','250,000-299,999', '300,000-500,000', '> $500,000']



python_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_1 == 'Python'].value_counts()[compensation_order].values

SQL_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_3 == 'SQL'].value_counts()[compensation_order].values

R_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_2 == 'R' ].value_counts()[compensation_order].values

Java_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_6 == 'Java' ].value_counts()[compensation_order].values

Cpp_devs = multiple_choice_responses_2019.Q10[multiple_choice_responses_2019.Q18_Part_5 == 'C++' ].value_counts()[compensation_order].values



plt.figure(2, figsize=(12,20))

the_grid = GridSpec(3, 2, hspace=0.5)



# Python Developers.



plt.subplot(the_grid[0, 0])

g = sns.barplot(x=compensation_order, y=python_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for Python developers in 2019 survey.")





# SQL developers.



plt.subplot(the_grid[0, 1])

g = sns.barplot(x=compensation_order, y=SQL_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for SQL developers in 2019 survey")





# R developers.



plt.subplot(the_grid[1, 0])



g = sns.barplot(x=compensation_order, y=R_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for R developers in 2019 survey")



# Java developers.



plt.subplot(the_grid[1, 1])

g = sns.barplot(x=compensation_order, y=Java_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for Java developers in 2019 survey")





# C++ developers.



plt.subplot(the_grid[2, 0])



g = sns.barplot(x=compensation_order, y=Cpp_devs)

g.set_xticklabels(compensation_order, rotation=80)

g.set_ylabel("Number of respondents")

plt.title("Yearly compensation for C++ developers in 2019 survey")





plt.show()

low_python_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q18_Part_1 == 'Python') &

                                                   (multiple_choice_responses_2019.Q10 == '$0-999') | (multiple_choice_responses_2019.Q10 == '1000-1999')].value_counts()



high_python_devs = multiple_choice_responses_2019.Q3[(multiple_choice_responses_2019.Q18_Part_1 == 'Python') &

                                                   (multiple_choice_responses_2019.Q10 == '100,000-124,999') | (multiple_choice_responses_2019.Q10 == '125,000-149,999')].value_counts()

 



country = low_python_devs.index

country = pd.Series(country)

country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')

country = country.replace('United States of America','USA')

country_values = low_python_devs.values



plt.figure(figsize=(16, 6))

g = sns.barplot(x=country, y=country_values)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_ylabel("Number of respondens.")



for ix, x in zip(range(len(country_values)+1),country_values):

    g.text(ix,x,x, horizontalalignment='center')

    

plt.title("Low pay '$0-999' or '1000-1999' country distribution in 2019 survey.")

plt.show()
country = high_python_devs.index

country = pd.Series(country)

country = country.replace('United Kingdom of Great Britain and Northern Ireland','UK')

country = country.replace('United States of America','USA')

country_values = high_python_devs.values



plt.figure(figsize=(16, 6))

g = sns.barplot(x=country, y=country_values)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_ylabel("Number of respondens.")



for ix, x in zip(range(len(country_values)+1),country_values):

    g.text(ix,x,x, horizontalalignment='center')

    

plt.title("High pay '100,000-124,999' or '125,000-149,999' country distribution in 2019 survey.")

plt.show()
Q12cols = ['Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']



list_number_of_media_sources = []

list_of_media_sources = []



for col in Q12cols:

    media_source = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts().index[0] 

    number_media_sources = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts()[0] 

    list_number_of_media_sources.append(number_media_sources)

    list_of_media_sources.append(media_source)



fig, ax = plt.subplots() 

    

ax.barh(list_of_media_sources, list_number_of_media_sources, align='center', color=(0.6, 0.4, 0.6, 0.6))

ax.set_yticklabels(list_of_media_sources)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of respondents.')

ax.set_title('Pythonistas favorite media sources that report on data science topics.')

for i, v in enumerate(list_number_of_media_sources):

    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')



plt.show()
Q24cols = ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']



list_ml_tools = []

list_ml_tools_numbers = []



for col in Q24cols:

    ml_tools = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts().index[0] 

    ml_tools_numbers = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts()[0] 

    list_ml_tools.append(ml_tools)

    list_ml_tools_numbers.append(ml_tools_numbers)



fig, ax = plt.subplots()



ax.barh(list_ml_tools, list_ml_tools_numbers, align='center', color=(0.6, 0.4, 0.6, 0.6))

ax.set_yticklabels(list_ml_tools)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of respondents.')

ax.set_title('ML algorithms Pythonistas use on a regular basis in 2019.')

for i, v in enumerate(list_ml_tools_numbers):

    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')



plt.show()
Q13cols = ['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']



list_science_courses = []

list_number_science_courses = []



for col in Q13cols:

    science_course = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts().index[0] 

    number_science_courses = multiple_choice_responses_2019[col][(multiple_choice_responses_2019.Q18_Part_1 == 'Python')].value_counts()[0] 

    list_number_science_courses.append(number_science_courses)

    list_science_courses.append(science_course)



fig, ax = plt.subplots()



ax.barh(list_science_courses, list_number_science_courses, align='center', color=(0.6, 0.4, 0.6, 0.6))

ax.set_yticklabels(list_science_courses)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of respondents.')

ax.set_title('Platforms that Pythonistas have begun or completed data science courses in 2019 survey.')

for i, v in enumerate(list_number_science_courses):

    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')



plt.show()