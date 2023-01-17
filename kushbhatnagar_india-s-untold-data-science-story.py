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
#Importing the required libraries

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px
#Importing all data set

dataset_mcq=pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")
#Removing the header 

dataset_mcq.columns=dataset_mcq.iloc[0]

#Removing the first row

dataset_mcq=dataset_mcq.drop([0])
#First few rows

dataset_mcq.head()
#Total number of records

dataset_mcq.shape
#Gender wise distribution

fig = go.Figure(data=[go.Pie(labels=dataset_mcq['What is your gender? - Selected Choice'],hole=.4)])

fig.show()
# Replacing the ambigious countries name with Standard names

dataset_mcq['In which country do you currently reside?'].replace(

                                                   {'United States of America':'USA',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'UK',

                                                    "Hong Kong (S.A.R.)":"HongKong"},inplace=True)

# Replacing the long name in education level with abbrevations

dataset_mcq['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].replace(

                                                   {"Some college/university study without earning a bachelor’s degree":'Some Education'},inplace=True)

#Country wise distribution of Respondant

country_dist=dataset_mcq['In which country do you currently reside?'].value_counts()

fig = px.choropleth(country_dist.values, #Input DataFrame

                    locations=country_dist.index, #DataFrame column with locations

                    locationmode='country names', # DataFrame column with color values

                    color=country_dist.values, # Set to plot

                    color_continuous_scale="haline")

fig.update_layout(title="Countrywise Distribution of Respondant")

fig.show()
#Country wise distribution

fig = go.Figure(data=[go.Pie(labels=dataset_mcq['In which country do you currently reside?'],hole=.3)])

fig.show()
#Taking male and female count seprately

male_count=dataset_mcq[dataset_mcq['What is your gender? - Selected Choice'] == 'Male']

female_count=dataset_mcq[dataset_mcq['What is your gender? - Selected Choice'] == 'Female']



# Top-10 Countries with Respondents 

male_count_top10=male_count['In which country do you currently reside?'].value_counts()[:10].reset_index()

female_count_top10=female_count['In which country do you currently reside?'].value_counts()[:10].reset_index()



# Pie chart to depict male and female respondant country wise

pieMen=go.Figure(data=[go.Pie(labels=male_count_top10['index'],values=male_count_top10['In which country do you currently reside?'],name="Men",hole=.3)])

pieWomen=go.Figure(data=[go.Pie(labels=female_count_top10['index'],values=female_count_top10['In which country do you currently reside?'],name="Women",hole=.3)])
pieMen.show()
pieWomen.show()
# Respondant's GenderWise Education level

male_educationlevel=male_count['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().reset_index()

female_educationlevel=female_count['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().reset_index()



# Pie chart to depict male and female respondant country wise

pieMenEducation=go.Figure(data=[go.Pie(labels=male_educationlevel['index'],values=male_educationlevel['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'],name="Men",hole=.3)])

pieWomenEducation=go.Figure(data=[go.Pie(labels=female_educationlevel['index'],values=female_educationlevel['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'],name="Women",hole=.3)])

pieMenEducation.show()
pieWomenEducation.show()
# Add a gender colum to male and female education level data frames 

male_educationlevel=male_educationlevel.assign(Gender = ['Male', 'Male', 'Male', 'Male','Male','Male','Male']) 

female_educationlevel=female_educationlevel.assign(Gender = ['Female', 'Female', 'Female', 'Female','Female','Female','Female']) 

#Concat both data frame to generate comparision graph

frames1 = [male_educationlevel, female_educationlevel]

result1 = pd.concat(frames1)

result1 = result1.rename(columns = {"index": "Education Level", 

                                  "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?":"Count"}) 

#Bar graph to compare Education level for both genders

fig = px.bar(result1, x='Education Level', y='Count',color='Gender')

fig.show()
#Indian Male/Female techie's 

male_count_india=male_count[male_count['In which country do you currently reside?'] == 'India']

female_count_india=female_count[female_count['In which country do you currently reside?'] == 'India']



indian_male_educationlevel=male_count_india['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().reset_index()

indian_female_educationlevel=female_count_india['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().reset_index()



# Pie chart to depict Indian male and female respondant's

pieIndianMenEducation=go.Figure(data=[go.Pie(labels=indian_male_educationlevel['index'],values=indian_male_educationlevel['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'],name="Men",hole=.3)])

pieIndianWomenEducation=go.Figure(data=[go.Pie(labels=indian_female_educationlevel['index'],values=indian_female_educationlevel['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'],name="Women",hole=.3)])
pieIndianMenEducation.show()
pieIndianWomenEducation.show()
# Add a gender colum to male and female education level data frames 

indian_male_educationlevel=indian_male_educationlevel.assign(Gender = ['Male', 'Male', 'Male', 'Male','Male','Male','Male']) 

indian_female_educationlevel=indian_female_educationlevel.assign(Gender = ['Female', 'Female', 'Female', 'Female','Female','Female','Female']) 

#Concat both data frame to generate comparision graph

frames2 = [indian_male_educationlevel, indian_female_educationlevel]

result2 = pd.concat(frames2)

result2 = result2.rename(columns = {"index": "Education Level", 

                                  "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?":"Count"}) 

#Bar graph to compare Education level for both genders

fig = px.bar(result2, x='Education Level', y='Count',color='Gender')

fig.show()
#Taking out job role and Education level to another data frame for visualization

dataset_salary_jobrole=dataset_mcq[['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?','Select the title most similar to your current role (or most recent title if retired): - Selected Choice','What is your current yearly compensation (approximate $USD)?','What is your gender? - Selected Choice']]

dataset_salary_jobrole=dataset_salary_jobrole.rename(columns = {'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'EducationLevel','Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'JobTitle','What is your current yearly compensation (approximate $USD)?':'Salary','What is your gender? - Selected Choice':'Gender'})



#Different Job roles counts

dataset_salary_jobrole['JobTitle'].value_counts().plot.bar()
#Visualize the Job Role and Education level

fig = px.scatter(dataset_salary_jobrole,x='JobTitle',y='Salary',hover_data=['EducationLevel'])

fig.show()

#Parallel Category graph to compare all three categories

fig=px.parallel_categories(dataset_salary_jobrole)

fig.show()
#Taking out data entries related to India

dataset_salary_jobrole_india=dataset_mcq[dataset_mcq['In which country do you currently reside?'] == 'India']

#Keeping only relevant columns

dataset_salary_jobrole_india=dataset_salary_jobrole_india[['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?','Select the title most similar to your current role (or most recent title if retired): - Selected Choice','What is your current yearly compensation (approximate $USD)?','What is your gender? - Selected Choice']]

#Replacing column name to short and relevant name

dataset_salary_jobrole_india=dataset_salary_jobrole_india.rename(columns = {'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'EducationLevel','Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'JobTitle','What is your current yearly compensation (approximate $USD)?':'Salary','What is your gender? - Selected Choice':'Gender'})

#Different Job roles counts

dataset_salary_jobrole_india['JobTitle'].value_counts().plot.bar()
#Visualize the Job Role and Education level

fig = px.scatter(dataset_salary_jobrole_india,x='JobTitle',y='Salary',hover_data=['EducationLevel'])

fig.show()
#Parallel Category graph to compare all three categories

fig=px.parallel_categories(dataset_salary_jobrole_india)

fig.show()
#Taking out job role and and age to another data frame for visualization

dataset_age_jobrole=dataset_mcq[['What is your age (# years)?','Select the title most similar to your current role (or most recent title if retired): - Selected Choice','What is your current yearly compensation (approximate $USD)?','What is your gender? - Selected Choice']]

dataset_age_jobrole=dataset_age_jobrole.rename(columns = {'What is your age (# years)?':'AgeGroup','Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'JobTitle','What is your current yearly compensation (approximate $USD)?':'Salary','What is your gender? - Selected Choice':'Gender'})

#Different Job roles counts

dataset_age_jobrole['AgeGroup'].value_counts().plot.bar()
#Visualize the Age Group and Salary

fig = px.scatter(dataset_age_jobrole,x='AgeGroup',y='Salary',hover_data=['JobTitle'])

fig.show()
#Parallel Category graph to compare all three categories

fig=px.parallel_categories(dataset_age_jobrole)

fig.show()
#Taking out data entries related to India

dataset_age_jobrole_india=dataset_mcq[dataset_mcq['In which country do you currently reside?'] == 'India']

#Keeping only relevant columns

dataset_age_jobrole_india=dataset_age_jobrole_india[['What is your age (# years)?','Select the title most similar to your current role (or most recent title if retired): - Selected Choice','What is your current yearly compensation (approximate $USD)?','What is your gender? - Selected Choice']]

#Replacing column name to short and relevant name

dataset_age_jobrole_india=dataset_age_jobrole_india.rename(columns = {'What is your age (# years)?':'AgeGroup','Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'JobTitle','What is your current yearly compensation (approximate $USD)?':'Salary','What is your gender? - Selected Choice':'Gender'})

#Different Job roles counts

dataset_age_jobrole_india['AgeGroup'].value_counts().plot.bar()
#Visualize the Age Group and Salary

fig = px.scatter(dataset_age_jobrole_india,x='AgeGroup',y='Salary',hover_data=['JobTitle'])

fig.show()
#Parallel Category graph to compare all three categories

fig=px.parallel_categories(dataset_age_jobrole_india)

fig.show()
#Take out Online resources seprately and put in Data Dictonary

mediasource_count_dict = {

    'Twitter' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)'].value_counts().values[0]),

    'Hacker': (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News (https://news.ycombinator.com/)'].value_counts().values[0]),

    'Reddit' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)'].value_counts().values[0]),

    'Kaggle' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)'].value_counts().values[0]),

    'Course Forums' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)'].value_counts().values[0]),

    'YouTube' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)'].value_counts().values[0]),

    'Podcasts' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)'].value_counts().values[0]),

    'Blogs' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)'].value_counts().values[0]),

    'Journal Publications' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)'].value_counts().values[0]),

    'Slack Communities' : (dataset_mcq['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)'].value_counts().values[0])

            }

#Convert Data dictonary to series

mediasource_series=pd.Series(mediasource_count_dict)

#Visualizing media source

fig = px.bar(mediasource_series, x=mediasource_series.values, y=mediasource_series.index,orientation='h')

fig.show()
#Take out Online resources seprately and put in Data Dictonary

onlinecourses_dict = {

    'Udacity' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity'].value_counts().values[0]),

    'Coursera': (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera'].value_counts().values[0]),

    'edX' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX'].value_counts().values[0]),

    'DataCamp' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp'].value_counts().values[0]),

    'DataQuest' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest'].value_counts().values[0]),

    'Kaggle Course' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Courses (i.e. Kaggle Learn)'].value_counts().values[0]),

    'Fast.ai' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai'].value_counts().values[0]),

    'Udemy' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy'].value_counts().values[0]),

    'LinkedIn Learning' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning'].value_counts().values[0]),

    'University Course' : (dataset_mcq['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)'].value_counts().values[0])

}

#Convert Data dictonary to series

onlinecourses_series=pd.Series(onlinecourses_dict)

#Visualizing onlinecourses series

fig = px.bar(onlinecourses_series, x=onlinecourses_series.values, y=onlinecourses_series.index,orientation='h')

fig.show()
#Taking out India's participants

dataset_mcq_india=dataset_mcq[dataset_mcq['In which country do you currently reside?'] == 'India']

#Take out Online resources seprately and put in Data Dictonary

mediasource_count_dict_india = {

    'Twitter' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)'].value_counts().values[0]),

    'Hacker': (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News (https://news.ycombinator.com/)'].value_counts().values[0]),

    'Reddit' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)'].value_counts().values[0]),

    'Kaggle' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)'].value_counts().values[0]),

    'Course Forums' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)'].value_counts().values[0]),

    'YouTube' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)'].value_counts().values[0]),

    'Podcasts' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)'].value_counts().values[0]),

    'Blogs' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)'].value_counts().values[0]),

    'Journal Publications' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)'].value_counts().values[0]),

    'Slack Communities' : (dataset_mcq_india['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)'].value_counts().values[0])

            }

#Convert Data dictonary to series

mediasource_series_india=pd.Series(mediasource_count_dict_india)

#Visualizing media source

fig = px.bar(mediasource_series_india, x=mediasource_series_india.values, y=mediasource_series_india.index,orientation='h')

fig.show()
#Take out Online resources seprately and put in Data Dictonary

onlinecourses_dict_india = {

    'Udacity' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity'].value_counts().values[0]),

    'Coursera': (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera'].value_counts().values[0]),

    'edX' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX'].value_counts().values[0]),

    'DataCamp' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp'].value_counts().values[0]),

    'DataQuest' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest'].value_counts().values[0]),

    'Kaggle Course' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Courses (i.e. Kaggle Learn)'].value_counts().values[0]),

    'Fast.ai' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai'].value_counts().values[0]),

    'Udemy' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy'].value_counts().values[0]),

    'LinkedIn Learning' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning'].value_counts().values[0]),

    'University Course' : (dataset_mcq_india['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)'].value_counts().values[0])

}

#Convert Data dictonary to series

onlinecourses_series_india=pd.Series(onlinecourses_dict_india)

#Visualizing onlinecourses series

fig = px.bar(onlinecourses_series_india, x=onlinecourses_series_india.values, y=onlinecourses_series_india.index,orientation='h')

fig.show()
#Take out programming language seprately and put in Data Dictonary

programminglang_dict = {

 'Python' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python'].count()),

 'R': (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R'].count()),

 'SQL' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL'].count()),

 'C' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C'].count()),

 'C++' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++'].count()),

 'Java ' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java'].count()),

 'Javascript' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript'].count()),

 'Typescript' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - TypeScript'].count()),

 'Bash ' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash'].count()),

 'MATLAB' : (dataset_mcq['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB'].count())

}



#Convert Data dictonary to series

programminglang_series=pd.Series(programminglang_dict)

#Visualizing frequently used programming language series

fig = px.scatter(programminglang_series, y=programminglang_series.values, x=programminglang_series.index,size=programminglang_series.values)

fig.show()



#Taking out recomemded language in a series

recommendedlang_series = dataset_mcq['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()

#Visualizing recommended programming language series

fig = px.scatter(recommendedlang_series, y=recommendedlang_series.values, x=recommendedlang_series.index,size=recommendedlang_series.values)

fig.show()
#Take out programming language for india seprately and put in Data Dictonary

programminglang_india_dict = {

 'Python' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python'].count()),

 'R': (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R'].count()),

 'SQL' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL'].count()),

 'C' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C'].count()),

 'C++' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++'].count()),

 'Java ' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java'].count()),

 'Javascript' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript'].count()),

 'Typescript' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - TypeScript'].count()),

 'Bash ' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash'].count()),

 'MATLAB' : (dataset_mcq_india['What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB'].count())

}

#Convert Data dictonary to series

programminglang_india_series=pd.Series(programminglang_india_dict)

#Visualizing onlinecourses series

fig = px.scatter(programminglang_india_series, y=programminglang_india_series.values, x=programminglang_india_series.index,size=programminglang_india_series.values)

fig.show()
#Taking out recomemded language from india in a series

recommendedlang_india_series = dataset_mcq_india['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()

#Visualizing recommended programming language series

fig = px.scatter(recommendedlang_india_series, y=recommendedlang_india_series.values, x=recommendedlang_india_series.index,size=recommendedlang_india_series.values)

fig.show()
#Take out programming language seprately and put in Data Dictonary

visuallib_dict = {

 'Ggplot' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 '].count()),

 'Matplotlib': (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib '].count()),

 'Altair' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair '].count()),

 'Shiny' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny '].count()),

 'D3' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3.js '].count()),

 'Plotly' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express '].count()),

 'Bokeh' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh '].count()),

 'Seaborn' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn '].count()),

 'Geoplotlib' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib '].count()),

 'Leaflet-Folium' : (dataset_mcq['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium '].count())

}



#Convert Data dictonary to series

visuallib_series=pd.Series(visuallib_dict)

#Visualizing frequently used programming language series

fig = px.scatter(visuallib_series, y=visuallib_series.values, x=visuallib_series.index,size=visuallib_series.values)

fig.show()

#Take out programming language seprately and put in Data Dictonary

visuallib_india_dict = {

 'Ggplot' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 '].count()),

 'Matplotlib': (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib '].count()),

 'Altair' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair '].count()),

 'Shiny' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny '].count()),

 'D3' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3.js '].count()),

 'Plotly' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express '].count()),

 'Bokeh' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh '].count()),

 'Seaborn' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn '].count()),

 'Geoplotlib' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib '].count()),

 'Leaflet-Folium' : (dataset_mcq_india['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium '].count())

}



#Convert Data dictonary to series

visuallib_india_series=pd.Series(visuallib_india_dict)

#Visualizing frequently used programming language series

fig = px.scatter(visuallib_india_series, y=visuallib_india_series.values, x=visuallib_india_series.index,size=visuallib_india_series.values)

fig.show()

#Take out machine learning algo seprately and put in Data Dictonary

mlaglo_dict = {

 'Linear or Logistic Regression' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression'].count()),

 'Decision Trees or Random Forests': (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests'].count()),

 'Gradient Boosting Machines' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)'].count()),

 'Bayesian Approaches' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches'].count()),

 'Evolutionary Approaches' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches'].count()),

 'Dense Neural Networks (MLPs, etc) ' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)'].count()),

 'Convolutional Neural Networks' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks'].count()),

 'Generative Adversarial Networks ' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks'].count()),

 'Recurrent Neural Networks' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks'].count()),

 'Transformer Networks (BERT, gpt-2, etc)' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-2, etc)'].count()),

 'None' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - None'].count()),

 'Other' : (dataset_mcq['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other'].count()),

}



mlaglo_india_dict = {

 'Linear or Logistic Regression' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression'].count()),

 'Decision Trees or Random Forests': (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests'].count()),

 'Gradient Boosting Machines' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)'].count()),

 'Bayesian Approaches' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches'].count()),

 'Evolutionary Approaches' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches'].count()),

 'Dense Neural Networks (MLPs, etc) ' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)'].count()),

 'Convolutional Neural Networks' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks'].count()),

 'Generative Adversarial Networks ' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks'].count()),

 'Recurrent Neural Networks' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks'].count()),

 'Transformer Networks (BERT, gpt-2, etc)' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-2, etc)'].count()),

 'None' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - None'].count()),

 'Other' : (dataset_mcq_india['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other'].count()),

}



#Convert Data dictonary to series

mlaglo_series=pd.Series(mlaglo_dict)

mlaglo_india_series=pd.Series(mlaglo_india_dict)



#Visualizing frequently used machine learning algorithm series

fig = go.Figure(data=[

    go.Bar(name='ROW', x=mlaglo_series.index, y=mlaglo_series.values),

    go.Bar(name='India',  x=mlaglo_india_series.index, y=mlaglo_india_series.values)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
#Take out machine learning tools seprately and put in Data Dictonary

mltool_dict = {

 'Automated data augmentation (e.g. imgaug, albumentations)' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)'].count()),

 'Automated feature engineering/selection (e.g. tpot, boruta_py)': (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)'].count()),

 'Automated model architecture searches (e.g. darts, enas)' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)'].count()),

 'Automated model selection (e.g. auto-sklearn, xcessiv)' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)'].count()),

 'Automated hyperparameter tuning (e.g. hyperopt, ray.tune)' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune)'].count()),

 'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)'].count()),

 'None ' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None'].count()),

 'Other' : (dataset_mcq['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].count()),

}

 

mltool_india_dict = {

 'Automated data augmentation (e.g. imgaug, albumentations)' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)'].count()),

 'Automated feature engineering/selection (e.g. tpot, boruta_py)': (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)'].count()),

 'Automated model architecture searches (e.g. darts, enas)' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)'].count()),

 'Automated model selection (e.g. auto-sklearn, xcessiv)' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)'].count()),

 'Automated hyperparameter tuning (e.g. hyperopt, ray.tune)' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune)'].count()),

 'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)'].count()),

 'None ' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None'].count()),

 'Other' : (dataset_mcq_india['Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].count()),

 }

 

 #Convert Data dictonary to series

mltool_series=pd.Series(mltool_dict)

mltool_india_series=pd.Series(mltool_india_dict)



#Visualizing frequently used machine learning algorithm series

fig = go.Figure(data=[

    go.Bar(name='ROW', x=mltool_series.index, y=mltool_series.values),

    go.Bar(name='India',  x=mltool_india_series.index, y=mltool_india_series.values)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
#Take out machine learning framework seprately and put in Data Dictonary



mlfw_dict = {

 'Scikit-learn' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn '].count()),

 'TensorFlow': (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow '].count()),

 'Keras' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras '].count()),

 'RandomForest' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  RandomForest'].count()),

 'Xgboost' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost '].count()),

 'PyTorch' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch '].count()),

 'Caret' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret '].count()),

 'LightGBM' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM '].count()),

 'SparkMLib' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Spark MLib '].count()),

 'Fast.ai' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai '].count()),

  'None' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - None'].count()),

 'Other' : (dataset_mcq['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].count())

}



mlfw_india_dict = {

 'Scikit-learn' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn '].count()),

 'TensorFlow': (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow '].count()),

 'Keras' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras '].count()),

 'RandomForest' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  RandomForest'].count()),

 'Xgboost' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost '].count()),

 'PyTorch' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch '].count()),

 'Caret' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret '].count()),

 'LightGBM' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM '].count()),

 'SparkMLib' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Spark MLib '].count()),

 'Fast.ai' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai '].count()),

 'None' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - None'].count()),

 'Other' : (dataset_mcq_india['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].count())

 }



 

 #Convert Data dictonary to series

mlfw_series=pd.Series(mlfw_dict)

mlfw_india_series=pd.Series(mlfw_india_dict)



#Visualizing frequently used machine learning algorithm series

fig = go.Figure(data=[

    go.Bar(name='ROW', x=mlfw_series.index, y=mlfw_series.values),

    go.Bar(name='India',  x=mlfw_india_series.index, y=mlfw_india_series.values)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
#Taking out relevant columns related to Experience in different data frame

dataset_codemlexp=dataset_mcq[['How long have you been writing code to analyze data (at work or at school)?','For how many years have you used machine learning methods?']]

dataset_codemlexp_india=dataset_mcq_india[['How long have you been writing code to analyze data (at work or at school)?','For how many years have you used machine learning methods?']]



#Replacing column name to short and relevant name

dataset_codemlexp=dataset_codemlexp.rename(columns = {'How long have you been writing code to analyze data (at work or at school)?':'CodeExp','For how many years have you used machine learning methods?':'MLExp'})

dataset_codemlexp_india=dataset_codemlexp_india.rename(columns = {'How long have you been writing code to analyze data (at work or at school)?':'CodeExp','For how many years have you used machine learning methods?':'MLExp'})



#Converting Code experince into series for visualization

codeexp_series_india=pd.Series(dataset_codemlexp_india.iloc[:,0])

codeexp_series=pd.Series(dataset_codemlexp.iloc[:,0])



#Converting ML experince into series for visualization

mlexp_series_india=pd.Series(dataset_codemlexp_india.iloc[:,1])

mlexp_series=pd.Series(dataset_codemlexp.iloc[:,1])



#Visualizing Code Experince

fig = go.Figure(data=[

    go.Bar(name='ROW',x=codeexp_series.index,y=codeexp_series.values,orientation='h'),

    go.Bar(name='India',x=codeexp_series_india.index, y=codeexp_series_india.values,orientation='h')

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
#Visualizing Code Experince

fig = go.Figure(data=[

    go.Bar(name='ROW', x=mlexp_series.index,y=mlexp_series.values,orientation='h'),

    go.Bar(name='India',  x=mlexp_series_india.index,y=mlexp_series_india.values,orientation='h')

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
#Take out machine learning framework seprately and put in Data Dictonary



mltasks_dict = {

 'Analyze Data' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions'].count()),

 'Build-Run DataInfrastructure': (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data'].count()),

 'BuildPrototypes' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas'].count()),

 'Build-Run MLService' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows'].count()),

 'Experiment-Iterate ML Model' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models'].count()),

 'AdvanceResearch' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning'].count()),

 'None' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work'].count()),

 'Other' : (dataset_mcq['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'].count()),

}



mltasks_india_dict = {

 'Analyze Data' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions'].count()),

 'Build-Run DataInfrastructure': (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data'].count()),

 'BuildPrototypes' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas'].count()),

 'Build-Run MLService' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows'].count()),

 'Experiment-Iterate ML Model' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models'].count()),

 'AdvanceResearch' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning'].count()),

 'None' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work'].count()),

 'Other' : (dataset_mcq_india['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'].count()),

}



 #Convert Data dictonary to series

mltasks_series=pd.Series(mltasks_dict)

mltasks_india_series=pd.Series(mltasks_india_dict)



#Visualize the ml tasks globally

fig = px.scatter(mltasks_series, y=mltasks_series.values, x=mltasks_series.index,size=mltasks_series.values,color=mltasks_series.values)

fig.show()
#Visualizing ml tasks for Indians

fig = px.scatter(mltasks_india_series, y=mltasks_india_series.values, x=mltasks_india_series.index,size=mltasks_india_series.values,color=mltasks_india_series.values)

fig.show()