# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# ##import plotly

# import plotly.plotly as py1

# import plotly.offline as py

# py.init_notebook_mode(connected=True)

# from plotly.offline import init_notebook_mode, iplot

# init_notebook_mode(connected=True)



# import plotly.offline as offline

# offline.init_notebook_mode()

from plotly import tools





import plotly.graph_objs as go



import matplotlib.pyplot as plt

# import cufflinks and offline mode





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
survey = pd.read_csv('../input/survey_results_public.csv', low_memory=False)

survey_schema = pd.read_csv('../input/survey_results_schema.csv')
survey.head()
temp = pd.DataFrame(survey.DevType.dropna().str.split(';').tolist()).stack().value_counts().sort_values(ascending = True)



data = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

    )

    

)

layout = dict(

    title = 'Who Took the Surveys?',

    height = 800,

#     xaxis = '# of Participants', 

    margin = dict(

        l=300,

    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = pd.DataFrame(survey.Gender.dropna().str.split(';').tolist()).stack().value_counts(dropna = False).sort_values(ascending = True)

labels = temp.index

values = temp.values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = pd.DataFrame(round(survey.CompanySize.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print(" {} ".format(temp.columns[1]).center(60,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% works in a company of {}".format(b, a))

print ('#####')

print ('27.64 participants had no feedback for this question')
temp = pd.DataFrame(survey.Country.value_counts()).reset_index()

temp.rename(columns={'index':'Country','Country':'value'}, inplace=True)



fig = go.Figure(data=go.Choropleth(

    locations = temp['Country'],

    z = temp['value'],

    text = temp['Country'],

    locationmode='country names',

    colorscale = 'bluered',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '$',

    colorbar_title = 'Count of the<br>Participants',

))



fig.update_layout(

    title_text='Where are they From?',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),



)



fig.show()
temp = survey.Country.value_counts(ascending = False).head(15)

data = go.Bar(

    y = temp.values,

    x = temp.index,

#     orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

    )

    

)

layout = dict(

    title = 'Who Took the Surveys?',

    height = 800,

#     xaxis = '# of Participants', 

    margin = dict(

#         l=300,

    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = pd.DataFrame(round(survey.Country.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

temp = temp.head(10)

print(" {} ".format(temp.columns[1]).center(60,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% participants of the survey are from {}.".format(b, a))

#print ('#####')

#print ('27.64 participants did not share an answer for this question')

print ("Let's find out what they do..")
temp = survey.Student.value_counts(dropna = False)

labels = temp.index

values = temp.values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = survey.UndergradMajor.value_counts(dropna = False).sort_values(ascending = True)

trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

    )

    

)

layout = dict(

    title = 'What is/was Their Major in College?',

    height = 800,

    margin = dict(

        l=450,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()



#round(survey.UndergradMajor.value_counts(dropna = False, normalize=True)*100,2) 
temp = survey.Employment.value_counts(dropna = False)

data = go.Bar(

    y = temp.values,

    x = temp.index,

#     orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

    )

    

)

layout = dict(

    title = 'Who Took the Surveys?',

    height = 800,

#     xaxis = '# of Participants', 

    margin = dict(

#         l=300,

    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = survey.FormalEducation.value_counts(dropna = True).sort_values(ascending = True)

trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

    )

    

)

layout = dict(

    title = 'What is/was Their Formal Education?',

    height = 800,

    margin = dict(

        l=530,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()



#round(survey.FormalEducation.value_counts(dropna = False, normalize=True)*100,2) 
temp = survey.CompanySize.value_counts(dropna = False)

labels = temp.index

values = temp.values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = pd.DataFrame(round(survey.CompanySize.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print(" {} ".format(temp.columns[1]).center(60,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% works in a company of {}".format(b, a))

print ('#####')

print ('27.64 participants had no feedback for this question')
temp = survey.OperatingSystem.value_counts().sort_values(ascending = True )

labels = temp.index

values = temp.values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = pd.DataFrame(round(survey.OperatingSystem.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print(" {} ".format(temp.columns[1]).center(60,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% uses {} as their operating system".format(b, a))

print ('#####')

print ('22.9% participants did not share an answer for this question')
temp = pd.DataFrame(survey.PlatformWorkedWith.dropna().str.split(';').tolist()).stack().value_counts().sort_values(ascending = True)



trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        color = 'rgba(50, 205, 50, 0.6)',

        line = dict(

            color = 'rgba(0, 0, 0, 1.0)',

            width = 2))

    

)

layout = dict(

    title = 'What Platform did They Work With?',

    height = 1000,

    #xaxis = '# of Participants', 

    margin = dict(

        l=210,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = pd.DataFrame(survey.IDE.dropna().str.split(';').tolist()).stack().value_counts().sort_values(ascending = True)



trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

        line = dict(

            color = 'rgba(0, 0, 0, 1.0)',

            width = 2)

    )

    

)

layout = dict(

    title = 'What IDE environment do they use?',

    height = 800,

    #xaxis = '# of Participants', 

    margin = dict(

        l=250,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = pd.DataFrame(survey.VersionControl.dropna().str.split(';').tolist()).stack().value_counts().sort_values(ascending = True)



trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

        line = dict(

            color = 'rgba(0, 0, 0, 1.0)',

            width = 2)

    )

    

)

layout = dict(

    title = 'Who Took the Surveys?',

    height = 800,

    #xaxis = '# of Participants', 

    margin = dict(

        l=290,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = survey.YearsCoding.value_counts(dropna = False)



trace = go.Bar(

    y = temp.values,

    x = temp.index,

#     orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

        line = dict(

            color = 'rgba(0, 0, 0, 1.0)',

            width = 2)

    )

    

)

layout = dict(

    title = 'How many years of experience do they have?',

    height = 800,

    #xaxis = '# of Participants', 

    margin = dict(

#         l=290,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = survey.YearsCodingProf.value_counts(dropna = False).sort_values(ascending = True)

trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

        line = dict(

            color = 'rgba(0, 0, 0, 1.0)',

            width = 2)

    )

    

)

layout = dict(

    title = 'Who Took the Surveys?',

    height = 800,

    #xaxis = '# of Participants', 

    margin = dict(

        l=290,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()
trace1 = go.Bar(

    y=survey.JobSatisfaction.value_counts().sort_values(ascending = True).index,

    x=survey.JobSatisfaction.value_counts().sort_values(ascending = True).values,

    name='Job Satisfaction',

    orientation = 'h',

    marker = dict(

        color = 'rgba(246, 78, 139, 0.6)',

        line = dict(

            color = 'rgba(246, 78, 139, 1.0)',

            width = 3)

    )

)

trace2 = go.Bar(

    y=survey.CareerSatisfaction.value_counts().index,

    x=survey.CareerSatisfaction.value_counts().values,

    name='Career Satisfaction',

    orientation = 'h',

    marker = dict(

        color = 'rgba(50, 205, 50, 0.6)',

        line = dict(

            color = 'rgba(50, 205, 50, 1.0)',

            width = 3)

    )

)



data = [trace1, trace2]

layout = go.Layout(

    margin = dict(l=200,),

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

fig.show()



#print("*****Job Satisfaction****")

#print(round(survey.JobSatisfaction.value_counts(dropna = False, normalize=True)*100,2) )

#print("****Career Satisfaction****")

#print(round(survey.CareerSatisfaction.value_counts(dropna = False, normalize=True)*100,2) )
temp = pd.DataFrame(round(survey.JobSatisfaction.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print ("***** Job Satisfaction *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% are {}".format(b, a))

print ('*****')

print ('29.92 participants did not share an answer for this question')





print (" ")

temp = pd.DataFrame(round(survey.CareerSatisfaction.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print ("***** Career Satisfaction *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% are {}".format(b, a))

print ('*****')

print ('22.61 participants did not share an answer for this question')

print()

print("It looks like most of the participants are doing alright. There are; however, 2482 participants who extremely dissatisfied with their jobs and 2613 participants, who are extremely dissatisfied with their career. I am interested in finding out why?. In any situation in life, I believe to be satisfied, few things are essential. I think health is the first one, then the community work. There is no replacement of the power of giving. Then there is also the necessity of being perceived in a society, in other words how accepted you are in a community. We are built to live together, and our difference is quite tiny compared to our similarities. However, we live in a society where people cling to things and value things that has less importance in the long run. We will look into some of this aspects since I believe this dataset can give us a chance to do that. And last but not the least, how financially solved someone is. I hope to dig deep into these issues based on the data we have. Let us study them one by one.")
temp = survey.Hobby.value_counts(dropna = False)

labels = temp.index

values = temp.values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = pd.DataFrame(round(survey.Hobby.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print(" {} ".format(temp.columns[1]).center(40,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% says {} to coding as a hobby".format(b, a))

#print ('#####')

#print ('23.40% participants did not share an answer for this question')

print ('It seems like approximately 20% do not consider coding as their hobby. ')
temp = survey.OpenSource.value_counts(dropna = False)



labels = temp.index

values = temp.values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = pd.DataFrame(round(survey.OpenSource.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print(" {} ".format(temp.columns[1]).center(55,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% said {} to contributing to open source projects".format(b, a))

#print ('#####')

#print ('23.40% participants did not share an answer for this question')
temp = survey.HopeFiveYears.value_counts(dropna = True).sort_values(ascending = True)

trace = go.Bar(

    x = temp.values,

    y = temp.index,

    orientation = 'h',

    marker = dict(

        colorscale = 'Greens',

    )

    

)

layout = dict(

    title = 'What are their Hopes in Five Years?',

    height = 800,

    margin = dict(

        l=500,

    )

)

data =[trace]

fig = go.Figure(data = data, layout = layout)

fig.show()



#round(survey.FormalEducation.value_counts(dropna = False, normalize=True)*100,2) 
temp = pd.DataFrame(round(survey.HopeFiveYears.value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':'job_satisfaction','JobSatisfaction':'percentage'})

temp.dropna(inplace=True)

print(" {} ".format(temp.columns[1]).center(60,"*"))

#print ("***** CompanySize *****".center(60, '*') )

for a, b in temp.itertuples(index=False):

    print("{}% hopes to {}".format(b, a))

print ('#####')

print ('23.40% participants did not share an answer for this question')