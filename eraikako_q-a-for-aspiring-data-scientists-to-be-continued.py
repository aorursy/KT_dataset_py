# Import Libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import matplotlib.ticker as ticker

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image



# Plotly visualizations

from plotly import tools

import chart_studio.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the Dataset and apply a filter for retrieval of Data Scientist



responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

text_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

data_scientist_responses = responses.loc[responses['Q5'] == 'Data Scientist']



# Drop all the unused columns

data_scientist_responses = data_scientist_responses.drop(['Q13_OTHER_TEXT', 'Q12_OTHER_TEXT'], axis=1)

data_scientist_text_responses = pd.concat([data_scientist_responses, text_responses], axis=1, join='inner')



print("The number of data scientists is ", data_scientist_responses.shape[0])

print("The number of data scientists text responses is ", data_scientist_text_responses.shape[0])

def concatenate_multiple_answers(list_of_options, df, column_new_name):

    """

    

    """

    frames = []

    for option in list_of_options:

        frames.append(df.loc[:, [option, "Q5"]].dropna().rename(columns={option: column_new_name, "Q5": "Job_Title"}))

    

    return pd.concat(frames)
multiple_answers = [

    'Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_Part_4', 'Q12_Part_5',

    'Q12_Part_6', 'Q12_Part_7', 'Q12_Part_8', 'Q12_Part_9', 'Q12_Part_10',

    'Q12_Part_12'

]

media_sources = concatenate_multiple_answers(multiple_answers, data_scientist_text_responses, "Media_Sources")

media_sources = media_sources.Media_Sources.value_counts().to_dict()



# Sorted Descending

x = list(media_sources.values())

y = list(media_sources.keys())

sorty = [x for _,x in sorted(zip(x,y))]

sortx = sorted(x)



fig = go.Figure(go.Bar(

            x=sortx,

            y=sorty,

            orientation='h',

            marker=dict(

                color='rgba(122, 120, 168, 0.8)')

            

))



fig.update_layout(

    title="The favorite media sources for data science topics",



    font=dict(

        family="Arial"

    )

)

fig.show()
multiple_answers = [

    'Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5',

    'Q13_Part_6', 'Q13_Part_7', 'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10'

]

online_courses = concatenate_multiple_answers(multiple_answers, data_scientist_responses, "Online_Courses")

other_online_courses = data_scientist_text_responses.loc[:, ['Q5', 'Q13_OTHER_TEXT']].dropna().rename(columns={'Q13_OTHER_TEXT': 'Online_Courses', "Q5": "Job_Title"})

top3_other_courses = list(other_online_courses.Online_Courses.value_counts().to_dict())[:2]

other_online_courses = other_online_courses[other_online_courses['Online_Courses'].isin(top3_other_courses)]

final_online_courses = pd.concat([online_courses, other_online_courses], sort=False)



ncount = data_scientist_responses.shape[0]

plt.figure(figsize=(12,6))

ax = sns.countplot(x="Online_Courses", data=final_online_courses, order=final_online_courses['Online_Courses'].value_counts().index, palette="RdBu")

plt.title('Platforms of online courses chosen by Data Scientists')

plt.xlabel('Platforms for Data Science Courses')

plt.ylabel(None)

plt.xticks(rotation=45, horizontalalignment='right')



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text



# Use a LinearLocator to ensure the correct number of ticks

ax.yaxis.set_major_locator(ticker.LinearLocator(11))



# Fix the frequency range to 0-100

ax.set_ylim(0,ncount)
multiple_answers = [

    'Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5',

    'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10',

    'Q18_Part_11', 'Q18_Part_12',

]

programming_languages = concatenate_multiple_answers(multiple_answers, data_scientist_responses, "Programming_Language")

programming_languages_counts = programming_languages.Programming_Language.value_counts().to_dict()

recommended_programming_languages = data_scientist_responses.Q19.value_counts().to_dict()




y = list(programming_languages_counts.values())



x = list(programming_languages_counts.keys())



sortx = [x for _,x in sorted(zip(y,x))]

sorty = sorted(y)



y2 = list(recommended_programming_languages.values())



x2 = list(recommended_programming_languages.keys())



sortx2 = [x2 for _,x2 in sorted(zip(y2,x2))]

sorty2 = sorted(y2)



import plotly.graph_objects as go

from plotly.subplots import make_subplots



import numpy as np







# Creating two subplots

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False,

                    shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(go.Bar(

    x=sorty,

    y=sortx,

    marker=dict(

        color='rgba(50, 171, 96, 0.6)',

        line=dict(

            color='rgba(50, 171, 96, 1.0)',

            width=1),

    ),

    name='What programming languages do you use on a regular basis?',

    orientation='h',

), 1, 1)



fig.append_trace(go.Bar(

    x=sorty2,

    y=sortx2,

    marker=dict(

        color='rgb(128, 0, 128)',

        line=dict(

            color='rgb(128, 0, 128)',

            width=1),

    ),

    name='What programming language would you recommend an aspiring data scientist to learn first?',

    orientation='h',

), 1, 2)





fig.update_layout(

    title='Programming Languages Most Used and Recommended by Data Scientists',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    yaxis2=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0, 0.42],

    ),

    xaxis2=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0.47, 1],

        dtick=25000,

    ),

    legend=dict(x=0.029, y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []



y_s = np.round(y, decimals=2)



# Adding labels

for yd, xd in zip(y_s, x):

    # labeling the scatter savings



    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 300,

                            text=str(np.round((yd * 100)/data_scientist_responses.shape[0], decimals=2)) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(50, 171, 96)'),

                            showarrow=False))





y_s2 = np.round(y2, decimals=2)



# Adding labels

for yd, xd in zip(y_s2, x2):

    # labeling the scatter savings



    # labeling the bar net worth

    annotations.append(dict(xref='x2', yref='y2',

                            y=xd, x=yd + 300,

                            text=str(np.round((yd * 100)/data_scientist_responses.shape[0], decimals=2)) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(128, 0, 128)'),

                            showarrow=False))

    

fig.update_layout(annotations=annotations)



fig.show()

multiple_answers = [

    'Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3', 'Q29_Part_4', 'Q29_Part_5',

    'Q29_Part_6', 'Q29_Part_7', 'Q29_Part_8', 'Q29_Part_9', 'Q29_Part_10'

]

cloud_computing_platforms = concatenate_multiple_answers(multiple_answers, data_scientist_responses, "Cloud_Computing_Platforms")

cloud_computing_platforms = cloud_computing_platforms['Cloud_Computing_Platforms'].value_counts() 

cloud_computing_platforms = cloud_computing_platforms.sort_values(ascending=False)

# plot the data using seaborn like before

plt.figure(figsize=(14,6))

sns.set(font_scale=1.5)

ax = sns.barplot(x = cloud_computing_platforms.values , y = cloud_computing_platforms.index, alpha=0.5)

plt.ylabel('Platforms')

plt.xlabel('Data Scientists')

plt.title('Cloud Computing Platforms used on a regular basis')

plt.show()
fig = plt.figure()

plt.figure(figsize=(8,8))

ax = sns.countplot(x="Q4", data=data_scientist_responses,palette="Set2",

             order = data_scientist_responses['Q4'].value_counts().index)



total = float(len(data_scientist_responses)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height * 100 /total) ,

            ha="center") 

    

plt.xticks(rotation=90)

plt.title("Data Scientists' Formal Education level",fontsize=15)

plt.xticks(rotation=45, horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Highest Level of Education')

plt.show()
plt.figure(figsize=(8,8))

ax=data_scientist_responses['Q6'].value_counts().plot.barh(width=0.9,color=sns.color_palette('Set3',25))

plt.gca().invert_yaxis()

plt.title('Company Sizes')

plt.show()
education_cross = pd.crosstab(data_scientist_responses['Q6'], data_scientist_responses['Q4'])

number_of_education = pd.crosstab(data_scientist_responses['Q6'], data_scientist_responses['Q4'])





education_cross['Master’s degree'] = education_cross['Master’s degree'].apply(lambda x: round(x, 2))

education_cross['Bachelor’s degree'] = education_cross['Bachelor’s degree'].apply(lambda x: round(x, 2))

education_cross['Doctoral degree'] = education_cross['Doctoral degree'].apply(lambda x: round(x, 2))

education_cross['Professional degree'] = education_cross['Professional degree'].apply(lambda x: round(x, 2))

education_cross['Some college/university study without earning a bachelor’s degree'] = education_cross['Some college/university study without earning a bachelor’s degree'].apply(lambda x: round(x, 2))

education_cross['I prefer not to answer'] = education_cross['I prefer not to answer'].apply(lambda x: round(x, 2))

education_cross['No formal education past high school'] = education_cross['No formal education past high school'].apply(lambda x: round(x, 2))



number_of_education['Total'] = number_of_education.sum(axis=1) 

ds_master = education_cross['Master’s degree'].values.tolist()

ds_bachelor = education_cross['Bachelor’s degree'].values.tolist()

ds_phd = education_cross['Doctoral degree'].values.tolist()

ds_prof = education_cross['Professional degree'].values.tolist()

ds_without_degree = education_cross['Some college/university study without earning a bachelor’s degree'] .values.tolist()

ds_not_answer = education_cross['I prefer not to answer'].values.tolist()

ds_not_educated = education_cross['No formal education past high school'].values.tolist()



masters = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y= ds_master,

    name='Master’s degree',

    marker=dict(

        color='rgb(192, 148, 246)'

    )

)



bachelors = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y=ds_bachelor,

    name='Bachelor’s degree',

    marker=dict(

        color='rgb(176, 26, 26)'

    )

)



phds = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y= ds_phd,

    name='Doctoral degree',

    marker = dict(

        color='rgb(229, 121, 36)'

    )

)



pros = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y= ds_prof,

    name='Professional degree',

    marker = dict(

        color='rgb(147, 147, 147)'

    )

)



uneducated = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y= ds_without_degree,

    name='Some college/university study without earning a bachelor’s degree', 

    marker = dict(

        color='rgb(246, 157, 135)'

    )

)



not_answer = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y= ds_not_answer,

    name='I prefer not to answer',

    marker = dict(

        color = 'rgb(238, 76, 73)'

        )

)



no_formal_education = go.Bar(

    x=['0-49 employees', '1000-9,999 employees', '250-999 employees',

       '50-249 employees', '> 10,000 employees'],

    y= ds_not_educated,

    name='No formal education past high school',

    marker = dict(

        color = 'rgb(247, 235, 195)'

        )

)



data = [masters, bachelors, phds, pros, uneducated, not_answer, no_formal_education]

layout = go.Layout(

    barmode='stack',

    title = '% of several education level Data Scientists by Company Size',

    xaxis=dict(title="Company's Size")

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stacked-bar')
def calculate_area(row):

    salary_range = row['Q10'].replace(',', '').replace('>', '').replace('$', '')

    salaries = list(map(int, salary_range.split('-')))

    

    

    return np.mean(salaries)



df_data_scientists = data_scientist_responses.dropna(subset=['Q10'])



df_data_scientists['boxplot']= df_data_scientists.apply(calculate_area, axis=1)

plt.figure(figsize=(14,8))

ax = sns.boxplot(x="Q4", y="boxplot", data=df_data_scientists, palette="Set3")

ax.set_xticklabels(ax.get_xticklabels(),rotation=85)

ax.set_xlabel("Education Level", fontsize=12)

ax.set_ylabel("Yearly compensation - $USD", fontsize=12)

ax.set_title("Distribution of Yearly compensation \n according to the education level", fontsize=16)
plt.figure(figsize=(18,18))

ax = sns.boxplot(x="Q6", y="boxplot", data=df_data_scientists, palette="Set3", hue="Q4")

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax.set_xlabel("Company's size", fontsize=12)

ax.set_ylabel("Yearly compensation - $USD", fontsize=12)

ax.set_title("Distribution of Yearly compensation according to \n the education level and the company's size", fontsize=16)
ml_algorithms = []



for column in data_scientist_responses.loc[:,'Q24_Part_1':'Q24_Part_12']:

    columnSeriesObj = data_scientist_responses[column]

    

    current_list = columnSeriesObj.values.tolist()

    ml_algorithms.extend(current_list)



df_ml_algorithms = pd.DataFrame(ml_algorithms, columns =['ML_Algorithms'])

df_ml_algorithms = df_ml_algorithms.dropna()

df_ml_algorithms.head()

text = " ".join(algorithm for algorithm in df_ml_algorithms.ML_Algorithms)


text = df_ml_algorithms.ML_Algorithms.value_counts().to_dict()

wc = WordCloud(width=800, height=400, max_words=200, margin=0, colormap="Blues", min_font_size=10).generate_from_frequencies(text)

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 10))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
multiple_answers = ['Q28_Part_1', 'Q28_Part_2', 'Q28_Part_3', 'Q28_Part_4', 'Q28_Part_5', 

                    'Q28_Part_6', 'Q28_Part_7', 'Q28_Part_8', 'Q28_Part_9', 'Q28_Part_10']



ml_frameworks = concatenate_multiple_answers(multiple_answers, data_scientist_responses, "ML_frameworks")



fig = plt.figure()

plt.figure(figsize=(8,6))

ax = sns.countplot(x="ML_frameworks", data=ml_frameworks, color="blue",

             order = ml_frameworks['ML_frameworks'].value_counts().index)

    

plt.xticks(rotation=90)

plt.title("Data Scientists' Machine Learning Frameworks",fontsize=15)

plt.xticks(rotation=45, horizontalalignment='right')

plt.yticks( fontsize=10)

plt.ylabel(None)

plt.xlabel(None)

plt.show()
multiple_answers = ['Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4']



ds_nlp_methods = concatenate_multiple_answers(multiple_answers, data_scientist_responses, "NLP_Methods")



fig = plt.figure()

plt.figure(figsize=(8,6))

ax = sns.countplot(x="NLP_Methods", data=ds_nlp_methods, color="blue",

             order = ds_nlp_methods['NLP_Methods'].value_counts().index)

    

plt.xticks(rotation=90)

plt.title("The most popular Natural Language Processing Methods",fontsize=15)

plt.xticks(rotation=45, horizontalalignment='right')

plt.yticks( fontsize=10)

plt.ylabel(None)

plt.xlabel(None)

plt.show()