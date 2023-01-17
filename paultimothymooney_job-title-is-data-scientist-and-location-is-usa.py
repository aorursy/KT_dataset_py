# Import Python packages



import os

import numpy as np 

import pandas as pd

pd.set_option('display.max_columns', 5000)

import seaborn as sns

import plotly.graph_objs as go

import warnings 

warnings.filterwarnings("ignore")

from plotly.offline import init_notebook_mode, iplot 

init_notebook_mode(connected=True)



# Define Helper Functions



def loadCSV(base_dir,file_name):

    """Loads a CSV file into a Pandas DataFrame"""

    file_path = os.path.join(base_dir,file_name)

    df = pd.read_csv(file_path)

    return df



# Load Data



base_dir = '/kaggle/input/kaggle-survey-2019/'

survey_schema = loadCSV(base_dir,'survey_schema.csv')

multiple_choice = loadCSV(base_dir,'multiple_choice_responses.csv')

responses_only = multiple_choice[1:] 

print('Total number of responses to the 2019 Kaggle Data Science and Machine Learning Survey: ',responses_only.shape[0])

responses_only['Time from Start to Finish (seconds)'] = pd.DataFrame(pd.to_numeric(responses_only['Time from Start to Finish (seconds)'], errors='coerce'))

responses_only = responses_only[responses_only['Q5']=='Data Scientist'] 

responses_only = responses_only[responses_only['Q3']=='United States of America'] 

print('Number of responses when filtered for job title is Data Scientist and residence is USA: ',responses_only.shape[0])
def return_count(data,question_part):

    """Counts occurences of each value in a given column"""

    counts = data[question_part].value_counts()

    counts_df = pd.DataFrame(counts)

    return counts_df



def return_percentage(data,question_part,response_count):

    """Calculates percent of each value in a given column"""

    counts = data[question_part].value_counts()

    total = response_count

    percentage = (counts*100)/total

    value = [percentage]

    question = [data[question_part]][0]

    percentage_df = pd.DataFrame(data=value).T     

    return percentage_df



def return_percentage_multiple_choice_multiple_choice_selection(data,question_part,question_number,response_count):

    """Calculates percent of each value in a given column for multiple choice multiple selection questions"""

    counts = data[question_part].value_counts()

    total = response_count

    percentage = (counts*100)/total

    value = [percentage]

    question = [data[question_part]][0]

    percentage_df = pd.DataFrame(data=value).T 

    return percentage_df



def plot_multiple_choice(question_number,data,data_with_question_on_top,

                         title,y_axis_title,percent_instead_of_count,response_count):

    """Plot multiple choice questions sorted by total percent of responses"""

    print(question_number,':',data_with_question_on_top[question_number][0])

    if percent_instead_of_count == True:

        df = return_percentage(data, question_number,response_count)

    else:

        df = return_count(data, question_number)

    trace1 = go.Bar(

                    x = df.index,

                    y = df[question_number][0:10],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title, 

                       yaxis= dict(title=y_axis_title),showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)

      

def plot_unsorted_multiple_choice(question_number,data,data_with_question_on_top,title,y_axis_title,

                                  sorting_order_list,percent_instead_of_count,response_count):

    """Plot multiple choice questions sorted in the order that you prefer"""

    print(question_number,':',data_with_question_on_top[question_number][0])

    if percent_instead_of_count == True:

        df = return_percentage(data, question_number,response_count)

    else:

        df = return_count(data, question_number)

    trace1 = go.Bar(

                    x = df.index,

                    y = df[question_number],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title, 

                       xaxis=dict(type='category',categoryorder='array',categoryarray=sorting_order_list), 

                       yaxis= dict(title=y_axis_title),showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)    



    

def plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                                question_part_1,question_part_2,question_part_3,question_part_4,

                                                question_part_5,question_part_6,question_part_7,question_part_8,

                                                question_part_9,question_part_10,question_part_11,

                                                question_part_12,percent_instead_of_count,response_count):

    """Plot multiple selection multiple choice question with 10 options"""    

    print(question_number,':',data_with_question_on_top[question_part_1][0])

    if percent_instead_of_count == True:

        df1 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_1, question_number,response_count)

        df2 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_2, question_number,response_count)

        df3 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_3, question_number,response_count)

        df4 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_4, question_number,response_count)

        df5 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_5, question_number,response_count)

        df6 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_6, question_number,response_count)

        df7 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_7, question_number,response_count)

        df8 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_8, question_number,response_count)

        df9 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_9, question_number,response_count)

        df10 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_10, question_number,response_count)

        df11 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_11, question_number,response_count)

        df12 = return_percentage_multiple_choice_multiple_choice_selection(data, question_part_12, question_number,response_count)

    else:

        df1 = return_count(data, question_part_1)

        df2 = return_count(data, question_part_2)

        df3 = return_count(data, question_part_3)

        df4 = return_count(data, question_part_4)

        df5 = return_count(data, question_part_5)

        df6 = return_count(data, question_part_6)

        df7 = return_count(data, question_part_7)

        df8 = return_count(data, question_part_8)

        df9 = return_count(data, question_part_9)

        df10 = return_count(data, question_part_10)

        df11 = return_count(data, question_part_11)

        df12 = return_count(data, question_part_12)

    trace1 = go.Bar(

                    x = df1.index,

                    y = df1[question_part_1],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df1.index)

    trace2 = go.Bar(

                    x = df2.index,

                    y = df2[question_part_2],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df2.index)    

    trace3 = go.Bar(

                    x = df3.index,

                    y = df3[question_part_3],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df3.index)   

    trace4 = go.Bar(

                    x = df4.index,

                    y = df4[question_part_4],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df4.index)   

    trace5 = go.Bar(

                    x = df5.index,

                    y = df5[question_part_5],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df5.index)       

    trace6 = go.Bar(

                    x = df6.index,

                    y = df6[question_part_6],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df6.index)     

    trace7 = go.Bar(

                    x = df7.index,

                    y = df7[question_part_7],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df7.index)

    trace8 = go.Bar(

                    x = df8.index,

                    y = df8[question_part_8],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df8.index)   

    trace9 = go.Bar(

                    x = df9.index,

                    y = df9[question_part_9],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df9.index)   

    trace10 = go.Bar(

                    x = df10.index,

                    y = df10[question_part_10],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df10.index)   

    trace11 = go.Bar(

                    x = df11.index,

                    y = df11[question_part_11],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df11.index)   

    trace12 = go.Bar(

                    x = df12.index,

                    y = df12[question_part_12],

                    name = "Kaggle Survey 2019",

                    marker = dict(color = 'blue',

                                 line=dict(color='black',width=1.5)),

                    text = df12.index)   

    

    data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]

    layout = go.Layout(barmode = "group",title=title, yaxis= dict(title=y_axis_title),showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)    

data = responses_only

data_with_question_on_top = multiple_choice

y_axis_title='Percent of Respondents'

in_order = ["No formal education past high school","Some college/university study without earning a bachelor’s degree",

            "Bachelor’s degree","Master’s degree","Doctoral degree", "Professional degree","I prefer not to answer"]

question_number = 'Q4'

title='Percent of Respondents per Education Level'

plot_unsorted_multiple_choice(question_number,data,data_with_question_on_top,title,y_axis_title,

                              in_order,percent_instead_of_count=True,response_count=782)
in_order = ['$0-999',

                        '1,000-1,999',

                        '2,000-2,999',

                        '3,000-3,999',

                        '4,000-4,999',

                        '5,000-7,499',

                        '7,500-9,999',

                        '10,000-14,999',

                        '15,000-19,999',

                        '20,000-24,999',

                        '25,000-29,999',

                        '30,000-39,999',

                        '40,000-49,999',

                        '50,000-59,999',

                        '60,000-69,999',

                        '70,000-79,999',

                        '80,000-89,999',

                        '90,000-99,999',

                        '100,000-124,999',

                        '125,000-149,999',

                        '150,000-199,999',

                        '200,000-249,999',

                        '250,000-299,999',

                        '300,000-500,000',

                        '> $500,000']



question_number = 'Q10'

title='Percent of Respondents per Salary Range'

plot_unsorted_multiple_choice(question_number,data,data_with_question_on_top,title,y_axis_title,

                              in_order,percent_instead_of_count=True,response_count=709)
question_number = 'Q28'

title='Percent of Respondents per Machine Learning Framework'

plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                           question_part_1 = 'Q28_Part_1',

                                            question_part_2 = 'Q28_Part_5',

                                            question_part_3 = 'Q28_Part_4',

                                            question_part_4 = 'Q28_Part_2',

                                            question_part_5 = 'Q28_Part_3',

                                            question_part_6 = 'Q28_Part_6',

                                            question_part_7 = 'Q28_Part_7',

                                            question_part_8 = 'Q28_Part_8',

                                            question_part_9 = 'Q28_Part_9',

                                            question_part_10 = 'Q28_Part_10',

                                            question_part_11 = 'Q28_Part_11',

                                            question_part_12 = 'Q28_Part_12',

                                            percent_instead_of_count=True,

                                            response_count=661)
question_number = 'Q20'

title='Percent of Respondents per Data Visualization Library'

plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                           question_part_1 = 'Q20_Part_2',

                                            question_part_2 = 'Q20_Part_8',

                                            question_part_3 = 'Q20_Part_1',

                                            question_part_4 = 'Q20_Part_6',

                                            question_part_5 = 'Q20_Part_4',

                                            question_part_6 = 'Q20_Part_7',

                                            question_part_7 = 'Q20_Part_5',

                                            question_part_8 = 'Q20_Part_10',

                                            question_part_9 = 'Q20_Part_9',

                                            question_part_10 = 'Q20_Part_3',

                                            question_part_11 = 'Q20_Part_11',

                                            question_part_12 = 'Q20_Part_12',

                                            percent_instead_of_count=True,

                                            response_count=666)
question_number = 'Q29'

title='Percent of Respondents per Cloud Computing Service'

plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                          question_part_1 = 'Q29_Part_2',

                                           question_part_2 = 'Q29_Part_1',

                                           question_part_3 = 'Q29_Part_3',

                                           question_part_4 = 'Q29_Part_4',

                                           question_part_5 = 'Q29_Part_6',

                                           question_part_6 = 'Q29_Part_10',

                                           question_part_7 = 'Q29_Part_7',

                                           question_part_8 = 'Q29_Part_9',

                                           question_part_9 = 'Q29_Part_8',

                                           question_part_10 = 'Q29_Part_5',

                                           question_part_11 = 'Q29_Part_11',

                                           question_part_12 = 'Q29_Part_12',

                                           percent_instead_of_count=True,

                                           response_count=510)
question_number = 'Q18'

title='Percent of Respondents per Programming Language'

plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                           question_part_1 = 'Q18_Part_1',

                                            question_part_2 = 'Q18_Part_3',

                                            question_part_3 = 'Q18_Part_2',

                                            question_part_4 = 'Q18_Part_9',

                                            question_part_5 = 'Q18_Part_7',

                                            question_part_6 = 'Q18_Part_6',

                                            question_part_7 = 'Q18_Part_5',

                                            question_part_8 = 'Q18_Part_10',

                                            question_part_9 = 'Q18_Part_4',

                                            question_part_10 = 'Q18_Part_8',

                                            question_part_11 = 'Q18_Part_11',

                                            question_part_12 = 'Q18_Part_12',

                                            percent_instead_of_count=True,

                                            response_count=667)
question_number = 'Q13'

title='Percent of Respondents per Learning Platform'

plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                           question_part_1 = 'Q13_Part_2',

                                            question_part_2 = 'Q13_Part_10',

                                            question_part_3 = 'Q13_Part_4',

                                            question_part_4 = 'Q13_Part_8',

                                            question_part_5 = 'Q13_Part_1',

                                            question_part_6 = 'Q13_Part_6',

                                            question_part_7 = 'Q13_Part_3',

                                            question_part_8 = 'Q13_Part_7',

                                            question_part_9 = 'Q13_Part_9',

                                            question_part_10 = 'Q13_Part_5',

                                            question_part_11 = 'Q13_Part_11',

                                            question_part_12 = 'Q13_Part_12',

                                            percent_instead_of_count=True,

                                            response_count=685)
question_number = 'Q12'

title='Percent of Respondents per ML Media Source'

plot_multiple_choice_12_multiple_selections(data,data_with_question_on_top,title,y_axis_title,question_number,

                                            question_part_1 = 'Q12_Part_8',

                                            question_part_2 = 'Q12_Part_4',

                                            question_part_3 = 'Q12_Part_9',

                                            question_part_4 = 'Q12_Part_1',

                                            question_part_5 = 'Q12_Part_3',

                                            question_part_6 = 'Q12_Part_6',

                                            question_part_7 = 'Q12_Part_7',

                                            question_part_8 = 'Q12_Part_5',

                                            question_part_9 = 'Q12_Part_2',

                                            question_part_10 = 'Q12_Part_10',

                                            question_part_11 = 'Q12_Part_11',

                                            question_part_12 = 'Q12_Part_12',

                                            percent_instead_of_count=True,

                                            response_count=691)                                            