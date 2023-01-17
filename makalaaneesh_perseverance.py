import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

mcr = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
all_questions = questions.loc[0]

categorical_columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q10','Q11','Q15','Q23']
# Helper methods



from collections import Counter

import plotly.graph_objects as go



def isint(v):

    try:

        int(v)

        return True

    except:

        return False

    

def get_value_counts_of_mcr(df, question, percentages=False, ignore_numbers_in_answers=True):

    """

    Returns a DataFrame of index = labels(corresponding to different answers for the question),

                           count = label count 

    DataFrame is sorted in descending order by count

                           

    @param percentages: bool. If True, return percentages instead of counts

    @param ignore_numbers_in_answers: ignore if the answer to the question is a number

    """

    q_options = [col for col in df.columns if col.startswith(question)]

    answer_values = {}

    for q_option in q_options:

        q_option_value_counts = df[q_option].value_counts()

        for answer, answer_count in q_option_value_counts.items():

            if ignore_numbers_in_answers:

                if isint(answer):

                    continue

            if 'OTHER_TEXT' in q_option and answer in ['-1',-1]:

                continue

            if answer in ['None', None]:

                answer = 'Other'

            answer_values[answer] = answer_values.get(answer,0) +  answer_count



    if percentages:

        data_length = sum(answer_values.values())

        answer_values = {ans: (float(val)*100)/float(data_length) 

                         for ans, val in answer_values.items()}

    return pd.DataFrame.from_dict(answer_values,

                                  orient='index',

                                  columns=['count']).sort_values(by='count',

                                                                 ascending=False)



def plot_pie(title, labels, values, colors=None, line=None, **kwargs):

    fig = go.Figure(data=[go.Pie(labels=labels,

                             values=values,

                                title=title,

                                showlegend=False,

                                **kwargs)])

    fig.update_traces(hoverinfo='label+percent', textinfo='label',

                      textfont_size=15,

                      marker=dict(

                          colors=colors,

                          line=line,

                      )

                     )

    fig.show()

    

    

def plot_bar(title, x, y, ylabel, xaxis_tickangle=None):

    fig = go.Figure()

    fig.add_trace(go.Bar(x=x,

                        y=y,

                        textposition='auto',))

    fig.update_layout(title=title,

                      yaxis=dict(title=ylabel),

                     xaxis_tickangle=xaxis_tickangle)

    fig.show()

actual_data = mcr[1:]
country_counts = actual_data['Q3'].value_counts()



colors = ['aliceblue']* (len(country_counts.index) - 1) +['crimson'] # Highlighting Saudi Arabia's share

plot_pie("Country Composition of respondents", country_counts.index, country_counts.values,

         colors=colors,

        line=dict(color='#000000', width=0.05))
mcr_saudi = mcr[mcr['Q3'] == "Saudi Arabia"]
roles_value_counts = mcr_saudi['Q5'].value_counts()

plot_pie("Roles of Saudis", roles_value_counts.index, roles_value_counts.values)
ml_frameworks = get_value_counts_of_mcr(mcr_saudi, 'Q28', percentages=True)[:7]



plot_bar("Top ML frameworks used by Saudis on a regular basis",

        x=ml_frameworks.index,

        y=ml_frameworks['count'].values,

        ylabel="percentages")

startups = mcr[mcr['Q6'] == '0-49 employees']
startups_ml_algos = get_value_counts_of_mcr(startups, 'Q24', ignore_numbers_in_answers=True)



plot_pie("Popular ML algorithms at Startups",

         startups_ml_algos.index, startups_ml_algos['count'].values)
money_spent = get_value_counts_of_mcr(startups, 'Q11', percentages=True)



indices = [idx.replace("-", "to")

          for idx in money_spent.index]

plot_bar("Money they spend on ML/ cloud computing products",

        x=indices,

        y=money_spent['count'].values,

        ylabel="percentages",

        xaxis_tickangle=-45)
longest_coders = pd.concat([mcr.iloc[0:1] ,mcr[mcr['Q15'] == "20+ years"]])

longest_coders_highest_earners = pd.concat([longest_coders.iloc[0:1] ,

                                            longest_coders[longest_coders['Q10'] == "> $500,000"]])
programming_languages_used = get_value_counts_of_mcr(longest_coders_highest_earners[1:], 'Q18')



plot_pie("Programming languages used by those coding since +20 years and earning > 500k USD",

        programming_languages_used.index, programming_languages_used['count'].values)
veterans_programming_languages_advocated = get_value_counts_of_mcr(longest_coders_highest_earners[1:], 'Q19')



plot_pie("Programming languages advocated by high-earning veterans for aspiring data scientists",

        programming_languages_used.index, programming_languages_used['count'].values)
wannabes = mcr[mcr['Q15'] == '< 1 years']
top_5_country_share = wannabes['Q3'].value_counts()[:10]



plot_bar("Top countries' share",

        x=top_5_country_share.index,

        y=top_5_country_share.values,

        ylabel="Counts")
wannabe_programming_languages_used = get_value_counts_of_mcr(wannabes, 'Q18', percentages=True)[:8]

veterans_programming_languages_advocated = get_value_counts_of_mcr(longest_coders, 'Q19', percentages=True)[:8]



wannabe_pls = wannabe_programming_languages_used.copy()

veterans_pla = veterans_programming_languages_advocated.copy()

wannabe_pls['category'] = "Wannabes"

veterans_pla['category'] = "Veterans advocate"

wv = pd.concat([wannabe_pls, veterans_pla])







fig = go.Figure()

fig.add_trace(go.Bar(x=wannabe_pls.index,

                     y=wannabe_pls['count'].values,

                    name="Wannabes use"))

fig.add_trace(go.Bar(x=veterans_pla.index,

                     y=veterans_pla['count'].values,

                    name="Veterans advocate"))

fig.update_layout(yaxis=dict(title="percentages"))

fig.show()
cv = get_value_counts_of_mcr(wannabes, 'Q26')

cv['category'] = "Computer Vision"

nlp = get_value_counts_of_mcr(wannabes, 'Q27')

nlp['category'] = "NLP"

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,

                   specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=cv.index, 

                    values=cv['count'].values,

                    title="Computer Vision",

                    ), 1,1)



fig.add_trace(go.Pie(labels=nlp.index, 

                    values=nlp['count'].values,

                    title="NLP",

                    ), 1, 2)

fig.update(layout_showlegend=False)

fig.update_traces(hoverinfo='label+percent+name', textinfo='label', textfont_size=15)



fig.show()