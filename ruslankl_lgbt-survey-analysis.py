!pip install emoji-country-flag
import numpy as np 

import pandas as pd 

import os

import flag

import pycountry

import json



from plotly.utils import PlotlyJSONEncoder

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode

import colorlover as cl



from IPython.display import clear_output



pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_colwidth', 1000)

init_notebook_mode(connected=True)
def plot_table(df):

    values = []

    for col_name in df.columns:

        values.append(df[col_name])

        

    trace0 = go.Table(

        header = dict(

            values = ['<b>'+x.upper()+'</b>' for x in df.columns],

            line = dict(color = 'black'),

            fill = dict(color = 'yellow'),

            align = ['center'],

            font = dict(color = 'black', size = 9)

        ),

        cells = dict(

            values = values,

            align = 'center',

            font = dict(color = 'black', size = 11)

        ))



    fig = go.Figure([trace0])

    return fig

    

    

def calculate_and_plot_total_score(df, title):

    temp_df = df[df['CountryName'] != 'Average']

    temp_df = temp_df.merge(subset_size_df, how='left', on=['CountryName', 'CountryFlag', 'subset'])

    temp_df['perc_ratio'] = temp_df['percentage']*temp_df['subset_weight'] / 100

    

    # calculate the score

    temp_df[title+' Score'] = temp_df['weight'] * temp_df['perc_ratio']

    

    # get the average

    score_df = temp_df.groupby(['CountryName', 'CountryFlag'], as_index=False)[[title+' Score']].mean()

    score_df = score_df[['CountryFlag', 'CountryName', title+' Score']]

    score_df = score_df.sort_values([title+' Score'], ascending=False).reset_index(drop=True)

    score_df[title+' Rank'] = score_df[title+' Score'].rank(method='dense', ascending=False).astype('int')

    score_df[title+' Score'] = score_df[title+' Score'].apply(lambda x: round(x,4))

        

    # plot the score_df

    fig = plot_table(score_df[[title+' Rank', 'CountryFlag', 'CountryName', title+' Score']])

    return fig, score_df





def show_InEx_questions(df, included=True):

    temp_df = df.groupby(['question_label', 'question_code'], as_index=False)[['weight']].max()

    if included:

        display(pd.DataFrame(

            temp_df[['question_label', 'question_code']][~temp_df['weight'].isnull()].drop_duplicates(), 

            columns=['question_label', 'question_code']).rename(columns={'question_label': 'Included Questions'})

               )

    else:

        display(pd.DataFrame(

            temp_df[['question_label', 'question_code']][temp_df['weight'].isnull()].drop_duplicates(), 

            columns=['question_label', 'question_code']).rename(columns={'question_label': 'Excluded Questions'})

               )
# get the list of countries ID (like Germany - DE)

countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_2



# data inport

daily_life_df = pd.read_csv('../input/LGBT_Survey_DailyLife.csv')

rights_awareness_df = pd.read_csv('../input/LGBT_Survey_RightsAwareness.csv')

violence_harassment_df = pd.read_csv('../input/LGBT_Survey_ViolenceAndHarassment.csv')

discrimination_df = pd.read_csv('../input/LGBT_Survey_Discrimination.csv')

subset_size_df = pd.read_csv('../input/LGBT_Survey_SubsetSize.csv')
# data cleaning

def clean_data(df):

    df.rename(columns={'CountryCode': 'CountryName'}, inplace=True)

    codes = [countries.get(country, 'Unknown code') for country in df['CountryName']]

    df['CountryID'] = codes

    df.loc[df['CountryName'] == 'Czech Republic', 'CountryID'] = 'CZ'

    df['CountryFlag'] = df['CountryID'].apply(lambda x: x+flag.flagize(':'+x+':'))

    df.loc[df['notes'] == ' [1] ', 'notes'] = '[1]'

    df.loc[df['notes'] == '[1]', 'percentage'] = np.NaN

    df['percentage'] = df['percentage'].astype('float')

    return df





daily_life_df = clean_data(daily_life_df)

rights_awareness_df = clean_data(rights_awareness_df)

violence_harassment_df = clean_data(violence_harassment_df)

discrimination_df = clean_data(discrimination_df)
overview_df = pd.DataFrame({

    'Data set': [], 

    'Total number of questions': [], 

    'Number of records with small sample size': [],

    '% of total records(0)': [],

    'Number of missing values due to the small sample size': [],

    '% of total records(1)': [],

    'Number of missing values': [],

    '% of total records(2)': []

})





def data_overview(df, df_name=''):

    global overview_df

    temp = [[

        df_name, 

        df['question_label'].nunique(), 

        np.sum(df['notes'] == '[0]'),

        round(np.sum(df['notes'] == '[0]') * 100/ len(df), 1),

        np.sum(df['notes'] == '[1]'),

        round(np.sum(df['notes'] == '[1]') * 100/ len(df), 1),

        np.sum(df['notes'] == '[2]'),

        round(np.sum(df['notes'] == '[2]') * 100/ len(df), 1),

    ]]

    temp_df = pd.DataFrame(temp, columns=overview_df.columns)

    overview_df = overview_df.append(temp_df)

    

    

data_overview(daily_life_df, df_name='Daily Life')

data_overview(rights_awareness_df, df_name='Rights Awareness')

data_overview(violence_harassment_df, df_name='Violence and Harassment')

data_overview(discrimination_df, df_name='Discrimination')

display(overview_df.set_index(['Data set']))
subset_size_df.rename(columns={'Lesbian women': 'Lesbian', 'Gay men':'Gay'}, inplace=True)



for column in subset_size_df.loc[:,"Lesbian":].columns:

    subset_size_df[column + ' weight'] = subset_size_df[column] / subset_size_df['N']



    

subset_size_df = subset_size_df.round(2)

subset_size_df = subset_size_df.merge(daily_life_df[['CountryID', 'CountryFlag', 'CountryName']], how='left')

subset_size_df = subset_size_df.drop_duplicates().reset_index(drop=True)
clmnstkp = ['Lesbian', 'Gay', 'Bisexual women', 'Bisexual men', 'Transgender']



    

subset_size_df.loc[subset_size_df['CountryID'] == 'EU Total', 'CountryName'] = 'EU Total'

subset_size_df.loc[subset_size_df['CountryID'] == 'EU Total', 'CountryFlag'] = 'EU Total'



subset_size_df = subset_size_df[['CountryName', 'CountryFlag', 'N'] + clmnstkp + [x + ' weight' for x in clmnstkp]]
fig = plot_table(subset_size_df)



fig.show()
subset_size_df = subset_size_df[['CountryName', 'CountryFlag'] + [x + ' weight' for x in clmnstkp]]



subset_size_df = pd.melt(

    subset_size_df, 

    id_vars=['CountryName', 'CountryFlag'], 

    value_vars=list(subset_size_df.columns[2:]),

    var_name='subset', 

    value_name='subset_weight'

).sort_values(['CountryName'])



subset_size_df['subset'] = subset_size_df['subset'].apply(lambda x: x.replace(' weight', ''))
def set_WidespreadRare_weight(df, questions_list, rare_negative=False):

    if rare_negative:

        weight = -1

    else:

        weight = 1

    for quesID in questions_list:

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Very widespread'), 'weight'] = -weight

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Fairly widespread'), 'weight'] = -weight/2

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Fairly rare'), 'weight'] = weight/2

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Very rare'), 'weight'] = weight

        



def set_YesNo_weight(df, questions_list, yes_negative=False):

    if yes_negative:

        weight = -1

    else:

        weight = 1

    for quesID in questions_list:

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Yes'), 'weight'] = weight

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'No'), 'weight'] = -weight

        

        

def set_AlwaysNever_weight(df, questions_list, alsways_negative=False):

    if alsways_negative:

        weight = -1

    else:

        weight = 1

    for quesID in questions_list:

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Always'), 'weight'] = weight

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Often'), 'weight'] = weight/2

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Rarely'), 'weight'] = -weight/2

        df.loc[(df['question_code'] == quesID) & (df['answer'] == 'Never'), 'weight'] = -weight
daily_life_df['weight'] = np.NaN

daily_life_df.loc[daily_life_df['answer'] == 'Don`t know', 'weight'] = np.NaN



set_WidespreadRare_weight(

    df=daily_life_df,

    questions_list=[

        'b1_a', 'b1_b', 'b1_c', 'b1_d', 'c1a_a', 'c1a_b', 'c1a_c', 'c1a_d', ''

    ],

    rare_negative=False

)

set_WidespreadRare_weight(

    df=daily_life_df,

    questions_list=[

        'b1_e', 'b1_g', 'b1_h', 'b1_i'

    ],

    rare_negative=True

)



daily_life_df.loc[(daily_life_df['question_code'] == 'g4_a') & (daily_life_df['answer'] == 'Never happened in the last sixth months'), 'weight'] = 1

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_a') & (daily_life_df['answer'] == 'Happened only once in the last six months'), 'weight'] = 0.5

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_a') & (daily_life_df['answer'] == '2-5 times in the last six months'), 'weight'] = -0.5

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_a') & (daily_life_df['answer'] == '6 times or more in the last six months'), 'weight'] = -1



daily_life_df.loc[(daily_life_df['question_code'] == 'g4_b') & (daily_life_df['answer'] == 'Never happened in the last sixth months'), 'weight'] = 1

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_b') & (daily_life_df['answer'] == 'Happened only once in the last six months'), 'weight'] = 0.5

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_b') & (daily_life_df['answer'] == '2-5 times in the last six months'), 'weight'] = -0.5

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_b') & (daily_life_df['answer'] == '6 times or more in the last six months'), 'weight'] = -1



daily_life_df.loc[(daily_life_df['question_code'] == 'g4_c') & (daily_life_df['answer'] == 'Never happened in the last sixth months'), 'weight'] = 1

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_c') & (daily_life_df['answer'] == 'Happened only once in the last six months'), 'weight'] = 0.5

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_c') & (daily_life_df['answer'] == '2-5 times in the last six months'), 'weight'] = -0.5

daily_life_df.loc[(daily_life_df['question_code'] == 'g4_c') & (daily_life_df['answer'] == '6 times or more in the last six months'), 'weight'] = -1



daily_life_df.loc[(daily_life_df['question_code'] == 'h15') & (daily_life_df['answer'] == 'Yes'), 'weight'] = -1

daily_life_df.loc[(daily_life_df['question_code'] == 'h15') & (daily_life_df['answer'] == 'No'), 'weight'] = 1

daily_life_df.loc[(daily_life_df['question_code'] == 'h15') & (daily_life_df['answer'] == 'I did not need or use any benefits or services'), 'weight'] = np.NaN
fig, dl_scores = calculate_and_plot_total_score(daily_life_df, title='Daily Life')

fig.show()
rights_awareness_df['weight'] = np.NaN

rights_awareness_df.loc[rights_awareness_df['answer'] == 'Don`t know', 'weight'] = np.NaN

rights_awareness_df.loc[rights_awareness_df['answer'] == 'No', 'weight'] = -1

rights_awareness_df.loc[rights_awareness_df['answer'] == 'Yes', 'weight'] = 1
fig, ra_scores = calculate_and_plot_total_score(rights_awareness_df, title='Rights Awareness')

fig.show()
discrimination_df['weight'] = np.NaN



discrimination_df.loc[discrimination_df['answer'] == 'Don`t know', 'weight'] = np.NaN

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'None of the above'), 'weight'] = 0

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'I have never accessed healthcare services'), 'weight'] = 0

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Difficulty in gaining access to healthcare'), 'weight'] = -1

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Having to change general practitioners or other specialists due to their negative reaction'), 'weight'] = -1

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Receiving unequal treatment when dealing with medical staff'), 'weight'] = -1

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Foregoing treatment for fear of discrimination or intolerant reactions'), 'weight'] = -1

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Specific needs ignored (not taken into account)'), 'weight'] = -1

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Inappropriate curiosity'), 'weight'] = -1

discrimination_df.loc[(discrimination_df['question_code'] == 'c10') & (discrimination_df['answer'] == 'Pressure or being forced to undergo any medical or psychological test'), 'weight'] = -1



set_YesNo_weight(

    df=discrimination_df,

    questions_list=[

        'c2a_a', 'c2a_b', 'c2a_c', 'c2a_d', 'c2_b', 'c2_c', 'c4_a', 'c4_b', 

        'c4_c', 'c4_d', 'c4_e', 'c4_f', 'c4_g', 'c4_h', 'c4_i', 'c4_j', 'c4_k', 'discrim1yr'

    ],

    yes_negative=True

)



set_AlwaysNever_weight(

    df=discrimination_df,

    questions_list=[

        'c8a_b', 'c8a_c', 'c8a_d', 'c8a_e', 'c8a_f', 'c9_b', 'c9_c', 'c9_d', 'c9_e'

    ],

    alsways_negative=True

)



set_AlwaysNever_weight(

    df=discrimination_df,

    questions_list=[

        'c8a_a', 'c9_a'

    ],

    alsways_negative=False

)
fig, discr_scores = calculate_and_plot_total_score(discrimination_df, title='Discrimination')

fig.show()
violence_harassment_df['weight'] = np.NaN



violence_harassment_df.loc[violence_harassment_df['answer'] == 'Don`t know', 'weight'] = np.NaN

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'e1') & (violence_harassment_df['answer'] == 'I do not have a same-sex partner'), 'weight'] = np.NaN

set_YesNo_weight(

    df=violence_harassment_df,

    questions_list=[

        'e1', 'e2', 'f1_a', 'f1_b', 'fa1_5', 'fa2_5', 'fb1_5', 'fb2_5'

    ],

    yes_negative=True

)

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'More than ten times'), 'weight'] = -1

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'Six to ten times'), 'weight'] = -0.86

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'Five times'), 'weight'] = -0.71

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'Four times'), 'weight'] = -0.57

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'Three times'), 'weight'] = -0.43

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'Twice'), 'weight'] = -0.29

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fa1_3') & (violence_harassment_df['answer'] == 'Once'), 'weight'] = -0.14



violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'More than ten times'), 'weight'] = -1

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'Six to ten times'), 'weight'] = -0.86

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'Five times'), 'weight'] = -0.71

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'Four times'), 'weight'] = -0.57

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'Three times'), 'weight'] = -0.43

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'Twice'), 'weight'] = -0.29

violence_harassment_df.loc[(violence_harassment_df['question_code'] == 'fb1_3') & (violence_harassment_df['answer'] == 'Once'), 'weight'] = -0.14
fig, violhar_scores = calculate_and_plot_total_score(violence_harassment_df, title='Violence and Harassment')

fig.show()
overall_rang = pd.merge(

    dl_scores[['CountryFlag', 'CountryName', 'Daily Life Rank']],

    violhar_scores[['CountryFlag', 'CountryName', 'Violence and Harassment Rank']]).merge(

    discr_scores[['CountryFlag', 'CountryName', 'Discrimination Rank']]).merge(

    ra_scores[['CountryFlag', 'CountryName', 'Rights Awareness Rank']])



overall_rang['Average Rank'] = overall_rang.mean(axis=1)

overall_rang = overall_rang.sort_values(['Average Rank'], ascending=True).reset_index(drop=True)



overall_rang['Total Rank'] = overall_rang['Average Rank'].rank(method='dense', ascending=True).astype('int')



fig = plot_table(overall_rang)

fig.show()
subset_df = daily_life_df[daily_life_df['question_label'] == 'All things considered, how satisfied would you say you are with your life these days? *'].copy()

subset_df = subset_df.merge(subset_size_df, how='left', on=['CountryName', 'CountryFlag', 'subset'])

subset_df['answer'] = subset_df['answer'].astype('int') / 10

subset_df['percentage'] = subset_df['percentage'] * subset_df['subset_weight'] / 100

subset_df['Daily Life Score'] = subset_df['answer'] * subset_df['percentage'] 

subset_df = subset_df.groupby(['CountryName'], as_index=False)['Daily Life Score'].mean()

subset_df.columns = ['CountryName', 'Score']

subset_df = subset_df.merge(dl_scores[['CountryName', 'CountryFlag']])

subset_df['Score'] = subset_df['Score'].apply(lambda x: round(x,4))

subset_df = subset_df.merge(overall_rang[['CountryName', 'Total Rank']])

subset_df = subset_df[['CountryFlag', 'CountryName', 'Score', 'Total Rank']].sort_values(['Score'], ascending=False)



trace0 = go.Choropleth(

    colorscale = 'Greens', #'YlOrRd',

    autocolorscale = False,

    reversescale=True,

    locations = subset_df['CountryName'],

    text=subset_df['CountryFlag'],

    z = subset_df['Score'],

    locationmode = 'country names',

    colorbar = go.choropleth.ColorBar(

        title = '<b>Score</b>')

)



layout = go.Layout(

    title='<b>How satisfied would you say you are with your life these days?</b>',

    geo = go.layout.Geo(

        scope='europe',

        showlakes=False),

)



fig = go.Figure(data=[trace0], layout=layout)

fig.show()
subset_df['Satisfaction Rank'] = subset_df['Score'].rank(method='dense', ascending=False).astype('int')

subset_df['Rank Diff'] = subset_df['Satisfaction Rank'] - subset_df['Total Rank'] 

fig = plot_table(subset_df)

fig.show()
openess_df = daily_life_df[daily_life_df['question_label'] == '4 levels of being open about LGBT background *']

openess_df = openess_df.merge(dl_scores[['CountryName', 'CountryFlag']])

openess_df = openess_df.merge(overall_rang[['CountryName', 'Total Rank']])

openess_df = openess_df.sort_values(['Total Rank'], ascending=False)





colors = cl.scales['5']['seq']['Greens']

colors = colors[1:]

answers = ['Never Open', 'Rarely Open', 'Fairly open', 'Very open']

data = []

buttons = []



for (i,subset) in enumerate(openess_df['subset'].unique()):

    if i == 0:

        visible = True

    else:

        visible = False

    subset_df = openess_df[openess_df['subset'] == subset]

    subset_df = subset_df.sort_values(['Total Rank'], ascending=False)

    sum_df = subset_df.groupby(['CountryName'], as_index=False)[['percentage']].sum()

    sum_df.rename(columns={'percentage': 'sum_perc'}, inplace=True)

    subset_df = subset_df.merge(sum_df, how='left')

    for (j,ans_opt) in enumerate(answers):

        trace = go.Bar(

            name=ans_opt, 

            y=subset_df['CountryName'][subset_df['answer'] == ans_opt] +' ('+ subset_df['CountryFlag'][subset_df['answer'] == ans_opt] +')', 

            x=subset_df['percentage'][subset_df['answer'] == ans_opt]/subset_df['sum_perc'][subset_df['answer'] == ans_opt],

            orientation='h',

            legendgroup=ans_opt,

            marker=dict(color=colors[j]),

            visible=visible

        )

        data.append(trace)

        

    visible_list = [False] * (len(subset_df['answer'].unique()) * len(openess_df['subset'].unique()) + 4)

    visible_list[i*4:i*4+4] = [True] * 4

    buttons_temp = dict(

        label = subset,

        method = 'update',

        args = [

            {'visible': visible_list}

        ]

    )

    buttons.append(buttons_temp)



temp_df = openess_df.groupby(['CountryName', 'CountryFlag', 'answer', 'Total Rank'], as_index=False)[['percentage']].mean()

temp_df = temp_df.sort_values(['Total Rank'], ascending=False)

sum_df = openess_df.groupby(['CountryName','answer'], as_index=False)[['percentage']].mean().groupby(['CountryName'], as_index=False)[['percentage']].sum()

sum_df.rename(columns={'percentage': 'sum_perc'}, inplace=True)

temp_df = temp_df.merge(sum_df, how='left')

for (j,ans_opt) in enumerate(answers):

    trace = go.Bar(

        name=ans_opt, 

        y=temp_df['CountryName'][temp_df['answer'] == ans_opt] +' ('+ temp_df['CountryFlag'][temp_df['answer'] == ans_opt]+')', 

        x=temp_df['percentage'][temp_df['answer'] == ans_opt]/temp_df['sum_perc'][temp_df['answer'] == ans_opt],

        orientation='h',

        legendgroup=ans_opt,

        marker=dict(color=colors[j]),

        visible=visible

    )

    data.append(trace)



visible_list = [False] * (len(subset_df['answer'].unique()) * len(openess_df['subset'].unique()) + 4)

visible_list[-4:] = [True] * 4

buttons_temp = dict(

    label = 'All',

    method = 'update',

    args = [

        {'visible': visible_list}

    ]

)

buttons.append(buttons_temp)



updatemenus = list([

    dict(type="buttons",

         active=0,

         buttons=buttons,

         direction = "left",

         x=0.1,

         xanchor="left",

         y=1.1,

         yanchor="top"

        )

])



layout = go.Layout(

    updatemenus=updatemenus,

    annotations=[     

        go.layout.Annotation(

            text="<b>Subset:</b>", 

            showarrow=False,

            x=0, y=1.08, 

            yref="paper", align="left"

        )

    ],

    margin=dict(l=200, t=200),

    height=700,

    barmode='stack',

    title='<b>4 levels of being open about LGBT background</b><br><i>Taken from EU LGBT survey results (2012)</i>',

    xaxis=dict(title='<b>Ratio</b>')

    )

  

# layout = json.dumps(layout, cls=PlotlyJSONEncoder)

fig = go.Figure(data=data, layout=layout)

fig.show()
data = []

buttons = []

for (j,ans_opt) in enumerate(answers):

    if j == 0:

        visible = True

    else:

        visible = False

    x=temp_df['Total Rank'][temp_df['answer'] == ans_opt]

    y=temp_df['percentage'][temp_df['answer'] == ans_opt]

    corrCoef = np.corrcoef(temp_df['Total Rank'][temp_df['answer'] == ans_opt], temp_df['percentage'][temp_df['answer'] == ans_opt])[0,1]

    trace0 = go.Scatter(

        name=ans_opt,

        x=x, 

        y=y,

        mode="markers",

        marker=dict(color=colors[j]),

        visible=visible

    )

    

    trace1 = go.Scatter(

        x=[x.max()*0.9],

        y=[y.max()*0.8],

        mode='text',

        text='Correlation: {}'.format(round(corrCoef,2)),

        textfont=dict(

          family='sans serif',

          size=16,

          color='#FF4136'

        ),

        name=ans_opt,

        visible=visible

  )

    data.append(trace0)

    data.append(trace1)

    

    visible_list = [False] * len(answers) * 2

    visible_list[j*2:j*2+2] = [True] * 2

    buttons_temp = dict(

        label = ans_opt,

        method = 'update',

        args = [

            {'visible': visible_list}

        ]

    )

    buttons.append(buttons_temp)



updatemenus = list([

    dict(

        type="buttons",

        active=0,

        buttons=buttons,

        direction = "left",

        x=0.3,

        xanchor="left",

        y=1.1,

        yanchor="top"

    )

])



layout = go.Layout(

    showlegend=False,

    updatemenus=updatemenus,

    annotations=[     

        go.layout.Annotation(

            text="<b>Answer Option:</b>", 

            showarrow=False,

            x=1, y=1.08, 

            yref="paper", align="left"

        )

    ],

    title='<b>Correlation between "Openess" Ratio and Total Rank</b>',

    xaxis=dict(title='<b>Country Total Rank</b>'),

    yaxis=dict(title='<b>Percent of people answered</b>')

    )

  

fig = go.Figure(data=data, layout=layout)

# layout = json.dumps(layout, cls=PlotlyJSONEncoder)

fig.show()
total_open = openess_df.groupby(['answer'], as_index=False)[['percentage']].mean()

total_open['subset'] = 'Total'

subset_open = openess_df.groupby(['subset', 'answer'], as_index=False)[['percentage']].mean()

subset_open = subset_open.append(total_open, sort=True)

subset_open['percentage'] = subset_open['percentage'].apply(lambda x: round(x,1))



data = []

for (i,ans_opt) in enumerate(subset_open['answer'].unique()):

    trace = go.Bar(

        name=ans_opt, 

        y=subset_open['subset'][subset_open['answer'] == ans_opt], 

        x=subset_open['percentage'][subset_open['answer'] == ans_opt]/100,

        orientation='h',

        legendgroup=ans_opt,

        marker=dict(color=colors[i])

    )

    data.append(trace)



layout = go.Layout(

    margin=dict(l=100),

    height=400,

    barmode='stack',

    title='<b>4 levels of being open about LGBT background</b><br><i>Total for all countries</i>',

#     xaxis=dict(title='Ratio'),

    legend_orientation="h"

)



fig = go.Figure(data=data, layout=layout)

# layout = json.dumps(layout, cls=PlotlyJSONEncoder)

fig.show()
question_codes = ['b2_a', 'b2_f', 'b2_b', 'b2_d', 'b2_c', 'b2_h', 'b2_g', 'b2_e']

comfort_df = daily_life_df[daily_life_df['question_code'].apply(lambda x: x in question_codes)].reset_index(drop=True)

comfort_df['question_label'] = comfort_df['question_label'].apply(lambda x: x.split('? ')[1])

comfort_df = comfort_df.groupby(['CountryName', 'CountryFlag', 'question_code', 'question_label', 'answer'], as_index=False)[['percentage']].mean()

comfort_df['question_label'] = comfort_df['question_label'].apply(lambda x: x.replace('lesbian, gay and bisexual', 'LGB'))

comfort_df['answer'] = comfort_df['answer'].apply(lambda x: x.replace('lesbian, gay and bisexual', 'LGB'))
comfort_df = comfort_df.merge(overall_rang[['CountryName', 'Total Rank']])

comfort_df = comfort_df.sort_values(['Total Rank'], ascending=False)



answers = ['Don`t know', 'Strongly disagree', 'Disagree', 'Current situation is fine', 'Agree', 'Strongly agree']

colors = cl.scales[str(len(answers))]['seq']['Greens']

data = []

buttons = []



for (i,question) in enumerate(comfort_df['question_label'].unique()):

    if i == 0:

        visible = True

    else:

        visible = False

    subset_df = comfort_df[comfort_df['question_label'] == question]

    subset_df = subset_df.sort_values(['Total Rank'], ascending=False)

    sum_df = subset_df.groupby(['CountryName'], as_index=False)[['percentage']].sum()

    sum_df.rename(columns={'percentage': 'sum_perc'}, inplace=True)

    subset_df = subset_df.merge(sum_df, how='left')

    for (j,ans_opt) in enumerate(answers):

        trace = go.Bar(

            name=ans_opt, 

            y=subset_df['CountryName'][subset_df['answer'] == ans_opt] +' ('+ subset_df['CountryFlag'][subset_df['answer'] == ans_opt] +')', 

            x=subset_df['percentage'][subset_df['answer'] == ans_opt]/subset_df['sum_perc'][subset_df['answer'] == ans_opt],

            orientation='h',

            legendgroup=ans_opt,

            marker=dict(color=colors[j]),

            visible=visible

        )

        data.append(trace)

        

    visible_list = [False] * (len(answers) * len(comfort_df['question_label'].unique()))

    visible_list[i*6:i*6+6] = [True] * 6

    buttons_temp = dict(

        label = question,

        method = 'update',

        args = [

            {'visible': visible_list}

        ]

    )

    buttons.append(buttons_temp)



updatemenus = list([

    dict(type="dropdown",

         active=0,

         showactive=True,

         buttons=buttons,

         direction = "down",

         x=0.0,

         xanchor="left",

         y=1.1,

         yanchor="top"

        )

])



layout = go.Layout(

    updatemenus=updatemenus,

    annotations=[     

        go.layout.Annotation(

            text="Question:", 

            showarrow=False,

            x=0.01, y=1.08, 

            yref="paper", align="left"

        )

    ],

    margin=dict(l=200, t=120),

    height=700,

    barmode='stack',

    title='<b>What would allow you to be more comfortable living as a LGB person?</b>',

    xaxis=dict(title='<b>Ratio</b>')

    )

  

fig = go.Figure(data=data, layout=layout)

# layout = json.dumps(layout, cls=PlotlyJSONEncoder)

fig.show()
comfort_total = comfort_df.groupby(['question_label', 'answer', 'question_code'], as_index=False)[['percentage']].mean()

comfort_total = comfort_total.sort_values(['question_code'], ascending=False)

comfort_total['percentage'] = comfort_total['percentage'].apply(lambda x: round(x,1))



data = []

for (i,ans_opt) in enumerate(answers):

    trace = go.Bar(

        name=ans_opt, 

        y=comfort_total['question_code'][comfort_total['answer'] == ans_opt], 

        x=comfort_total['percentage'][comfort_total['answer'] == ans_opt]/100,

        orientation='h',

        legendgroup=ans_opt,

        marker=dict(color=colors[i])

    )

    data.append(trace)



layout = go.Layout(

    height=400,

#     margin=dict(l=100),

    barmode='stack',

    title='<b>What would allow you to be more comfortable living as a LGB?</b><br><i>Total for all countries</i>',

#     xaxis=dict(title='Ratio'),

    yaxis=dict(

        title='<b>Question Code</b>',

        automargin=True),

    legend_orientation="h"

)



fig = go.Figure(data=data, layout=layout)

fig.show()



fig = plot_table(comfort_total[['question_code', 'question_label']].drop_duplicates().sort_values(['question_code']))

fig.show()