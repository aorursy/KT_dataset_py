"""

We create the groups of countries that we wish to study and compare

"""



GROUPS = {

    'United Kingdom of Great Britain and Northern Ireland' : 'EUROPE',

    'United States of America' : 'NORTH_AMERICA',

    'India' : 'ASIA',

    'Japan' : 'ASIA',

    'China' : 'ASIA',

    'Nigeria' : 'AFRICA',

    'Morocco' : 'AFRICA',

    'South Africa' : 'AFRICA',

    'Egypt' : 'AFRICA',

    'Tunisia' : 'AFRICA',

    'Kenya' : 'AFRICA',

    'Algeria' : 'AFRICA'

}



AI_LEADERS = [

    'China',

    'United States of America',

    'India',

    'Japan',

    'United Kingdom of Great Britain and Northern Ireland'

]
"""

Importing necessary modules

"""



import random

random.seed(2019)

import pandas

import seaborn

seaborn.set_style("darkgrid")

import numpy

import plotly

import plotly.colors
"""

Reading the dataset and preparing it for more convenient data analysis

"""



multiple_choice_responses = pandas.read_csv(

    filepath_or_buffer='../input/kaggle-survey-2019/multiple_choice_responses.csv', 

    skiprows=[1])



Q4_name_map = {

    'Master’s degree' : 4,

    'Professional degree' : 1,

    'Bachelor’s degree': 3,

    'Some college/university study without earning a bachelor’s degree': 2,

    'Doctoral degree': 6,

    'I prefer not to answer' : -1,

    'No formal education past high school' : 0,

}



Q23_name_map = {

    '< 1 years' : 0,

    '1-2 years' : 1,

    '2-3 years': 2,

    '3-4 years': 3,

    '4-5 years': 4,

    '5-10 years' : 5,

    '10-15 years' : 10,

    '20+ years' : 20,

}



Q15_name_map = {

    '< 1 years' : 0,

    '1-2 years' : 1,

    '3-5 years': 3,

    '5-10 years' : 5,

    '10-20 years' : 10,

    '20+ years' : 20,

    'I have never written code' : -1

}



Q11_name_map = {

    '$0 (USD)' : 0,

    '$1-$99' : 1,

    '$100-$999' : 100,

    '$1000-$9,999' : 1000,

    '$10,000-$99,999' : 10000,

    '> $100,000 ($USD)' : 100000

}



Q22_name_map = {

    'Never' : 0,

    'Once' : 1,

    '2-5 times' : 2,

    '6-24 times' : 6,

    '> 25 times' : 25

}





multiple_choice_responses['Q4'] = multiple_choice_responses['Q4'].apply(lambda x: Q4_name_map.get(x, numpy.nan))

multiple_choice_responses['Q23'] = multiple_choice_responses['Q23'].apply(lambda x: Q23_name_map.get(x, numpy.nan))

multiple_choice_responses['Q11'] = multiple_choice_responses['Q11'].apply(lambda x: Q11_name_map.get(x, numpy.nan))

multiple_choice_responses['Q15'] = multiple_choice_responses['Q15'].apply(lambda x: Q15_name_map.get(x, numpy.nan))

multiple_choice_responses['Q22'] = multiple_choice_responses['Q22'].apply(lambda x: Q22_name_map.get(x, numpy.nan))



# We will remove cases for which the country is indication is 'Other', as we have no information that we can use to put these respondents in a continent group.

multiple_choice_responses = multiple_choice_responses[multiple_choice_responses['Q3'] != 'Other']



# Add column to represent the continent to which the participants belong

multiple_choice_responses['continent'] = multiple_choice_responses['Q3'].apply(lambda x: GROUPS.get(x, 'OTHER'))



# Add column corresponding to whether country is among `AI_LEADERS`

multiple_choice_responses['ai_leader'] = multiple_choice_responses['Q3'].apply(lambda x: 'AI_LEADERS' if x in AI_LEADERS else 'NON_AI_LEADERS')



african_countries = multiple_choice_responses[multiple_choice_responses['continent'] == 'AFRICA'].copy()



ai_leaders_africa = multiple_choice_responses[(multiple_choice_responses['ai_leader'] == 'AI_LEADERS') | (multiple_choice_responses['continent'] == 'AFRICA') ].copy()

ai_leaders_africa['ai_leader'] = ai_leaders_africa['ai_leader'].apply(lambda x: x if x == 'AI_LEADERS' else 'AFRICA')
"""

We define two functions to help us generate the necessary vizualizations

"""



def create_plotly_barchart(

    dfs, 

    strat_cols, 

    quant_col, 

    width=None,

    height=None,

    names_map=None,

    percent=True, 

    xaxis=None,

    yaxis=None,

    legend_orientation='h',

    legend_x=None,

    legend_y=None,

    legend_yanchor='top',

    shared_yaxes=True,

    rows=1,

    cols=1,

    specs=None,

    subplot_titles=None,

    title=None,

    xaxis_titles=None,

    yaxis_titles=None,

    vertical_spacing=None,

    horizontal_spacing=None):

    

    assert len(dfs) == len(strat_cols) 

    if subplot_titles:

        assert len(dfs) == len(subplot_titles)

    if xaxis_titles:

        assert len(dfs) == len(xaxis_titles)

    if yaxis_titles:

        assert len(dfs) == len(yaxis_titles)

    

    # add more colours

    scales = []

    for l in list(plotly.colors.PLOTLY_SCALES.values()):

        for s in l:

            scales.append(s[1])

    random.shuffle(scales)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    colors += scales

    

    # place subplot titles

    fig = plotly.subplots.make_subplots(

        shared_yaxes=shared_yaxes,

        rows=rows, 

        cols=cols,

        specs=specs,

        subplot_titles= None if not subplot_titles else subplot_titles,

        vertical_spacing = vertical_spacing,

        horizontal_spacing = horizontal_spacing

        

    ) 

    

    # select subplot (row,col) pairs

    coordinates = []

    for r in range(1, rows+1):

        for c in range(1, cols+1):

            coordinates.append((r,c))

    

    for i, df in enumerate(dfs):

        

        df = pandas.crosstab(df[strat_cols[i]],df[quant_col])

        if percent:

            df = df.apply(lambda r: r/r.sum(), axis=1)

        

        if names_map:

            names = [names_map[n] for n in df.columns]

        

        r,c = coordinates[i]

        for j, x in enumerate(df.columns):

            

            # add graph object to figure

            fig.add_trace(plotly.graph_objs.Bar(

                name=names[j] if names_map else str(x), 

                x=df.index, 

                y=df[x],

                marker_color=colors[j],

                legendgroup=f'group-{j}',

                showlegend= (i==0),

            ),

                row=r,

                col=c)

            

            # name axes of subplots

            fig.update_xaxes(

                title_text=xaxis_titles[i],

                row=r, 

                col=c)

            fig.update_yaxes(

                title_text=yaxis_titles[i],

                row=r, 

                col=c)

            

            # impose range limit on axes

            if xaxis:

                fig.update_xaxes(

                    range=xaxis, 

                    row=r, 

                    col=c)

            if yaxis:

                fig.update_yaxes(

                    range=yaxis, 

                    row=r, 

                    col=c)

    

    # add figure title

    if title:

        fig.update_layout(title_text=title)

    

    # adjust margins and legend settings

    fig.update_layout(

        legend_orientation=legend_orientation, 

        legend_x=legend_x, 

        legend_y=legend_y,

        legend_yanchor=legend_yanchor,

        width=width,

        height=height,

        margin=plotly.graph_objs.layout.Margin(

            l=25,

            r=10,

            b=50,

            t=50,

            pad=2

    ))

    return fig



def create_polar_multipart(

    df, 

    df_cols, 

    x_names, 

    strat_col, 

    legend_orientation='h',

    legend_x=None,

    legend_y=None,

    title='Plot title',

    xaxis_title='x axis title',

    yaxis_title='y axis title'):



    

    assert len(df_cols) == len(x_names)

    

    new_df = ~df[df_cols].copy().isnull()

    new_df.columns = x_names

    new_df[strat_col] = df[strat_col].copy()



    

    datums = []

    

    for c in x_names:

        tup = []

        for x in new_df[strat_col].unique():

            slice_x = new_df[new_df[strat_col] == x].copy()

            tup.append(slice_x[c].sum()/slice_x.shape[0])

        datums.append(tuple(tup))

    

    tuple_lengths = [len(x) for x in datums]

    min_slices = min(tuple_lengths)

    

    colors = random.sample(plotly.colors.DEFAULT_PLOTLY_COLORS, min_slices)

    fig = plotly.graph_objs.Figure()

    

    for i in range(min_slices):

        fig.add_trace(plotly.graph_objs.Scatterpolar(

            theta=x_names,

            r=[x[i] for x in datums],

            name=f'{new_df[strat_col].unique()[i]}',

            fill='toself'

            ))

        

    # add plot title

    fig.update_layout(

        title={

            'text' : title,

            'y' : 0.87, 

            'x' : 0.4, 

            'xanchor' : 'center',

            'yanchor' : 'top'

        }

    )        

    

    # adjust margins and legend settings

    fig.update_layout(

        legend_orientation=legend_orientation, 

        legend_x=legend_x, 

        legend_y=legend_y,

        margin=plotly.graph_objs.layout.Margin(

            l=25,

            r=10,

            b=50,

            t=50,

            pad=2

    ))

    

    return fig
grouped_ai_leaders = ai_leaders_africa[ai_leaders_africa['ai_leader'] == 'AI_LEADERS'].groupby('Q3')['ai_leader'].count().sort_values(ascending=False)

fig = plotly.graph_objs.Figure(

    data=[

        plotly.graph_objs.Table(

            header=dict(values=['Countries', 'Number of Respondents']),

            cells=dict(values=[grouped_ai_leaders.index, grouped_ai_leaders.values])

        )

    ])

fig.update_layout(

    autosize=False,

    width=700,

    height=150,

    margin=plotly.graph_objs.layout.Margin(

        l=0,

        r=0,

        b=0,

        t=0,

        pad=0

    )

)





fig.show()
grouped_african = african_countries.groupby('Q3')['ai_leader'].count().sort_values(ascending=False)

fig = plotly.graph_objs.Figure(

    data=[

        plotly.graph_objs.Table(

            header=dict(values=['Countries', 'Number of Respondents']),

            cells=dict(values=[grouped_african.index, grouped_african.values])

        )

    ])

fig.update_layout(

    autosize=False,

    width=700,

    height=169,

    margin=plotly.graph_objs.layout.Margin(

        l=0,

        r=0,

        b=0,

        t=0,

        pad=0

    )

)



fig.update_layout(

    paper_bgcolor="LightSteelBlue",

)





fig.show()
parameters = {

    'dfs' : [ai_leaders_africa, african_countries],

    'strat_cols' : ['ai_leader', 'Q3'],

    'quant_col' : 'Q1',

    'title' : None,

    'xaxis_titles' : [None] * 2,

    'yaxis_titles' : ['Percentage of respondents', None],

    'rows' : 1,

    'cols' : 2,

    'horizontal_spacing' : 0.02,

    'yaxis' : [0, 1.02],

    'legend_y' : -0.15,

    'legend_orientation' : 'h',

}



fig = create_plotly_barchart(**parameters)

fig.show()
parameters = {

    'dfs' : [ai_leaders_africa, african_countries],

    'strat_cols' : ['ai_leader', 'Q3'],

    'quant_col' : 'Q2',

    'title' : None,

    'xaxis_titles' : [None]*2,

    'yaxis_titles' : ['Percentage of respondents', None],

    'rows' : 1,

    'cols' : 2,

    'horizontal_spacing' : 0.02,

    'yaxis' : [0, 1.02],

    'legend_y' : -0.15

}



fig = create_plotly_barchart(**parameters)

fig.show()
invert_Q4_name_map = {

    4 : 'Master\'s',

    1 : 'Professional',

    3 : 'Bachelor\'s',

    2 : 'Some uni courses',

    6 : 'Doctoral',

    -1 : 'No answer',

    0 : 'High school',

}



parameters = {

    'dfs' : [ai_leaders_africa, african_countries],

    'strat_cols' : ['ai_leader', 'Q3'],

    'quant_col' : 'Q4',

    'title' : None,

    'xaxis_titles' : [None]*2,

    'yaxis_titles' : ['Percentage of respondents', None],

    'rows' : 1,

    'cols' : 2,

    'horizontal_spacing' : 0.02,

    'names_map' : invert_Q4_name_map,

    'yaxis' : [0, 1.02],

    'legend_y' : -0.15

}



fig = create_plotly_barchart(**parameters)

fig.show()
parameters = {

    'dfs' : [ai_leaders_africa, african_countries],

    'strat_cols' : ['ai_leader', 'Q3'],

    'quant_col' : 'Q15',

    'title' : None,

    'xaxis_titles' : [None]*2,

    'yaxis_titles' : ['Percentage of respondents', None],

    'rows' : 1,

    'cols' : 2,

    'horizontal_spacing' : 0.02,

    'names_map' : dict(map(reversed, Q15_name_map.items())),

    'yaxis' : [0, 1.02],

    'legend_y' : -0.15

}



fig = create_plotly_barchart(**parameters)

fig.show()
parameters = {

    'dfs' : [ai_leaders_africa, african_countries],

    'strat_cols' : ['ai_leader', 'Q3'],

    'quant_col' : 'Q23',

    'title' : None,

    'xaxis_titles' : [None]*2,

    'yaxis_titles' : ['Percentage of respondents', None],

    'rows' : 1,

    'cols' : 2,

    'horizontal_spacing' : 0.02,

    'names_map' : dict(map(reversed, Q23_name_map.items())),

    'yaxis' : [0, 1.02],

    'legend_y' : -0.15

}



fig = create_plotly_barchart(**parameters)

fig.show()
media_platforms_cols = ['Q12_Part_' + str(x) for x in range(1,11)]

media_platforms_names = [

    'Twitter', 

    'Hacker News',

    'Reddit',

    'Kaggle',

    'Course Forums',

    'Youtube',

    'Podcasts',

    'Blogs',

    'Journals',

    'Slack'

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : media_platforms_cols,

    'x_names' : media_platforms_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_x' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
edu_platforms_cols = ['Q13_Part_' + str(x) for x in range(1,11)]

edu_platforms_names = [

    'Udacity', 

    'Coursera',

    'edX',

    'DataCamp',

    'DataQuest',

    'Kaggle Courses',

    'Fast.ai',

    'Udemy',

    'LinkedIn Learning',

    'University Courses (resulting in degree)'

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : edu_platforms_cols,

    'x_names' : edu_platforms_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_x' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
algos_cols = ['Q24_Part_' + str(x) for x in range(1,11)]

algos_names = [

    'Regression', 

    'DT/RF',

    'GBM',

    'Bayesian',

    'Evolutionary',

    'Dense NN',

    'CNN',

    'GAN',

    'RNN',

    'Transformers'

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : algos_cols,

    'x_names' : algos_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_x' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
cv_cols = ['Q26_Part_' + str(x) for x in range(1,6)]

cv_names = [

    'General Purpose', 

    'Image Segmentation',

    'Object Detection',

    'Image Classification',

    'Generative Networks',

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : cv_cols,

    'x_names' : cv_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_y' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
nlp_cols = ['Q27_Part_' + str(x) for x in range(1,5)]

nlp_names = [

    'Word embeddings', 

    'Encoder-decoders',

    'Contextualized embeddings',

    'Transformer LM',

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : nlp_cols,

    'x_names' : nlp_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_x' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
ml_frameworks_cols = ['Q28_Part_' + str(x) for x in range(1,11)]

ml_frameworks_names = [

    'Scikit-learn', 

    'Tensorflow',

    'Keras',

    'RandomForest',

    'Xgboost',

    'PyTorch',

    'Caret',

    'LightGBM',

    'Spark MLib',

    'Fast.ai'

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : ml_frameworks_cols,

    'x_names' : ml_frameworks_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_x' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
hardware_cols = ['Q21_Part_' + str(x) for x in range(1,4)]

hardware_names = [

    'CPUs', 

    'GPUs',

    'TPUs'

]



parameters = {

    'df' : ai_leaders_africa,

    'strat_col' : 'ai_leader',

    'df_cols' : hardware_cols,

    'x_names' : hardware_names,

    'title' : None,

    'xaxis_title' : None,

    'legend_x' : 0.35,

}



fig = create_polar_multipart(**parameters)

fig.show()
parameters = {

    'dfs' : [ai_leaders_africa],

    'strat_cols' : ['ai_leader'],

    'quant_col' : 'Q22',

    'title' : None,

    'xaxis_titles' : [None],

    'yaxis_titles' : ['Percentage of respondents'],

    'yaxis' : [0, 1.02],

    'names_map' : dict(map(reversed, Q22_name_map.items())),

}



fig = create_plotly_barchart(**parameters)

fig.show()
invert_Q11_name_map = {

    0 : 'USD 0',

    1: 'USD 1-99',

    100: 'USD 100-999',

    1000 : 'USD 1,000-9,999',

    10000 : 'USD 10,000-99,999',

    100000 : 'USD >100,000'

}



parameters = {

    'dfs' : [ai_leaders_africa, african_countries],

    'strat_cols' : ['ai_leader', 'Q3'],

    'quant_col' : 'Q11',

    'title' : None,

    'xaxis_titles' : [None]*2,

    'yaxis_titles' : ['Percentage of respondents', None],

    'rows' : 1,

    'cols' : 2,

    'horizontal_spacing' : 0.02,

    'yaxis' : [0, 1.02],

    'names_map' : invert_Q11_name_map,

    'legend_y' : -0.15

}



fig = create_plotly_barchart(**parameters)

fig.show()