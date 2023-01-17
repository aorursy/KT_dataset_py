import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.io as pio

from plotly.subplots import make_subplots

from sklearn.cluster import KMeans



pio.templates.default = "plotly_white"



path = '../input/kaggle-survey-2019/'



questions_key = pd.read_csv(path+'multiple_choice_responses.csv', nrows=1)

questions_key = questions_key.transpose().reset_index()

questions_key.columns = ['q_num', 'q_text']

q_text = questions_key.q_text.str.split('-', n=1, expand=True)

questions_key['q_text_1'] = q_text[0].str.strip()

questions_key['q_text_2'] = q_text[1].str.strip()

questions_key.drop(columns=['q_text'], inplace=True)



response_all = pd.read_csv(path+'multiple_choice_responses.csv', skiprows=2, header=None)
def new_job_label (row):

    if row['job_title'] == 'Data Scientist':

        return 'Data Scientists'

    if row['job_title'] == 'Student':

        return 'Students/Not employed/Others'

    if row['job_title'] == 'Not employed':

        return 'Students/Not employed/Others'

    if row['job_title'] == 'Other':

        return 'Students/Not employed/Others'

    else :

        return 'Non Data Scientist'



def get_subset (data, col_from, col_to):

    df_sub = data.iloc[:, col_from:col_to]

    df_sub = pd.get_dummies(df_sub)

    df_sub.columns = [col.split('_')[-1].strip().split('(')[0].strip() for col in df_sub.columns]

    df_sub['job_title'] = data[6]

    df_sub['job_ds'] = df_sub.apply(new_job_label, axis=1)

    marker_valid = (df_sub.sum(axis=1, numeric_only=True) > 0)

    return df_sub[marker_valid]





media = get_subset(response_all, 22, 34)

algorithms = get_subset(response_all, 118, 130)

frameworks = get_subset(response_all, 155, 167)



media_table = response_all.iloc[:, 22:34]

media_table = pd.get_dummies(media_table)

media_table.columns = [col.split('_')[-1].strip() for col in media_table.columns]

media_table = media_table[(media_table.sum(axis=1, numeric_only=True) > 0)]

# media_table.drop(columns=['None', 'Other'], inplace=True)



algorithms_media = pd.merge(algorithms.drop(columns=['None', 'Other']), 

         media.drop(columns=['None', 'Other', 'job_title', 'job_ds']), 

         left_index=True, right_index=True)



frameworks_media = pd.merge(frameworks.drop(columns=['None', 'Other']), 

         media.drop(columns=['None', 'Other', 'job_title', 'job_ds']), 

         left_index=True, right_index=True)

media_mean = media_table.mean()

media_sum = media_table.sum()

media_table_sum = pd.DataFrame({'Count': media_sum,

                               'Percentage': media_mean})

media_table_sum.columns.name = 'Favorite media source reporting on data scient topics'

media_table_sum.sort_values(by='Percentage', ascending=False, inplace=True)

media_table_sum.style.format({'Percentage': "{:.1%}"}).bar(subset=['Percentage'], color='skyblue')
media_count = media_table.drop(columns=['None']).sum(axis=1)

algo_count = algorithms.iloc[:, 0:10].sum(axis=1)

fm_count = frameworks.iloc[:, 0:10].sum(axis=1)



count_df = pd.DataFrame({'media_count': media_count,

                        'algo_count': algo_count,

                        'fm_count': fm_count})

count_df.dropna(inplace=True)

count_df['media_count_cat'] = pd.qcut(count_df['media_count'],

                                      [0, 0.25, 0.75, 1],

                                      labels=['Low','Medium','High'])

# count_df['media_count_cat'] = pd.qcut(count_df['media_count'], 2, 

#                                       labels=['Low','High'])



media_cat_n = count_df['media_count_cat'].value_counts(sort=False)



algo_low = count_df.query('media_count_cat == "Low"')[

    'algo_count'].value_counts().sort_index()

algo_low = algo_low/media_cat_n['Low']



algo_med = count_df.query('media_count_cat == "Medium"')[

    'algo_count'].value_counts().sort_index()

algo_med = algo_med/media_cat_n['Medium']



algo_hi = count_df.query('media_count_cat == "High"')[

    'algo_count'].value_counts().sort_index()

algo_hi = algo_hi/media_cat_n['High']



fm_low = count_df.query('media_count_cat == "Low"')[

    'fm_count'].value_counts().sort_index()

fm_low = fm_low/media_cat_n['Low']



fm_med = count_df.query('media_count_cat == "Medium"')[

    'fm_count'].value_counts().sort_index()

fm_med = fm_med/media_cat_n['Medium']



fm_hi = count_df.query('media_count_cat == "High"')[

    'fm_count'].value_counts().sort_index()

fm_hi = fm_hi/media_cat_n['High']
fig = go.Figure(

    data=[go.Histogram(

        x=media_count,

        histnorm='percent',

        marker= dict(

            color='skyblue',

            opacity=0.6,

            line= {"color": "white", "width": 2}

        ))])



fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False, ticksuffix="%")

fig.update_layout(

    title='Distribution of the number of media sources used',

    width=600,

    height=400,

    yaxis=dict(title="% of respondents",)

)

fig.show()
fig = make_subplots(rows=1, cols=2,

                    subplot_titles=("Number of ML Algorithms",

                                    "Number of ML Frameworks"))



# Algorithms

fig.add_trace(go.Scatter(

    x=algo_low.index,

    y=algo_low.values,

    name='Low',

    fill='tozeroy',

    marker=dict(color='gray'),

    showlegend=True,

    opacity=0.2), 1, 1

)



fig.add_trace(go.Scatter(

    x=algo_med.index,

    y=algo_med.values,

    name='Medium',

    fill='tozeroy',

    marker=dict(color='salmon'),

    showlegend=True,

    opacity=0.2), 1, 1

)



fig.add_trace(go.Scatter(

    x=algo_hi.index,

    y=algo_hi.values,

    name='High',

    fill='tozeroy',

    marker=dict(color='dodgerblue'),

    showlegend=True,

    opacity=0.2), 1, 1

)



# Frameworks

fig.add_trace(go.Scatter(

    x=fm_low.index,

    y=fm_low.values,

    name='Low',

    fill='tozeroy',

    marker=dict(color='grey'),

    showlegend=False,), 1, 2

)



fig.add_trace(go.Scatter(

    x=fm_med.index,

    y=fm_med.values,

    name='Medium',

    fill='tozeroy',

    marker=dict(color='salmon'),

    showlegend=False,), 1, 2

)



fig.add_trace(go.Scatter(

    x=fm_hi.index,

    y=fm_hi.values,

    name='High',

    fill='tozeroy',

    marker=dict(color='dodgerblue'),

    showlegend=False,), 1, 2

)



fig.update_traces(opacity=0.2, mode='lines')



fig.update_layout(

    width=800,

    height=400,

    yaxis1=dict(title="% of respondents",

                tickformat='%'),

    yaxis2=dict(tickformat='%'),

)



fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, range=[0, 0.3])

fig.show()
def draw_segments(data, thres_high, thres_low, xref, color='skyblue'):

    """

    Draw line segment connecting the two dots. 

    - If difference is greater than high/low threshold, draw segment

    - Otherwise draw near invisible segment

    """

    segment_list = []

    for i in range(len(data[0])):

        value = data['diff'].iloc[i]

        if (value >= thres_high):

            segment = dict(

                type='line',

                x0=data[0].iloc[i]*1.01,

                y0=i,

                x1=data[1].iloc[i]*0.99,

                y1=i,

                xref=xref,

                line=dict(

                    color=color,

                    width=2,

                )

            )

        elif (value <= thres_low):

            segment = dict(

                type='line',

                x0=data[0].iloc[i]*0.99,

                y0=i,

                x1=data[1].iloc[i]*1.01,

                y1=i,

                xref=xref,

                line=dict(

                    color='grey',

                    width=2,

                )

            )

        else:

            segment = dict(

                type='line',

                x0=data[0].iloc[i]+0.01,

                y0=i,

                x1=data[1].iloc[i]-0.01,

                y1=i,

                xref=xref,

                line=dict(

                    color='whitesmoke',

                    width=0.1

                )

            )

        segment_list.append(segment)

    return segment_list



def draw_annotation(data, thres_high, thres_low, xref):

    """

    Draw annotation 

    - If difference is greater than {thres}, annotate with value

    """

    annot_list = []

    for i in range(len(data[0])):

        value = data['diff'].iloc[i]

        if (value >= thres_high):

            annot = dict(

                x=data[1].iloc[i],

                y=i,

                xref=xref,

                xshift=30,

                text=f"+{value:0.1%}",

                showarrow=False

            )

        elif (value <= thres_low): 

            annot = dict(

                x=data[1].iloc[i],

                y=i,

                xref=xref,

                xshift=-30,

                text=f"{value:0.1%}",

                showarrow=False

            )

        else:

            annot = dict(

                x=data[1].iloc[i],

                y=i,

                text="",

                showarrow=False

            )

        annot_list.append(annot)

    return annot_list





def dotplot_diff_media (media_type):

    data_list = []

    for data_input in [algorithms_media, frameworks_media]:

        data = data_input.groupby(media_type).mean().T.iloc[0:10]

        data['Average'] = data.mean(axis=1)

        data.sort_values(by='Average', inplace=True)

        data['diff'] = data[1]-data[0]

        data_list.append(data)



    fig = make_subplots(

        rows=1, cols=2,

        subplot_titles=['ML Algorithms/Models', 

                        'ML Frameworks/Libraries'],

        horizontal_spacing=0.2,

        x_title='% in each group regularly using the given ML Algorithm/Framework'

    )



    # Algorithms

    fig.add_trace(

        go.Scatter(x=data_list[0][1],

                   y=data_list[0].index,

                   name='Yes',

                   mode="markers",

                   marker=dict(color='#4ec2f7', line_width=1, size=9)

                   ), 1, 1)



    fig.add_trace(

        go.Scatter(x=data_list[0][0],

                   y=data_list[0].index,

                   name='No',

                   mode="markers",

                   marker=dict(color='grey', symbol='x',

                               line_width=0.5, size=8)

                   ), 1, 1)



    # Frameworks

    fig.add_trace(

        go.Scatter(x=data_list[1][1],

                   y=data_list[1].index,

                   name='Yes',

                   showlegend=False,

                   mode="markers",

                   marker=dict(color='#4ec2f7', line_width=1, size=9)

                   ), 1, 2)



    fig.add_trace(

        go.Scatter(x=data_list[1][0],

                   y=data_list[1].index,

                   name='No',

                   showlegend=False,

                   mode="markers",

                   marker=dict(color='grey', symbol='x',

                               line_width=0.5, size=8)

                   ), 1, 2)



    # Annotations

    segment_1 = draw_segments(data_list[0], 0.05, -0.05, 'x1')

    annot_1 = draw_annotation(data_list[0], 0.05, -0.05, 'x1')



    segment_2 = draw_segments(data_list[1], 0.05, -0.05, 'x2')

    annot_2 = draw_annotation(data_list[1], 0.05, -0.05, 'x2')



    segments = segment_1 + segment_2

    annotations = annot_1 + annot_2



    title = {

        'text': f'{media_type} is a favorite media source',

        'y': 0.9,

        'x': 0.5,

        'font': {'size': 20},

        'xanchor': 'center',

        'yanchor': 'top'}



    fig.update_layout(title=title,

                      shapes=segments,

                      annotations=annotations,

                      width=900,

                      height=500,

                      xaxis_tickformat='%',

                      xaxis2_tickformat='%',

                      margin=dict(t=140,),

                      legend_orientation="h",

                      legend=dict(x=0.4,

                                  xanchor='center',

                                  y=1.2)

                      )



    fig.update_xaxes(range=[0, 0.85], showgrid=False, zeroline=False)

    fig.update_yaxes(showgrid=True, gridcolor='aliceblue')

    return(fig)
dotplot_diff_media('Kaggle')
dotplot_diff_media('Blogs')
dotplot_diff_media('YouTube')
media_cluster = media.iloc[:, 0:10]

y_pred = KMeans(n_clusters=4, random_state=42, max_iter=10000).fit_predict(

    media_cluster.values)

media_cluster['cluster_number'] = y_pred

media_cluster['cluster_label'] = media_cluster['cluster_number']

media_cluster.replace({'cluster_label': {0: 'blogs',

                                        1: 'kaggle_blogs',

                                        2: 'kaggle_youtube',

                                        3: 'kaggle_youtube_blogs'}}, inplace=True)
data = media_cluster.groupby('cluster_number').mean().T

cluster_count = media_cluster.cluster_number.value_counts(sort=False)



fig = make_subplots(

    rows=2,

    cols=2,

    subplot_titles=(f"1: Blogs<br> (N={cluster_count[0]})<br> ", 

                    f"2: Kaggle + Blogs<br> (N={cluster_count[1]})<br> ", 

                    f"3: Kaggle + YouTube<br> (N={cluster_count[2]})<br> ", 

                    f"4: Kaggle + Blogs + YouTube<br> (N={cluster_count[3]})<br> "),

    specs=[[{'type': 'polar'}]*2]*2,

)



fig.add_trace(

    go.Barpolar(

        r=data[0]*100,

        theta=data.index,

        name='Cluster 0',

        marker_color='teal',

        opacity=0.6,

    ),

    row=1, col=1

)



fig.add_trace(

    go.Barpolar(

        r=data[1]*100,

        theta=data.index,

        name='Cluster 1',

        marker_color='gold',

        opacity=0.6,

    ),

    row=1, col=2

)



fig.add_trace(

    go.Barpolar(

        r=data[2]*100,

        theta=data.index,

        name='Cluster 2',

        marker_color='tomato',

        opacity=0.6,

    ),

    row=2, col=1

)



fig.add_trace(

    go.Barpolar(

        r=data[3]*100,

        theta=data.index,

        name='Cluster 3',

        marker_color='skyblue',

        opacity=0.6,

    ),

    row=2, col=2

)



fig.update_layout(

    title={'text': '4 Profiles of Media Diets',

           'font_size': 22,

           'x': 0.5,

           'y': 0.95},

    showlegend=False,

    title_font_color='#333333',

    margin=dict(t=150, l=20, r=20),

    legend_font_color='gray',

    legend_itemclick=False,

    legend_itemdoubleclick=False,

    width=850,

    height=700,

    polar=dict(

        angularaxis=dict(

            direction='clockwise',

            rotation=110,

            color='grey',

            visible=True,

            showline=True,

        ),

        radialaxis=dict(

            ticksuffix='%',

            tickvals=[25, 50, 75],

            range=[0, 100],

            visible=True,

            showline=True,

        )),

    polar2=dict(

        angularaxis=dict(

            direction='clockwise',

            rotation=110,

            color='grey',

            visible=True,

            showline=True,

        ),

        radialaxis=dict(

            ticksuffix='%',

            tickvals=[25, 50, 75],

            range=[0, 100],

            visible=True,

            showline=True,

        )),

    polar3=dict(

        angularaxis=dict(

            direction='clockwise',

            rotation=110,

            color='grey',

            visible=True,

            showline=True,

        ),

        radialaxis=dict(

            ticksuffix='%',

            tickvals=[25, 50, 75],

            range=[0, 100],

            visible=True,

            showline=True,

        )),

    polar4=dict(

        angularaxis=dict(

            direction='clockwise',

            rotation=110,

            color='grey',

            visible=True,

            showline=True,

        ),

        radialaxis=dict(

            ticksuffix='%',

            tickvals=[25, 50, 75],

            range=[0, 100],

            visible=True,

            showline=True,

        )),

)



fig.show()
algorithms_media_cluster = pd.merge(algorithms.drop(columns=['None', 'Other']),

                                    media_cluster[['cluster_label']],

                                    left_index=True, right_index=True)



frameworks_media_cluster = pd.merge(frameworks.drop(columns=['None', 'Other']),

                                    media_cluster[['cluster_label']],

                                    left_index=True, right_index=True)
def dotplot_diff_cluster(group_1, group_2, title_text, line_color):



    cluster_colors = {

        'blogs': 'teal',

        'kaggle_blogs': 'gold',

        'kaggle_youtube': 'tomato',

        'kaggle_youtube_blogs': 'skyblue'

    }



    cluster_labels = {

        'blogs': 'Blogs only',

        'kaggle_blogs': 'Kaggle + Blogs',

        'kaggle_youtube': 'Kaggle + YouTube',

        'kaggle_youtube_blogs': 'Kaggle + Blogs + YouTube'

    }



    data_list = []

    for data_input in [algorithms_media_cluster, frameworks_media_cluster]:

        data = pd.DataFrame()

        data[0] = data_input.query(

            'cluster_label == @group_1').mean().T.iloc[0:-1]

        data[1] = data_input.query(

            'cluster_label == @group_2').mean().T.iloc[0:-1]

        data['Average'] = data_input.mean()

        data.sort_values(by='Average', inplace=True)

        data['diff'] = data[1]-data[0]

        data_list.append(data)



    fig = make_subplots(

        rows=1, cols=2,

        subplot_titles=['ML Algorithms/Models', 'ML Frameworks/Libraries'],

        horizontal_spacing=0.2,

        x_title='% in each group regularly using the given ML Algorithm/Framework'

    )



    # Algorithms

    fig.add_trace(

        go.Scatter(x=data_list[0][0],

                   y=data_list[0].index,

                   name=cluster_labels.get(group_1),

                   mode="markers",

                   marker=dict(

            color=cluster_colors.get(group_1),

            line_width=1, size=10)

        ), 1, 1)



    fig.add_trace(

        go.Scatter(x=data_list[0][1],

                   y=data_list[0].index,

                   name=cluster_labels.get(group_2),

                   mode="markers",

                   marker=dict(color=cluster_colors.get(group_2),

                               line_width=1, size=10)

                   ), 1, 1)



    # Frameworks

    fig.add_trace(

        go.Scatter(x=data_list[1][0],

                   y=data_list[1].index,

                   name=cluster_labels.get(group_1),

                   mode="markers",

                   showlegend=False,

                   marker=dict(

            color=cluster_colors.get(group_1),

            line_width=1, size=10)

        ), 1, 2)



    fig.add_trace(

        go.Scatter(x=data_list[1][1],

                   y=data_list[1].index,

                   name=cluster_labels.get(group_2),

                   showlegend=False,

                   mode="markers",

                   marker=dict(color=cluster_colors.get(group_2),

                               line_width=1, size=10)

                   ), 1, 2)



    # Annotations

    segment_1 = draw_segments(data_list[0], 0.05, -0.05, 'x1', line_color)

    annot_1 = draw_annotation(data_list[0], 0.05, -0.05, 'x1')



    segment_2 = draw_segments(data_list[1], 0.05, -0.05, 'x2', line_color)

    annot_2 = draw_annotation(data_list[1], 0.05, -0.05, 'x2')



    segments = segment_1 + segment_2

    annotations = annot_1 + annot_2



    title = {

        'text': title_text,

        'y': 0.9,

        'x': 0.5,

        'font': {'size': 20},

        'xanchor': 'center',

        'yanchor': 'top'}



    fig.update_layout(title=title,

                      shapes=segments,

                      annotations=annotations,

                      width=900,

                      height=500,

                      xaxis_tickformat='%',

                      xaxis2_tickformat='%',

                      margin=dict(t=140,),

                      legend_orientation="h",

                      legend=dict(x=0.4,

                                  xanchor='center',

                                  y=1.25)

                      )



    fig.update_xaxes(range=[0, 0.85], showgrid=False, zeroline=False)

    fig.update_yaxes(showgrid=True, gridcolor='aliceblue')

    return fig
dotplot_diff_cluster('kaggle_youtube', 'kaggle_youtube_blogs', 'Additional Values from Blogs', 'skyblue')
dotplot_diff_cluster('kaggle_blogs', 'kaggle_youtube_blogs', 'Additional Values from YouTube', 'skyblue')
dotplot_diff_cluster('kaggle_youtube', 'kaggle_blogs', 'Blogs vs YouTube', 'gold')
def get_slope_color (media):

    if media == 'Kaggle':

        return 'deepskyblue'

    if media == 'Blogs':

        return 'gold'

    if media == 'YouTube':

        return 'tomato'

    else :

        return 'lightgrey'



media_rank = media.query('job_ds != "Non Data Scientist"').groupby(

    'job_ds').mean()

media_rank.drop(columns=['None', 'Other'], inplace=True)
fig = go.Figure()



for col in media_rank.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=media_rank.index,

        y=media_rank[col],

        mode='lines+markers',

        name=col,

        line=dict(

            color=get_slope_color(col),

            #color= 'dodgerblue' if col == 'Blogs' or col == 'YouTube' else 'lightgrey',

            width=2.5

        ),

    ))





fig.update_layout(

    width=600,

    height=600,

    xaxis=dict(

        showline=False,

        showgrid=False,

        showticklabels=False,

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    showlegend=False,

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=200,

        r=100,

        t=110,

    ),

)



# Adding labels

annotations = []

for col in ['Kaggle', 'Blogs', 'YouTube']:

    # labeling the left_side of the plot

    annotations.append(dict(xref='paper', x=0.0, y=media_rank[col][0],

                            xanchor='right', yanchor='middle',

                            text=f'{col} {media_rank[col][0]*100:0.1f}%',

                            showarrow=False))

    # labeling the right_side of the plot

    annotations.append(dict(xref='paper', x=1.0, y=media_rank[col][1],

                            xanchor='left', yanchor='middle',

                            text=f'{col} {media_rank[col][1]*100:0.1f}%',

                            showarrow=False))



for col in ['Twitter', 'Hacker News', 'Reddit', 'Course Forums',

            'Podcasts', 'Journal Publications',

            'Slack Communities']:

    # labeling the left_side of the plot

    annotations.append(dict(xref='paper', x=0.0, y=media_rank[col][0],

                            xanchor='right', yanchor='middle',

                            text=f'{col}',

                            font=dict(size=10),

                            showarrow=False))



# Job labels

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=0.98,

                        xanchor='center', yanchor='bottom',

                        text='% of Data Scientists<br> ',

                        font=dict(size=14),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='paper', x=0.9, y=0.98,

                        xanchor='center', yanchor='bottom',

                        text='% of Students/<br>Not employed/Others',

                        font=dict(size=14),

                        showarrow=False))



title = {

        'text': 'Aspiring Data Scientists have Different<br> Media Choices from Data Scientists',

        'y': 0.95,

        'x': 0.5,

        'font': {'size': 18},

        'xref': 'paper',

        'yanchor': 'top'}



fig.update_layout(

    title=title,

    width=500,

    height=450,

    annotations=annotations,

    margin=dict(t=100, b=10))



fig.show()
media_cluster_by_job = pd.merge(media[['job_ds']],

                                media_cluster[['cluster_label']],

                                left_index=True, right_index=True)



media_cluster_by_job = pd.get_dummies(media_cluster_by_job.query('job_ds != "Non Data Scientist"'), columns=[

                                      'cluster_label']).groupby('job_ds').mean().T
fig = go.Figure()



cluster_labels = ['Blogs only', 

                  'Kaggle +<br>Blogs',

                  'Kaggle +<br>YouTube', 

                  'Kaggle +<br>Blogs + YouTube']



cluster_colors = ['teal', 'gold', 'tomato', 'skyblue']



x_data = [media_cluster_by_job['Students/Not employed/Others']*100,

         media_cluster_by_job['Data Scientists']*100]



y_data = ['Students/ <br>Not employed/Others',

         'Data Scientists']



fig = go.Figure()



for i in range(0, len(x_data[0])):

    for xd, yd in zip(x_data, y_data):

        fig.add_trace(go.Bar(

            x=[xd[i]], y=[yd],

            orientation='h',

            width=0.5,

            marker=dict(

                color=cluster_colors[i],

                opacity=0.7,

                line=dict(color='white', width=1)

            )

        ))

        

fig.update_layout(

    xaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=False,

        zeroline=False,

        domain=[0.15, 1]

    ),

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=False,

        zeroline=False,

    ),

    barmode='stack',

    margin=dict(l=120, r=10, t=140, b=80),

    showlegend=False,

)



annotations = []



for yd, xd in zip(y_data, x_data):

    # labeling the y-axis

    annotations.append(dict(xref='paper', yref='y',

                            x=0.14, y=yd,

                            xanchor='right',

                            text=str(yd),

                            showarrow=False, align='right'))

    # labeling the first percentage of each bar (x_axis)

    annotations.append(dict(xref='x', yref='y',

                            x=xd[0] / 2, y=yd,

                            text= f'{xd[0]:0.1f}%',

                            showarrow=False))

    # labeling the first Likert scale (on the top)

    if yd == y_data[-1]:

        annotations.append(dict(xref='x', yref='paper',

                                x=xd[0] / 2, y=1.1,

                                text=cluster_labels[0],

                                showarrow=False))

    space = xd[0]

    for i in range(1, len(xd)):

            # labeling the rest of percentages for each bar (x_axis)

            annotations.append(dict(xref='x', yref='y',

                                    x=space + (xd[i]/2), y=yd,

                                    text=f'{xd[i]:0.1f}%',

                                    showarrow=False))

            # labeling the Likert scale

            if yd == y_data[-1]:

                annotations.append(dict(xref='x', yref='paper',

                                        x=space + (xd[i]/2), y=1.1,

                                        text=cluster_labels[i],

                                        showarrow=False))

            space += xd[i]



title = {

        'text': 'Distribution of Media Diet Profiles <br>among Aspiring Data Scientists and Data Scientists',

        'y': 0.9,

        'x': 0.5,

        'font': {'size': 16},

        'xref': 'paper',

        'yanchor': 'top'}           

            

fig.update_layout(

    title=title,

    width=700,

    height=300,

    margin=dict(t=100,b=20,),

    annotations=annotations)

fig.show()
def get_slope_color_2(data):

    diff = data.iloc[:, 1] - data.iloc[:, 0]

    #diff = abs(diff)

    color_list = []

    for i in range(0, len(diff)):

        if diff[i] >= 0.05:

            color = 'dodgerblue'

        elif diff[i] <= -0.05:

            color = 'coral'

        else:

            color = 'lightgrey'

        color_list.append(color)

    return color_list





def gen_subplot_data (data_input, label):

    data = data_input.query('job_ds == "Students/Not employed/Others"').groupby('cluster_label').mean().T

    data['ds_avg'] = data_input.query('job_ds == "Data Scientists"').mean().T

    data.sort_values(by='ds_avg', inplace=True)

    data['label'] = label

    data['diff_ds_b'] = data['blogs']-data['ds_avg']

    data['diff_ds_kb'] = data['kaggle_blogs']-data['ds_avg']

    data['diff_ds_ky'] = data['kaggle_youtube']-data['ds_avg']

    data['diff_ds_kyb'] = data['kaggle_youtube_blogs']-data['ds_avg']



    data_sub_yt_blog = data[['diff_ds_ky', 'diff_ds_kb', 'label']]

    data_sub_yt_blog['color'] = get_slope_color_2(data_sub_yt_blog)

    data_sub_yt_blog = data_sub_yt_blog.T



    data_sub_blog = data[['diff_ds_ky', 'diff_ds_kyb', 'label']]

    data_sub_blog['color'] = get_slope_color_2(data_sub_blog)

    data_sub_blog = data_sub_blog.T



    data_sub_yt = data[['diff_ds_kb', 'diff_ds_kyb', 'label']]

    data_sub_yt['color'] = get_slope_color_2(data_sub_yt)

    data_sub_yt = data_sub_yt.T

    return data_sub_yt_blog, data_sub_blog, data_sub_yt

    

    

def annotate_subplot_left(data, column, xref):

    # labeling the left_side of the plot

    label_l = data[column]['label']

    if data[column]['color'] == 'lightgrey':

        annot_l = dict(xref=xref, x=-0.1, y=data[column][0],

                       xanchor='right', yanchor='middle',

                       text=f'{label_l}',

                       font=dict(color="lightgrey"),

                       showarrow=False)

    else:

        annot_l = dict(xref=xref, x=-0.1, y=data[column][0],

                       xanchor='right', yanchor='middle',

                       text=f'{label_l}<br> {data[column][0]*100:0.1f}%',

                       font=dict(color=data[column]['color']),

                       showarrow=False)

    return annot_l





def annotate_subplot_right(data, column, xref):

    # labeling the right side of the plot

    if data[column]['color'] == 'lightgrey':

        annot_r = dict(xref=xref, x=1.1, y=data[column][1],

                       xanchor='left', yanchor='middle',

                       text='',

                       showarrow=False)

    else:

        annot_r = dict(xref=xref, x=1.1, y=data[column][1],

                       xanchor='left', yanchor='middle',

                       text=f'{data[column][1]*100:0.1f}%',

                       font=dict(color=data[column]['color']),

                       showarrow=False)



    return annot_r





algo_label = ['GAN', 'Evoluationary', 'Transformer N.', 'RNN',

                     'Bayesian', 'DNN', 'CNN', 'GBM', 'DT/RF', 'Linear/Logistic']



framework_label = ['Fast.ai', 'Spark MLib', 'Caret', 'PyTorch', 'LightGBM', 'RandomForest',

                   'TensorFlow', 'Xgboost', 'Keras', 'Scikit-learn']



algo_sub_yt_blog, algo_sub_blog, algo_sub_yt = gen_subplot_data(algorithms_media_cluster, algo_label)

framework_sub_yt_blog, framework_sub_blog, framework_sub_yt = gen_subplot_data(frameworks_media_cluster, framework_label)

data_sub_yt_blog = algo_sub_yt_blog

data_sub_blog = algo_sub_blog

data_sub_yt = algo_sub_yt



fig = make_subplots(

    rows=1,

    cols=3,

    shared_xaxes=True,

    shared_yaxes=True,

)



# Blogs vs YouTube

for col in data_sub_yt_blog.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=['Kaggle +<br>YouTube', 'Kaggle +<br>Blogs'],

        y=data_sub_yt_blog[col],

        mode='lines+markers',

        name=col,

        line=dict(color=data_sub_yt_blog[col]['color'],

                  width=2.5),

    ), row=1, col=1)





# Adding Blogs

for col in data_sub_blog.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=['Kaggle +<br>YouTube', 'Kaggle + <br>Blogs + YouTube'],

        y=data_sub_blog[col],

        mode='lines+markers',

        name=col,

        line=dict(color=data_sub_blog[col]['color'],

                  width=2.5),

    ), row=1, col=2)



    

# Adding YouTube

for col in data_sub_yt.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=['Kaggle +<br>Blogs', 'Kaggle +<br>Blogs + YouTube'],

        y=data_sub_yt[col],

        mode='lines+markers',

        name=col,

        line=dict(color=data_sub_yt[col]['color'],

                  width=2.5),

    ), row=1, col=3)





# Adding annotatios

annotations = []

for col in data_sub_yt_blog.columns:

    # Left side label

    annot_l = annotate_subplot_left(data_sub_yt_blog, col, 'x1')

    annotations.append(annot_l)

    # Right side label

    annot_r = annotate_subplot_right(data_sub_yt_blog, col, 'x1')

    annotations.append(annot_r)



    

for col in data_sub_blog.columns:

    # Left side label

    annot_l = annotate_subplot_left(data_sub_blog, col, 'x2')

    annotations.append(annot_l)

    # Right side label

    annot_r = annotate_subplot_right(data_sub_blog, col, 'x2')

    annotations.append(annot_r)





for col in data_sub_yt.columns:

    # Left side label

    annot_l = annotate_subplot_left(data_sub_yt, col, 'x3')

    annotations.append(annot_l)

    # Right side label

    annot_r = annotate_subplot_right(data_sub_yt, col, 'x3')

    annotations.append(annot_r)

    



annotations.append(dict(xref='paper', yref='y', x=0.15, y=0.0,

                        xanchor='center', yanchor='middle',

                        text='Baseline: % of Data Scientists regularly<br> use this algorithm',

                        font=dict(size=11),

                        showarrow=True))





annotations.append(dict(xref='paper', yref='y', x=0.1, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.25, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>Blogs',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.42, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.58, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube + Blogs',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.78, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>Blogs',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.95, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube + Blogs',

                        font=dict(size=13),

                        showarrow=False))



title_text = 'ML Algorithms Usage:<br>Aspiring Data Scientists with Different Media Diets<br>and How Close They are to Data Scientists <br><br>'

title = {

        'text': title_text,

        'y': 0.95,

        'x': 0.5,

        'font': {'size': 18},

        'xref': 'paper', #'yref': 'paper',

        'yanchor': 'top'} 



# Axis formatting

layout_xaxis = dict(

        showline=False,

        showgrid=False,

        showticklabels=False,

    )

layout_y_axis = dict(

        showgrid=False,

        zeroline=True,

        zerolinewidth=2.5, 

        zerolinecolor='dimgrey',

        showline=False,

        showticklabels=False,

        #range=[-0.45, 0.05]

    )



fig.update_layout(

    title=title,

    xaxis=layout_xaxis,

    yaxis=layout_y_axis,

    xaxis2=layout_xaxis,

    yaxis2=layout_y_axis,

    xaxis3=layout_xaxis,

    yaxis3=layout_y_axis,

)



fig.update_layout(annotations=annotations,

                  width=850,

                  height=650,

                  showlegend=False,

                  autosize=False,

                  margin=dict(

                      t=100,

                      l=10,

                      r=10,

                  ),

                  )

fig.show()
data_sub_yt_blog = framework_sub_yt_blog

data_sub_blog = framework_sub_blog

data_sub_yt = framework_sub_yt



fig = make_subplots(

    rows=1,

    cols=3,

    shared_xaxes=True,

    shared_yaxes=True,

)



# Blogs vs YouTube

for col in data_sub_yt_blog.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=['Kaggle +<br>YouTube', 'Kaggle +<br>Blogs'],

        y=data_sub_yt_blog[col],

        mode='lines+markers',

        name=col,

        line=dict(color=data_sub_yt_blog[col]['color'],

                  width=2.5),

    ), row=1, col=1)





# Adding Blogs

for col in data_sub_blog.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=['Kaggle +<br>YouTube', 'Kaggle + <br>Blogs + YouTube'],

        y=data_sub_blog[col],

        mode='lines+markers',

        name=col,

        line=dict(color=data_sub_blog[col]['color'],

                  width=2.5),

    ), row=1, col=2)



    

# Adding YouTube

for col in data_sub_yt.columns:

    # Slope

    fig.add_trace(go.Scatter(

        x=['Kaggle +<br>Blogs', 'Kaggle +<br>Blogs + YouTube'],

        y=data_sub_yt[col],

        mode='lines+markers',

        name=col,

        line=dict(color=data_sub_yt[col]['color'],

                  width=2.5),

    ), row=1, col=3)





# Adding annotatios

annotations = []

for col in data_sub_yt_blog.columns:

    # Left side label

    annot_l = annotate_subplot_left(data_sub_yt_blog, col, 'x1')

    annotations.append(annot_l)

    # Right side label

    annot_r = annotate_subplot_right(data_sub_yt_blog, col, 'x1')

    annotations.append(annot_r)



    

for col in data_sub_blog.columns:

    # Left side label

    annot_l = annotate_subplot_left(data_sub_blog, col, 'x2')

    annotations.append(annot_l)

    # Right side label

    annot_r = annotate_subplot_right(data_sub_blog, col, 'x2')

    annotations.append(annot_r)





for col in data_sub_yt.columns:

    # Left side label

    annot_l = annotate_subplot_left(data_sub_yt, col, 'x3')

    annotations.append(annot_l)

    # Right side label

    annot_r = annotate_subplot_right(data_sub_yt, col, 'x3')

    annotations.append(annot_r)

    



annotations.append(dict(xref='paper', yref='y', x=0.15, y=0.0,

                        xanchor='center', yanchor='middle',

                        text='Baseline: % of Data Scientists regularly<br> use this framework',

                        font=dict(size=11),

                        showarrow=True))





annotations.append(dict(xref='paper', yref='y', x=0.1, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.25, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>Blogs',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.42, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.58, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube + Blogs',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.78, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>Blogs',

                        font=dict(size=13),

                        showarrow=False))



annotations.append(dict(xref='paper', yref='y', x=0.95, y=0.1,

                        xanchor='center', yanchor='top',

                        text='Kaggle +<br>YouTube + Blogs',

                        font=dict(size=13),

                        showarrow=False))



title_text = 'ML Frameworks Usage:<br>Aspiring Data Scientists with Different Media Diets<br>and How Close They are to Data Scientists<br><br> '

title = {

        'text': title_text,

        'y': 0.95,

        'x': 0.5,

        'font': {'size': 18},

        'xref': 'paper', #'yref': 'paper',

        'yanchor': 'top'} 





# Axis formatting

layout_xaxis = dict(

        showline=False,

        showgrid=False,

        showticklabels=False,

    )

layout_y_axis = dict(

        showgrid=False,

        zeroline=True,

        zerolinewidth=2.5, 

        zerolinecolor='dimgrey',

        showline=False,

        showticklabels=False,

        #range=[-0.45, 0.05]

    )



fig.update_layout(

    title=title,

    xaxis=layout_xaxis,

    yaxis=layout_y_axis,

    xaxis2=layout_xaxis,

    yaxis2=layout_y_axis,

    xaxis3=layout_xaxis,

    yaxis3=layout_y_axis,

)



fig.update_layout(annotations=annotations,

                  width=850,

                  height=650,

                  showlegend=False,

                  autosize=False,

                  margin=dict(

                      l=10,

                      r=10,

                  ),

                  )

fig.show()