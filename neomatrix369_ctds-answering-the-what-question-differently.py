import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



### import the utility script (see https://www.kaggle.com/neomatrix369/nlp-profiler-class)

from nlp_profiler_class import NLPProfiler 
import warnings

warnings.filterwarnings("ignore")



# visualization tools

import seaborn as sns

import matplotlib.pyplot as plt

import plotly_express as px

import plotly.graph_objects as go



# prettify plots

plt.rcParams['figure.figsize'] = [20.0, 7.0]



sns.set_palette(sns.color_palette("muted"))

sns.set_style("ticks")

sns.set(rc={'figure.figsize':(20.0, 20.0)})

sns.set(style="whitegrid")

sns.set(font_scale=1.25)
data = {

    "viewers": ['Viewer 1', 'Viewer 2', 'Viewer 3', 'Viewer 4', 'Viewer 5', 'Viewer 6'],

    "interest": ['related','unrelated','exploring','open','focussed','curious',],

    "priority": ['low','medium','high','medium','high','low',],

    "time-factor": ['hardto make time','free','usually free','work-life busy','busy','free',],

    "knows-about-the-show": ['yes, regular','no, firstime','yes','yes','no, firstime','no, firstime',],

    "thinks-about-the-show": ['likes','dislikes','neutral','neutral','neutral','likes',],

    "knows-the-host": ['yes','no','yes','yes','no','no',],

    "thinks-about-the-host": ['likes','dislikes','neutral','neutral','neutral','likes',],

    "knows-the-speaker": ['yes','no','yes','yes','no','no',],

    "thinks-about-the-speaker": ['likes','dislikes','neutral','neutral','neutral','likes',],

}

pd.DataFrame(data)
df = pd.read_csv('/kaggle/input/chai-time-data-science/Episodes.csv')

df_yt = pd.read_csv('/kaggle/input/chai-time-data-science/YouTube Thumbnail Types.csv')

df_yt = pd.merge(df,df_yt,on='youtube_thumbnail_type')

colors = {0:'#FDE803',1:'#0080B7',2:'#FF3D09',3:'#7CBB15'}

df_yt['color'] = df_yt.youtube_thumbnail_type.map(colors)

df_yt['ep_no'] = df_yt.episode_id.apply(lambda x: int(x[1:]) if x[0]=='E' else 75+int(x[1:]))

df_yt.sort_values('ep_no',inplace=True)

df_yt['heroes'] = df_yt['heroes'].fillna('NaN')

df_yt['episode'] = df_yt.apply(lambda x: x['episode_id'] + ' | ' + x['heroes'] 

                               if x['heroes']!='NaN' else x['episode_id'], axis=1)
PLOT_BGCOLOR='#DADEE3'

PAPER_BGCOLOR='rgb(255,255,255)'



y_avg = df_yt.youtube_impressions.mean()

y_med = df_yt.youtube_impressions.median()

fig = go.Figure()

fig.add_trace(go.Bar(name='Impressions',x=df_yt.episode_id,y=df_yt.youtube_impressions,

                     marker_line_width=1,marker_color='rgb(255,255,255)',marker_line_color='black',

                    text=df_yt['episode'],showlegend=False))

fig.add_trace(go.Scatter(name='Mean Impressions',x=df_yt.episode_id,

                         y=[y_avg]*len(df_yt),mode='lines',marker_color='black',

                        line = dict(dash='dash')))

fig.add_trace(go.Scatter(name='Median Impressions',x=df_yt.episode_id,

                         y=[y_med]*len(df_yt),mode='lines',marker_color='black',

                        line = dict(dash='dot')))

# Add image

fig.add_layout_image(

    dict(

        source='https://cdn.icon-icons.com/icons2/1584/PNG/512/3721679-youtube_108064.png',

        xref="paper", yref="paper",

        x=1, y=1,

        sizex=0.2, sizey=0.2,

        xanchor="right", yanchor="bottom"

    )

)

fig.update_layout(title='<b>Youtube Impressions</b> per Episode',

                width=700,height=300, barmode='stack',

                paper_bgcolor=PAPER_BGCOLOR,plot_bgcolor=PLOT_BGCOLOR,hovermode='x unified',

                margin=dict(t=40,b=0,l=0,r=0),legend=dict(x=0.5,y=1,orientation='h',bgcolor=PLOT_BGCOLOR),

                xaxis=dict(mirror=True,linewidth=2,linecolor='black',

                showgrid=False,tickfont=dict(size=8)),

                yaxis=dict(mirror=True,linewidth=2,linecolor='black',gridcolor='darkgray'))

fig.show()
aggregated_subtitles = pd.read_csv('/kaggle/input/ctds-subtitles-exploration/subtitles_aggregated.csv')

aggregated_subtitles_sorted = aggregated_subtitles.sort_values(by=['episode_id', 'relative_index']).reset_index(drop=True)

episode_filter = aggregated_subtitles_sorted['episode_id'] == 'E1'

first_episode = aggregated_subtitles_sorted[episode_filter].copy()

print("first_episode.shape:", first_episode.shape)
%%time

first_episode_nlp_profiled = NLPProfiler().apply_text_profiling(first_episode, 'Text', 

                                                                 params={'high_level': True, 'granular': False, 

                                                                         'spelling_check': True, 'grammar_check': True})

print("first_episode_nlp_profiled.shape:", first_episode_nlp_profiled.shape)
first_episode_nlp_profiled = pd.concat([first_episode['timestamp_relative'], first_episode['Speaker'], first_episode_nlp_profiled], axis=1)

first_episode_nlp_profiled
first_episode_nlp_profiled.to_csv('first_episode_nlp_profiled.csv')
def add_enhancements_to_chart(fig, plt_title, xaxis_title, yaxis_title, start_range, end_range, mean_val):

    fig.update_layout(

        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"),

        xaxis_title = xaxis_title, yaxis_title = yaxis_title

    )

    

    fig.add_shape(

        # add a horizontal line

        type="line",

        x0=start_range,

        y0=mean_val,

        x1=end_range,

        y1=mean_val,

        line=dict(color="black", width=2, dash="dash"),

    )



    # template enhancement

    fig.update_layout(

        template="ggplot2",

        title={

            "text": plt_title,

            "font": {"family": "Rockwell", "size": 20},

            "xanchor": "center",

            "yanchor": "top",

        },

        xaxis=dict(range=[start_range, end_range]),

    )

    return fig



def timeline_plot(df, feature, 

                  independent_feature="timestamp_relative", 

                  xaxis_title='Timeline', 

                  yaxis_title='', 

                  hover_data=[], 

                  title="You forgot the title"):

    """

    Credits: original function can be found in Ramshankar Yadhunath's CTDS Show winning kernel: 

             https://www.kaggle.com/thedatabeast/making-perfect-chai-and-other-tales#notebook-container. 

             I have adapted the function and made it generic for repurposing it in this kernel.

    

    Plots an interactive line plot depicting the values of each episode

    across a particular `feature` in the data

    -----

    

    df: The dataframe

    feature: The feature name from the relevant dataset

    independent_feature: The independent feature across which the values were shown

         `timestamp_relative` was the default

    xaxis_title: title appearing on the x-axis, in the absence the default is 'Timeline'

    yaxis_title: title appearing on the y-axis

    hover_data: values of list of fields that appears when you hover over the timeline

    title: Title of the plot

    """

    

    # find the mean value

    min_val = round(df[feature].min(), 2)

    max_val = round(df[feature].max(), 2)

    mean_val = round(df[feature].mean(), 2)

    margin = 25

    start_range = str(df[independent_feature].min() - margin)

    end_range =  str(df[independent_feature].max() + margin)



    

    # set the plot title

    plt_title = f'{title} (Min={min_val}, Max={max_val}, Mean={mean_val})'



    # plot the graph

    fig = px.line(df, x=independent_feature, y=feature, hover_data=hover_data)

    fig.update_traces(mode="lines+markers", line_color="#e87d23")

    

    fig = add_enhancements_to_chart(fig, plt_title, xaxis_title, yaxis_title, start_range, end_range, mean_val)

    

    # show plot

    fig.show()
timeline_plot(first_episode_nlp_profiled, 

              'sentiment_polarity_score', 

              xaxis_title="Episode E1: Timeline",

              yaxis_title="Sentiment polarity score",

              hover_data=["sentiment_polarity", "sentiment_polarity_score"],

              title='Sentiment polarity: Episode E1')
chart = sns.catplot(x="sentiment_polarity", data=first_episode_nlp_profiled, size=8, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment polarity moments", fontsize=16)

chart.set_xlabels("Sentiment polarity (in words)", fontsize=16)

chart.fig.suptitle(' Positive, Neutral and Negative graphs with the scales in between')

chart.set_xticklabels(rotation=30)
chart = sns.catplot(x="sentiment_polarity_summarised", data=first_episode_nlp_profiled, size=8, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment polarity moments", fontsize=16)

chart.set_xlabels("Sentiment polarity (in words)", fontsize=16)

chart.fig.suptitle('Positive, Neutral and Negative graphs (compact view)')
abhishek_filter = first_episode_nlp_profiled['Speaker'] == 'Abhishek Thakur'

sanyam_filter = first_episode_nlp_profiled['Speaker'] == 'Sanyam Bhutani'
def speakers_timeline_plot(df1, df2, feature, 

                  independent_feature="timestamp_relative", 

                  xaxis_title='Timeline', 

                  yaxis_title='', 

                  hover_data=[], 

                  title="You forgot the title"):

    """

    Credits: original function can be found in Ramshankar Yadhunath's CTDS Show winning kernel: 

             https://www.kaggle.com/thedatabeast/making-perfect-chai-and-other-tales#notebook-container. 

             I have adapted the function and made it generic for repurposing it in this kernel.

    

    Plots an interactive line plot depicting the values of each episode

    across a particular `feature` in the data

    -----

    

    df1, df2: The two dataframes containing the distinct information about the two speakers

    feature: The feature name from the relevant dataset

    independent_feature: The independent feature across which the values were shown

         `timestamp_relative` was the default

    xaxis_title: title appearing on the x-axis, in the absence the default is 'Timeline'

    yaxis_title: title appearing on the y-axis

    hover_data: values of list of fields that appears when you hover over the timeline

    title: Title of the plot

    """

    

    margin = 25

    combined_df = pd.concat([df1, df2], axis=0)

    mean_val = round(combined_df[feature].mean(), 2)

    start_range = str(combined_df[independent_feature].min() - margin)

    end_range =  str(combined_df[independent_feature].max() + margin)

    

    # set the plot title

    plt_title = f'{title} Mean: {mean_val}'

    hovertemplate = xaxis_title + ': %{x} <br>' + yaxis_title + ': %{y}'



    # plot the graph

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1[independent_feature], y=df1[feature], 

                             mode="lines", name=df1['Speaker'].values[0], 

                             hovertemplate=hovertemplate, line_color="#e87d23"))



    fig.add_trace(go.Scatter(x=df2[independent_feature], y=df2[feature], 

                             mode="lines", name=df2['Speaker'].values[0], 

                             hovertemplate=hovertemplate, line_color="brown"))

    fig.update_traces(mode="lines+markers")

    

    fig = add_enhancements_to_chart(fig, plt_title, xaxis_title, yaxis_title, start_range, end_range, mean_val)



    

    # show plot

    fig.show()

speakers_timeline_plot(first_episode_nlp_profiled[sanyam_filter], 

                      first_episode_nlp_profiled[abhishek_filter],

                      'sentiment_polarity_score',

                      xaxis_title="Episode E1: Timeline",

                      yaxis_title="Sentiment polarity score",

                      hover_data=["sentiment_polarity", "sentiment_polarity_score"],

                      title='Sentiment polarity: Episode E1 (Host v/s Guest)')
chart = sns.catplot(y="sentiment_polarity", hue="Speaker", data=first_episode_nlp_profiled, size=8, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment polarity moments", fontsize=16)

chart.set_xlabels("Sentiment polarity (in words)", fontsize=16)

chart.fig.suptitle('Host: Sanyam Bhutani & Guest: Abhishek Thakur')
chart = sns.catplot(y="sentiment_polarity_summarised", hue="Speaker", data=first_episode_nlp_profiled, size=8, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment polarity moments", fontsize=16)

chart.set_xlabels("Sentiment polarity (in words)", fontsize=16)

chart.fig.suptitle('Host: Sanyam Bhutani & Guest: Abhishek Thakur (compact)')
timeline_plot(first_episode_nlp_profiled, 

              'sentiment_subjectivity_score', 

              xaxis_title="Episode E1: Timeline",

              yaxis_title="Sentiment subjectivity score",

              hover_data=['sentiment_subjectivity', 'sentiment_subjectivity_score'],

              title="Sentiment subjectivity: Episode E1")
chart = sns.catplot(x="sentiment_subjectivity", data=first_episode_nlp_profiled, size=10, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment subjectivity moments", fontsize=16)

chart.set_xlabels("Sentiment subjectivity (in words)", fontsize=16)

chart.fig.suptitle('Subjectivity, Subjectivity/Objectivity and Objectivity with scales in between')

chart.set_xticklabels(rotation=30)
chart = sns.catplot(x="sentiment_subjectivity_summarised", data=first_episode_nlp_profiled, size=8, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment subjectivity moments", fontsize=16)

chart.set_xlabels("Sentiment subjectivity (in words)", fontsize=16)

chart.fig.suptitle('Subjectivity, Subjectivity/Objectivity and Objectivity (compact view)')
speakers_timeline_plot(first_episode_nlp_profiled[sanyam_filter], 

                      first_episode_nlp_profiled[abhishek_filter],

                      'sentiment_subjectivity_score',

                      xaxis_title="Episode E1: Timeline",

                      yaxis_title="Sentiment subjectivity score",

                      hover_data=["sentiment_subjectivity", "sentiment_subjectivity_score"],

                      title='Sentiment subjectivity: Episode E1 (Host v/s Guest)')
chart = sns.catplot(y="sentiment_subjectivity", hue="Speaker", data=first_episode_nlp_profiled, size=10, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment subjectivity moments", fontsize=16)

chart.set_xlabels("Sentiment subjectivity (in words)", fontsize=16)

chart.fig.suptitle('Host: Sanyam Bhutani & Guest: Abhishek Thakur')
chart = sns.catplot(y="sentiment_subjectivity_summarised", hue="Speaker", data=first_episode_nlp_profiled, size=8, kind="count")

chart.despine(left=True)

chart.set_ylabels("Sentiment subjectivity moments", fontsize=16)

chart.set_xlabels("Sentiment subjectivity (in words)", fontsize=16)

chart.fig.suptitle('Host: Sanyam Bhutani & Guest: Abhishek Thakur (compact)')
chart = sns.catplot(x="spelling_quality", data=first_episode_nlp_profiled, size=10, kind="count")

chart.despine(left=True)

chart.set_ylabels("Spoken moments", fontsize=16)

chart.set_xlabels("Spelling quality (in words)", fontsize=16)

chart.fig.suptitle('Spelling check of the transcripts (Episode E1)')

chart.set_xticklabels(rotation=30)
chart = sns.catplot(x="grammar_check", data=first_episode_nlp_profiled, size=10, kind="count")

chart.despine(left=True)

chart.set_ylabels("Spoken moments", fontsize=16)

chart.set_xlabels("Grammar quality (in words)", fontsize=16)

chart.fig.suptitle('Grammar check of the transcripts (Episode E1)')

chart.set_xticklabels(rotation=30)