import os



import warnings

warnings.simplefilter("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import missingno as msno



import plotly.express as px



import plotly.graph_objects as go

from plotly.subplots import make_subplots



!pip install pywaffle

from pywaffle import Waffle



from bokeh.layouts import column, row

from bokeh.models.tools import HoverTool

from bokeh.models import ColumnDataSource, Whisker

from bokeh.plotting import figure, output_notebook, show



output_notebook()



from IPython.display import IFrame



pd.set_option('display.max_columns', None)
YouTube_df=pd.read_csv("../input/chai-time-data-science/YouTube Thumbnail Types.csv")

print("No of Datapoints : {}\nNo of Features : {}".format(YouTube_df.shape[0], YouTube_df.shape[1]))

YouTube_df.head()
Anchor_df=pd.read_csv("../input/chai-time-data-science/Anchor Thumbnail Types.csv")

print("No of Datapoints : {}\nNo of Features : {}".format(Anchor_df.shape[0], Anchor_df.shape[1]))

Anchor_df.head()
IFrame('https://anchor.fm/chaitimedatascience', width=800, height=450)
Des_df=pd.read_csv("../input/chai-time-data-science/Description.csv")

print("No of Datapoints : {}\nNo of Features : {}".format(Des_df.shape[0], Des_df.shape[1]))

Des_df.head()
def show_description(specific_id=None, top_des=None):

    

    if specific_id is not None:

        print(Des_df[Des_df.episode_id==specific_id].description.tolist()[0])

        

    if top_des is not None:

        for each_des in range(top_des):  

            print(Des_df.description.tolist()[each_des])

            print("-"*100)

show_description("E1")
show_description(top_des=3)
Episode_df=pd.read_csv("../input/chai-time-data-science/Episodes.csv")

print("No of Datapoints : {}\nNo of Features : {}".format(Episode_df.shape[0], Episode_df.shape[1]))

Episode_df.head()
msno.matrix(Episode_df)




temp=Episode_df.isnull().sum().reset_index().rename(columns={"index": "Name", 0: "Count"})

temp=temp[temp.Count!=0]



Source=ColumnDataSource(temp)



tooltips = [

    

    ("Feature Name", "@Name"),

    ("No of Missing entites", "@Count")

]



fig1 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400,tooltips=tooltips, x_range = temp["Name"].values, title = "Count of Missing Values")

fig1.vbar("Name", top = "Count", source = Source, width = 0.4, color = "#76b4bd", alpha=.8)



fig1.xaxis.major_label_orientation = np.pi / 8

fig1.xaxis.axis_label = "Features"

fig1.yaxis.axis_label = "Count"



fig1.grid.grid_line_color="#feffff"





show(fig1)
Episode_df[Episode_df.heroes.isnull()]
temp=[id for id in Episode_df.episode_id if id.startswith('M')]

fastai_df=Episode_df[Episode_df.episode_id.isin(temp)]

Episode_df=Episode_df[~Episode_df.episode_id.isin(temp)]
dummy_df=Episode_df[(Episode_df.episode_id!="E0") & (Episode_df.episode_id!="E69")]
msno.matrix(dummy_df)
temp=dummy_df.isnull().sum().reset_index().rename(columns={"index": "Name", 0: "Count"})

temp=temp[temp.Count!=0]



Source=ColumnDataSource(temp)



tooltips = [

    ("Feature Name", "@Name"),

    ("No of Missing entites", "@Count")

]



fig1 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400,tooltips=tooltips, x_range = temp["Name"].values, title = "Count of Missing Values")

fig1.vbar("Name", top = "Count", source = Source, width = 0.4, color = "#76b4bd", alpha=.8)



fig1.xaxis.major_label_orientation = np.pi / 4

fig1.xaxis.axis_label = "Features"

fig1.yaxis.axis_label = "Count"



fig1.grid.grid_line_color="#feffff"



show(fig1)
parent=[]

names =[]

values=[]

temp=dummy_df.groupby(["category"]).heroes_gender.value_counts()

for k in temp.index:

    parent.append(k[0])

    names.append(k[1])

    values.append(temp.loc[k])



df1 = pd.DataFrame(

    dict(names=names, parents=parent,values=values))





parent=[]

names =[]

values=[]

temp=dummy_df.groupby(["category","heroes_gender"]).heroes_kaggle_username.count()

for k in temp.index:

    parent.append(k[0])

    names.append(k[1])

    values.append(temp.loc[k])



df2 = pd.DataFrame(

    dict(names=names, parents=parent,values=values))





fig = px.sunburst(df1, path=['names', 'parents'], values='values', color='parents',hover_data=["names"], title="Heroes associated with Categories")

fig.update_traces( 

                 textinfo='percent entry+label',

                 hovertemplate = "Industry:%{label}: <br>Count: %{value}"

                )

fig.show()





fig = px.sunburst(df2, path=['names', 'parents'], values='values', color='parents', title="Heroes associated with Categories having Kaggle Account")

fig.update_traces( 

                 textinfo='percent entry+label',

                 hovertemplate = "Industry:%{label}: <br>Count: %{value}"

                )

fig.show()
gender = Episode_df.heroes_gender.value_counts()



fig = plt.figure(

    FigureClass=Waffle, 

    rows=5,

    columns=12,

    values=gender,

    colors = ('#20639B', '#ED553B'),

    title={'label': 'Gender Distribution', 'loc': 'left'},

    labels=["{}({})".format(a, b) for a, b in zip(gender.index, gender) ],

    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(Episode_df), 'framealpha': 0},

    font_size=30, 

    icons = 'child',

    figsize=(12, 5),  

    icon_legend=True

)
dummy_df[dummy_df.apple_listeners.isnull()]
fig = go.Figure([go.Pie(labels=Episode_df.flavour_of_tea.value_counts().index.to_list(),values=Episode_df.flavour_of_tea.value_counts().values,hovertemplate = '<br>Type: %{label}</br>Count: %{value}<br>Popularity: %{percent}</br>', name = '')])

fig.update_layout(title_text="What Host drinks everytime ?", template="plotly_white", title_x=0.45, title_y = 1)

fig.data[0].marker.line.color = 'rgb(255, 255, 255)'

fig.data[0].marker.line.width = 2

fig.update_traces(hole=.4,)

fig.show()
temp=dummy_df.isnull().sum().reset_index().rename(columns={"index": "Name", 0: "Count"})

temp=temp[temp.Count!=0]



Source=ColumnDataSource(temp)



tooltips = [

    ("Feature Name", "@Name"),

    ("No of Missing entites", "@Count")

]



fig1 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400,tooltips=tooltips, x_range = temp["Name"].values, title = "Count of Missing Values")

fig1.vbar("Name", top = "Count", source = Source, width = 0.4, color = "#76b4bd", alpha=.8)



fig1.xaxis.major_label_orientation = np.pi / 4

fig1.xaxis.axis_label = "Features"

fig1.yaxis.axis_label = "Count"



fig1.grid.grid_line_color="#feffff"



show(fig1)
Episode_df.release_date = pd.to_datetime(Episode_df.release_date)

Source = ColumnDataSource(Episode_df)

fastai_df.release_date = pd.to_datetime(fastai_df.release_date)

Source2 = ColumnDataSource(fastai_df)



tooltips = [

    ("Episode Id", "@episode_id"),

    ("Episode Title", "@episode_name"),

    ("Hero Present", "@heroes"),

    ("CTR", "@youtube_ctr"),

    ("Category", "@category"),

    ("Date", "@release_date{%F}"),

    ]



tooltips2 = [

    ("Episode Id", "@episode_id"),

    ("Episode Title", "@episode_name"),

    ("Hero Present", "@heroes"),

    ("Subscriber Gain", "@youtube_subscribers"),

    ("Category", "@category"),

    ("Date", "@release_date{%F}"),

    ]





fig1 = figure(background_fill_color="#ebf4f6",plot_width = 600, plot_height = 400, x_axis_type = "datetime", title = "CTR Per Episode")

fig1.line("release_date", "youtube_ctr", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="youtube_ctr")

fig1.varea(source=Source, x="release_date", y1=0, y2="youtube_ctr", alpha=0.2, fill_color='#55FF88', legend_label="youtube_ctr")

fig1.line("release_date", Episode_df.youtube_ctr.mean(), source = Source, color = "#f2a652", alpha = 0.8,line_dash="dashed", legend_label="Youtube CTR Mean : {:.3f}".format(Episode_df.youtube_ctr.mean()))

fig1.circle(x="release_date", y="youtube_ctr", source = Source2, color = "#5bab37", alpha = 0.8, legend_label="M0-M8 Series")



fig1.add_tools(HoverTool(tooltips=tooltips,formatters={"@release_date": "datetime"}))

fig1.xaxis.axis_label = "Release Date"

fig1.yaxis.axis_label = "Click Per Impression"



fig1.grid.grid_line_color="#feffff"



fig2 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400, x_axis_type = "datetime", title = "Subscriber Gain Per Episode")

fig2.line("release_date", "youtube_subscribers", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="Subscribers")

fig2.varea(source=Source, x="release_date", y1=0, y2="youtube_subscribers", alpha=0.2, fill_color='#55FF88', legend_label="Subscribers")

fig2.circle(x="release_date", y="youtube_subscribers", source = Source2, color = "#5bab37", alpha = 0.8, legend_label="M0-M8 Series")





fig2.add_tools(HoverTool(tooltips=tooltips2,formatters={"@release_date": "datetime"}))

fig2.xaxis.axis_label = "Release Date"

fig2.yaxis.axis_label = "Subscriber Count"



fig2.grid.grid_line_color="#feffff"



show(column(fig1, fig2))
Source = ColumnDataSource(Episode_df)

Source2 = ColumnDataSource(fastai_df)



tooltips = [

    ("Episode Id", "@episode_id"),

    ("Hero Present", "@heroes"),

    ("Impression Views", "@youtube_impression_views"),

    ("Non Impression Views", "@youtube_nonimpression_views"),

    ("Category", "@category"),

    ("Date", "@release_date{%F}"),

    ]



tooltips2 = [

    ("Episode Id", "@episode_id"),

    ("Hero Present", "@heroes"),

    ("Subscriber Gain", "@youtube_subscribers"),

    ("Category", "@category"),

    ("Date", "@release_date{%F}"),

    ]





fig1 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400, x_axis_type = "datetime", title = "Impression-Non Impression Views Per Episode")

fig1.line("release_date", "youtube_impression_views", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="Impression Views")

fig1.line("release_date", "youtube_nonimpression_views", source = Source, color = "#f2a652", alpha = 0.8, legend_label="Non Impression Views")

fig1.varea(source=Source, x="release_date", y1=0, y2="youtube_impression_views", alpha=0.2, fill_color='#55FF88', legend_label="Impression Views")

fig1.varea(source=Source, x="release_date", y1=0, y2="youtube_nonimpression_views", alpha=0.2, fill_color='#e09d53', legend_label="Non Impression Views")

fig1.circle(x="release_date", y="youtube_impression_views", source = Source2, color = "#5bab37", alpha = 0.8, legend_label="M0-M8 Series Impression Views")

fig1.circle(x="release_date", y="youtube_nonimpression_views", source = Source2, color = "#2d3328", alpha = 0.8, legend_label="M0-M8 Series Non Impression Views")







fig1.add_tools(HoverTool(tooltips=tooltips,formatters={"@release_date": "datetime"}))

fig1.xaxis.axis_label = "Release Date"

fig1.yaxis.axis_label = "Total Views"



fig2 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400, x_axis_type = "datetime", title = "Subscriber Gain Per Episode")

fig2.line("release_date", "youtube_subscribers", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="Subscribers")

fig2.varea(source=Source, x="release_date", y1=0, y2="youtube_subscribers", alpha=0.2, fill_color='#55FF88', legend_label="Subscribers")

fig2.circle(x="release_date", y="youtube_subscribers", source = Source2, color = "#5bab37", alpha = 0.8, legend_label="M0-M8 Series")





fig2.add_tools(HoverTool(tooltips=tooltips2,formatters={"@release_date": "datetime"}))

fig2.xaxis.axis_label = "Release Date"

fig2.yaxis.axis_label = "Subscriber Count"





show(column(fig1, fig2))
data1={

      "Youtube Impressions":Episode_df.youtube_impressions.sum(), 

      "Youtube Impression Views": Episode_df.youtube_impression_views.sum(), 

      "Youtube NonImpression Views" : Episode_df.youtube_nonimpression_views.sum()

     }



text=("Youtube Impressions","Youtube Impression Views","Youtube NonImpression Views")



fig = go.Figure(go.Funnelarea(

    textinfo= "text+value",

    text =list(data1.keys()),

    values = list(data1.values()),

    title = {"position": "top center", "text": "Youtube and Views"},

      name = '', showlegend=False,customdata=['Video Thumbnail shown to Someone', 'Views From Youtube Impressions', 'Views without Youtube Impressions'], hovertemplate = '%{customdata} <br>Count: %{value}</br>'

  ))

fig.show()
colors = ["red", "olive", "darkred", "goldenrod"]



index={

    0:"YouTube default image",

    1:"YouTube default image with custom annotation",

    2:"Mini Series: Custom Image with annotations",

    3:"Custom image with CTDS branding, Title and Tags"

}



p = figure(background_fill_color="#ebf4f6", plot_width=600, plot_height=300, title="Thumbnail Type VS CTR")



base, lower, upper = [], [], []



for each_thumbnail_ref in index:

    if each_thumbnail_ref==2:

        temp = fastai_df[fastai_df.youtube_thumbnail_type==each_thumbnail_ref].youtube_ctr 

    else:

        temp = Episode_df[Episode_df.youtube_thumbnail_type==each_thumbnail_ref].youtube_ctr

    mpgs_mean = temp.mean()

    mpgs_std = temp.std()

    lower.append(mpgs_mean - mpgs_std)

    upper.append(mpgs_mean + mpgs_std)

    base.append(each_thumbnail_ref)



    source_error = ColumnDataSource(data=dict(base=base, lower=lower, upper=upper))

    p.add_layout(

        Whisker(source=source_error, base="base", lower="lower", upper="upper")

    )



    tooltips = [

        ("Episode Id", "@episode_id"),

        ("Hero Present", "@heroes"),

        ]



    color = colors[each_thumbnail_ref % len(colors)]

    p.circle(y=temp, x=each_thumbnail_ref, color=color, legend_label=index[each_thumbnail_ref])

    print("Mean CTR for Thumbnail Type {} : {:.3f} ".format(index[each_thumbnail_ref], temp.mean()))

show(p)
a=Episode_df[["episode_id", "episode_duration", "youtube_avg_watch_duration"]]

a["percentage"]=(a.youtube_avg_watch_duration/a.episode_duration)*100



b=fastai_df[["episode_id", "episode_duration", "youtube_avg_watch_duration"]]

b["percentage"]=(b.youtube_avg_watch_duration/b.episode_duration)*100



temp=a.append(b).reset_index().drop(["index"], axis=1)



Source = ColumnDataSource(temp)



tooltips = [

    ("Episode Id", "@episode_id"),

    ("Episode Duration", "@episode_duration"),

    ("Youtube Avg Watch_duration Views", "@youtube_avg_watch_duration"),

    ("Percentage of video watched", "@percentage"),

    ]





fig1 = figure(background_fill_color="#ebf4f6", plot_width = 1000, plot_height = 400, x_range  = temp["episode_id"].values, title = "Percentage of Episode Watched")

fig1.line("episode_id", "percentage", source = Source, color = "#03c2fc", alpha = 0.8)

fig1.line("episode_id", temp.percentage.mean(), source = Source, color = "#f2a652", alpha = 0.8,line_dash="dashed", legend_label="Mean : {:.3f}".format(temp.percentage.mean()))



fig1.add_tools(HoverTool(tooltips=tooltips))

fig1.xaxis.axis_label = "Episode Id"

fig1.yaxis.axis_label = "Percentage"



fig1.xaxis.major_label_orientation = np.pi / 3

show(column(fig1))
colors = ["red", "olive", "darkred", "goldenrod"]



index={

    0:"YouTube default playlist image",

    1:"CTDS Branding",

    2:"Mini Series: Custom Image with annotations",

    3:"Custom image with CTDS branding, Title and Tags"

}



p = figure(background_fill_color="#ebf4f6", plot_width=600, plot_height=300, title="Thumbnail Type VS Anchor Plays")



base, lower, upper = [], [], []



for each_thumbnail_ref in index:

    if each_thumbnail_ref==2:

        temp = fastai_df[fastai_df.youtube_thumbnail_type==each_thumbnail_ref].anchor_plays 

    else:

        temp = Episode_df[Episode_df.youtube_thumbnail_type==each_thumbnail_ref].anchor_plays

    mpgs_mean = temp.mean()

    mpgs_std = temp.std()

    lower.append(mpgs_mean - mpgs_std)

    upper.append(mpgs_mean + mpgs_std)

    base.append(each_thumbnail_ref)



    source_error = ColumnDataSource(data=dict(base=base, lower=lower, upper=upper))

    p.add_layout(

        Whisker(source=source_error, base="base", lower="lower", upper="upper")

    )



    tooltips = [

        ("Episode Id", "@episode_id"),

        ("Hero Present", "@heroes"),

        ]



    color = colors[each_thumbnail_ref % len(colors)]

    p.circle(y=temp, x=each_thumbnail_ref, color=color, legend_label=index[each_thumbnail_ref])

    print("Mean Anchor Plays for Thumbnail Type {} : {:.3f} ".format(index[each_thumbnail_ref], temp.mean()))

show(p)
Episode_df.release_date = pd.to_datetime(Episode_df.release_date)

Source = ColumnDataSource(Episode_df)



tooltips = [

    ("Episode Id", "@episode_id"),

    ("Episode Title", "@episode_name"),

    ("Hero Present", "@heroes"),

    ("Anchor Plays", "@anchor_plays"),

    ("Category", "@category"),

    ("Date", "@release_date{%F}"),

    ]



tooltips2 = [

    ("Episode Id", "@episode_id"),

    ("Episode Title", "@episode_name"),

    ("Hero Present", "@heroes"),

    ("Spotify Starts Plays", "@spotify_starts"),

    ("Spotify Streams", "@spotify_streams"),

    ("Spotify Listeners", "@spotify_listeners"),

    ("Category", "@category"),

    ("Date", "@release_date{%F}"),

    ]





fig1 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400, x_axis_type = "datetime", title = "Anchor Plays Per Episode")

fig1.line("release_date", "anchor_plays", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="Anchor Plays")

fig1.line("release_date", Episode_df.anchor_plays.mean(), source = Source, color = "#f2a652", alpha = 0.8, line_dash="dashed", legend_label="Anchor Plays Mean : {:.3f}".format(Episode_df.youtube_ctr.mean()))





fig1.add_tools(HoverTool(tooltips=tooltips,formatters={"@release_date": "datetime"}))

fig1.xaxis.axis_label = "Release Date"

fig1.yaxis.axis_label = "Anchor Plays"



fig2 = figure(background_fill_color="#ebf4f6", plot_width = 600, plot_height = 400, x_axis_type = "datetime", title = "Performance on Spotify Per Episode")

fig2.line("release_date", "spotify_starts", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="Spotify Starts Plays")

fig2.line("release_date", "spotify_streams", source = Source, color = "#f2a652", alpha = 0.8, legend_label="Spotify Streams")

fig2.line("release_date", "spotify_listeners", source = Source, color = "#03fc5a", alpha = 0.8, legend_label="Spotify Listeners")





fig2.add_tools(HoverTool(tooltips=tooltips2,formatters={"@release_date": "datetime"}))

fig2.xaxis.axis_label = "Release Date"

fig2.yaxis.axis_label = "Total Plays"





show(column(fig1,fig2))
temp=Episode_df.groupby(["heroes_location", "heroes"])["heroes_nationality"].value_counts()



parent=[]

names =[]

values=[]

heroes=[]

for k in temp.index:

    parent.append(k[0])

    heroes.append(k[1])

    names.append(k[2])

    values.append(temp.loc[k])



df = pd.DataFrame(

    dict(names=names, parents=parent,values=values, heroes=heroes))

df["World"] = "World"



fig = px.treemap(

    df,

    path=['World', 'parents','names','heroes'], values='values',color='parents')



fig.update_layout( 

    width=1000,

    height=700,

    title_text="Distribution of Heores by Country and Nationality")

fig.show()
a=Episode_df.release_date

b=(a-a.shift(periods=1, fill_value='2019-07-21')).astype('timedelta64[D]')

d = {'episode_id':Episode_df.episode_id, 'heroes':Episode_df.heroes, 'release_date': Episode_df.release_date, 'day_difference': b}

temp = pd.DataFrame(d)



Source = ColumnDataSource(temp)



tooltips = [

    ("Episode Id", "@episode_id"),

    ("Hero Present", "@heroes"),

    ("Day Difference", "@day_difference"),

    ("Date", "@release_date{%F}"),

    ]



fig1 = figure(background_fill_color="#ebf4f6", plot_width = 1000, plot_height = 400, x_axis_type  = "datetime", title = "Day difference between Each Release Date")

fig1.line("release_date", "day_difference", source = Source, color = "#03c2fc", alpha = 0.8)



fig1.add_tools(HoverTool(tooltips=tooltips,formatters={"@release_date": "datetime"}))

fig1.xaxis.axis_label = "Date"

fig1.yaxis.axis_label = "No of Days"



fig1.xaxis.major_label_orientation = np.pi / 3

show(column(fig1))
def show_script(id):

    return pd.read_csv("../input/chai-time-data-science/Cleaned Subtitles/{}.csv".format(id))
df = show_script("E1")

df
# feature engineer the transcript features

def conv_to_sec(x):

    """ Time to seconds """



    t_list = x.split(":")

    if len(t_list) == 2:

        m = t_list[0]

        s = t_list[1]

        time = int(m) * 60 + int(s)

    else:

        h = t_list[0]

        m = t_list[1]

        s = t_list[2]

        time = int(h) * 60 * 60 + int(m) * 60 + int(s)

    return time





def get_durations(nums, size):

    """ Get durations i.e the time for which each speaker spoke continuously """



    diffs = []

    for i in range(size - 1):

        diffs.append(nums[i + 1] - nums[i])

    diffs.append(30)  # standard value for all end of the episode CFA by Sanyam

    return diffs





def transform_transcript(sub, episode_id):

    """ Transform the transcript of the given episode """



    # create the time second feature that converts the time into the unified qty. of seconds

    sub["Time_sec"] = sub["Time"].apply(conv_to_sec)



    # get durations

    sub["Duration"] = get_durations(sub["Time_sec"], sub.shape[0])



    # providing an identity to each transcript

    sub["Episode_ID"] = episode_id

    sub = sub[["Episode_ID", "Time", "Time_sec", "Duration", "Speaker", "Text"]]



    return sub





def combine_transcripts(sub_dir):

    """ Combine all the 75 transcripts of the ML Heroes Interviews together as one dataframe """



    episodes = []

    for i in range(1, 76):

        file = "E" + str(i) + ".csv"

        try:

            sub_epi = pd.read_csv(os.path.join(sub_dir, file))

            sub_epi = transform_transcript(sub_epi, ("E" + str(i)))

            episodes.append(sub_epi)

        except:

            continue

    return pd.concat(episodes, ignore_index=True)





# create the combined transcript dataset

sub_dir = "../input/chai-time-data-science/Cleaned Subtitles"

transcripts = combine_transcripts(sub_dir)

transcripts.head()
temp = Episode_df[["episode_id","youtube_avg_watch_duration"]]

temp=temp[(temp.episode_id!="E0") & (temp.episode_id!="E4")]



intro=[]



for i in transcripts.Episode_ID.unique():

    intro.append(transcripts[transcripts.Episode_ID==i].iloc[0].Duration)

temp["Intro_Duration"]=intro

temp["diff"]=temp.youtube_avg_watch_duration-temp.Intro_Duration



Source = ColumnDataSource(temp)



tooltips = [

    ("Episode Id", "@episode_id"),

    ("Youtube Avg Watch_duration Views", "@youtube_avg_watch_duration"),

    ("Intro Duration", "@Intro_Duration"),

    ("Avg Duration of Content Watched", "@diff"),

    ]





fig1 = figure(background_fill_color="#ebf4f6", plot_width = 1000, plot_height = 600, x_range  = temp["episode_id"].values, title = "Impact of Intro Durations")

fig1.line("episode_id", "youtube_avg_watch_duration", source = Source, color = "#03c2fc", alpha = 0.8, legend_label="Youtube Avg Watch_duration Views")

fig1.line("episode_id", "Intro_Duration", source = Source, color = "#f2a652", alpha = 0.8, legend_label="Intro Duration")

fig1.line("episode_id", "diff", source = Source, color = "#03fc5a", alpha = 0.8, legend_label="Avg Duration of Content Watched")





fig1.add_tools(HoverTool(tooltips=tooltips))

fig1.xaxis.axis_label = "Episode Id"

fig1.yaxis.axis_label = "Percentage"



fig1.xaxis.major_label_orientation = np.pi / 3

show(column(fig1))
print("{:.2f} % of Episodes have Avg Duration of Content Watched less than 5 minutes".format(len(temp[temp["diff"]<300])/len(temp)*100))

print("{:.2f} % of Episodes have Avg Duration of Content Watched less than 4 minutes".format(len(temp[temp["diff"]<240])/len(temp)*100))

print("{:.2f} % of Episodes have Avg Duration of Content Watched less than 3 minutes".format(len(temp[temp["diff"]<180])/len(temp)*100))

print("{:.2f} % of Episodes have Avg Duration of Content Watched less than 2 minutes".format(len(temp[temp["diff"]<120])/len(temp)*100))

print("In {} case, Viewer left in the Intro Duration".format(len(temp[temp["diff"]<0])))
host_text = []

hero_text = []

for i in transcripts.Episode_ID.unique():

    host_text.append([i, transcripts[(transcripts.Episode_ID==i) & (transcripts.Speaker=="Sanyam Bhutani")].Text])

    hero_text.append([i, transcripts[(transcripts.Episode_ID==i) & (transcripts.Speaker!="Sanyam Bhutani")].Text])



temp_host={}

temp_hero={}

for i in range(len(transcripts.Episode_ID.unique())):

    host_text_count = len(host_text[i][1])

    hero_text_count = len(hero_text[i][1])

    temp_host[hero_text[i][0]]=host_text_count

    temp_hero[hero_text[i][0]]=hero_text_count

    

def getkey(dict): 

    list = [] 

    for key in dict.keys(): 

        list.append(key)          

    return list



def getvalue(dict): 

    list = [] 

    for key in dict.values(): 

        list.append(key)          

    return list
Source = ColumnDataSource(data=dict(

    x=getkey(temp_host),

    y=getvalue(temp_host),

    a=getkey(temp_hero),

    b=getvalue(temp_hero),

))



tooltips = [

    ("Episode Id", "@x"),

    ("No of Times Host Speaks", "@y"),

    ("No of Times Hero Speaks", "@b"),

]



fig1 = figure(background_fill_color="#ebf4f6",plot_width = 1000, tooltips=tooltips,plot_height = 400, x_range = getkey(temp_host), title = "Who Speaks More ?")

fig1.vbar("x", top = "y", source = Source, width = 0.4, color = "#76b4bd", alpha=.8, legend_label="No of Times Host Speaks")

fig1.vbar("a", top = "b", source = Source, width = 0.4, color = "#e7f207", alpha=.8, legend_label="No of Times Hero Speaks")



fig1.xaxis.axis_label = "Episode"

fig1.yaxis.axis_label = "Count"



fig1.grid.grid_line_color="#feffff"

fig1.xaxis.major_label_orientation = np.pi / 4



show(fig1)
ques=0

total_ques={}

for episode in range(len(transcripts.Episode_ID.unique())):

    for each_text in range(len(host_text[episode][1])):

        ques += host_text[episode][1].reset_index().iloc[each_text].Text.count("?")

    total_ques[hero_text[episode][0]]= ques

    ques=0
from statistics import mean 

Source = ColumnDataSource(data=dict(

    x=getkey(total_ques),

    y=getvalue(total_ques),

))



tooltips = [

    ("Episode Id", "@x"),

    ("No of Questions", "@y"),

]



fig1 = figure(background_fill_color="#ebf4f6",plot_width = 1000, plot_height = 400,tooltips=tooltips, x_range = getkey(temp_host), title = "Questions asked Per Episode")

fig1.vbar("x", top = "y", source = Source, width = 0.4, color = "#76b4bd", alpha=.8, legend_label="No of Questions asked Per Episode")

fig1.line("x", mean(getvalue(total_ques)), source = Source, color = "#f2a652", alpha = 0.8,line_dash="dashed", legend_label="Average Questions : {:.3f}".format(mean(getvalue(total_ques))))



fig1.xaxis.axis_label = "Episode"

fig1.yaxis.axis_label = "No of Questions"



fig1.legend.location = "top_left"



fig1.grid.grid_line_color="#feffff"

fig1.xaxis.major_label_orientation = np.pi / 4



show(fig1)
import re

import nltk

from statistics import mean 

from collections import Counter

import string



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text
transcripts['Text'] = transcripts['Text'].apply(str).apply(lambda x: text_preprocessing(x))
def get_data(speakername=None):

    label=[]

    value=[]



    text_data=transcripts[(transcripts.Speaker==speakername)].Text.tolist()

    temp=list(filter(lambda x: x.count(" ")<10 , text_data)) 



    freq=nltk.FreqDist(temp).most_common(7)

    for each in freq:

        label.append(each[0])

        value.append(each[1])

        

        

    Source = ColumnDataSource(data=dict(

        x=label,

        y=value,

    ))



    tooltips = [

        ("Favourite Text", "@x"),

        ("Frequency", "@y"),

    ]



    fig1 = figure(background_fill_color="#ebf4f6",plot_width = 600, tooltips=tooltips, plot_height = 400, x_range = label, title = "Favourite Text")

    fig1.vbar("x", top = "y", source = Source, width = 0.4, color = "#76b4bd", alpha=.8)



    fig1.xaxis.axis_label = "Text"

    fig1.yaxis.axis_label = "Frequency"





    fig1.grid.grid_line_color="#feffff"

    fig1.xaxis.major_label_orientation = np.pi / 4



    show(fig1)

get_data(speakername="Sanyam Bhutani")