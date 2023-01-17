import pandas as pd

from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go



init_notebook_mode(connected=True)
df = pd.read_csv("../input/news_collection.csv", parse_dates=["date"])

df.sample(5)
f"Number of entries: {df.shape[0]:,d}"
df[df.duplicated(["title", "desc", "image", "url", "source"], keep=False)].sort_values(["url", "date"]).head()
df[

    df.duplicated(["title", "desc", "image", "url", "source"], keep=False)

].groupby(["title", "url", "source"]).size().to_frame("cnt").sort_values("cnt", ascending=False).head()
blacklists = [

    "https://www.sgsme.sg/", "https://www.voachinese.com/", "https://www.voachinese.com/z/5102", 

    "https://www.instagram.com/voachinese/", "https://cn.wsj.com/"

]
df = df[~df.duplicated(["title", "desc", "image", "url", "source"], keep="first")]

df.shape[0]
df = df[~df.url.isin(blacklists)]

df.shape[0]
df = df[~df.url.str.startswith("https://www.youtube.com")]

df.shape[0]
source_counts = df.source.value_counts()



iplot(go.Figure(

    data=[

        go.Bar(

            x=source_counts.index,

            y=source_counts.values

        )

    ], 

    layout=go.Layout(

        title='Article Counts by Source',

        width=800, height=400, template="plotly_white"

    )

))
date_counts = df.date.value_counts()



iplot(go.Figure(

    data=[

        go.Bar(

            x=date_counts.index.strftime("%Y-%m-%d"),

            y=date_counts.values

        )

    ], 

    layout=go.Layout(

        title='Article Counts by Date',

        width=800, height=400, template="plotly_white"

    )

))
date_counts = df[df.source == "NYTimes"].date.value_counts()



iplot(go.Figure(

    data=[

        go.Bar(

            x=date_counts.index.strftime("%Y-%m-%d"),

            y=date_counts.values

        )

    ], 

    layout=go.Layout(

        title='New York Times (CN) Article Counts by Date',

        width=800, height=400, template="plotly_white"

    )

))
df["trump"] = (

    df.title.str.contains("川普") |

    df.title.str.contains("特朗普")

)

f'% of titles mentioning Trump: {df["trump"].sum() / df.shape[0] * 100:.2f}%'
trump_perc_by_source = df.groupby("source")["trump"] .mean().sort_values() * 100



iplot(go.Figure(

    data=[

        go.Bar(

            y=trump_perc_by_source.index,

            x=trump_perc_by_source.values,

            orientation = 'h'

        )

    ], 

    layout=go.Layout(

        title='Percentage of Titles mentioning Trump by Sources',

        xaxis=dict(

            title='%',

        ),

        width=800, height=400, template="plotly_white"

    )

))
turmp_perc_by_date = df.groupby("date")["trump"] .mean().sort_values() * 100



iplot(go.Figure(

    data=[

        go.Bar(

            x=turmp_perc_by_date.index.strftime("%Y-%m-%d"),

            y=turmp_perc_by_date.values

        )

    ], 

    layout=go.Layout(

        title='Percentage of Titles mentiong Trump by Date',

        yaxis=dict(

            title='%',

        ),

        width=800, height=400, template="plotly_white"

    )

))
df["xi"] = (

    df.title.str.contains("習近平") |

    df.title.str.contains("习近平")

)

f'% of titles mentioning Xi: {df["xi"].sum() / df.shape[0] * 100:.2f}%'
xi_perc_by_source = df.groupby("source")["xi"] .mean().sort_values() * 100



iplot(go.Figure(

    data=[

        go.Bar(

            y=xi_perc_by_source.index,

            x=xi_perc_by_source.values,

            orientation = 'h'

        )

    ], 

    layout=go.Layout(

        title='Percentage of Titles mentioning Xi by Sources',

        xaxis=dict(

            title='%',

        ),

        width=800, height=400, template="plotly_white"

    )

))
xi_perc_by_source = df.groupby("source")["xi"] .mean().sort_values() * 100



iplot(go.Figure(

    data=[

        go.Bar(

            y=trump_perc_by_source.index,

            x=trump_perc_by_source.values,

            orientation = 'h',

            name="Trump"

        ),

        go.Bar(

            y=xi_perc_by_source.index,

            x=xi_perc_by_source.values,

            orientation = 'h',

            name="Xi"

        ),        

    ], 

    layout=go.Layout(

        title='Percentage of Titles mentioning Xi and Trump by Sources',

        xaxis=dict(

            title='%',

        ),

        width=800, height=600, template="plotly_white"

    )

))
xi_perc_by_date = df.groupby("date")["xi"] .mean().sort_values() * 100



iplot(go.Figure(

    data=[

        go.Bar(

            x=xi_perc_by_date.index.strftime("%Y-%m-%d"),

            y=xi_perc_by_date.values

        )

    ], 

    layout=go.Layout(

        title='Percentage of Titles mentiong Xi by Date',

        yaxis=dict(

            title='%',

        ),

        width=800, height=400, template="plotly_white"

    ),

))
df[(df.date == "2019-01-03") & df.xi][["title", "desc", "source"]].sample(5)