from pathlib import Path



import pandas as pd

import plotly_express as px



INPUT_FOLDER = Path("../input/")
df = pd.read_csv(INPUT_FOLDER / "news_collection.csv", parse_dates=["date"])

df.sample(5)
f"Number of entries: {df.shape[0]:,d}"
df[df.duplicated(["title", "desc", "image", "url", "source"], keep=False)].sort_values(["url", "date"]).head()
df[

    df.duplicated(["title", "desc", "image", "url", "source"], keep=False)

].groupby(["title", "url", "source"]).size().to_frame("cnt").sort_values("cnt", ascending=False).head()
blacklists = [

    "https://www.sgsme.sg/", "https://www.voachinese.com/", "https://www.voachinese.com/z/5102", 

    "https://www.instagram.com/voachinese/", "https://cn.wsj.com/", "https://www.wsj.com/europe"

]
df = df[~df.duplicated(["title", "desc", "image", "url", "source"], keep="first")]

df.shape[0]
df = df[~df.url.isin(blacklists)]

df.shape[0]
df = df[~df.url.str.startswith("https://www.youtube.com")]

df.shape[0]
import re

def cjk_detect(texts):

    texts = str(texts)

    # korean

    if re.search("[\uac00-\ud7a3]", texts):

        return "ko"

    # japanese

    if re.search("[\u3040-\u30ff]", texts):

        return "ja"

    # chinese

    if re.search("[\u4e00-\u9FFF]", texts):

        return "zh"

    return "others"
%%time

df["lang"] = df.apply(cjk_detect, axis=1)
df["lang"].value_counts()
df[df["lang"]=="ko"].head()
df[df["lang"]=="others"].sample(5)
df[df["lang"]=="ja"].sample(5)
df[df["lang"]=="zh"].sample(5)
print("Before:", df.shape[0])

df = df[df.lang.isin(("ja", "zh"))].copy()

print("After:", df.shape[0])
source_counts = df.source.value_counts().to_frame("Count").reset_index()



px.bar(

    source_counts, x="index", y="Count", template="plotly_white",

    labels=dict(Count="Number of Entries", index="Source"), 

    width=800, height=400, title="# of News Entries by Source"

)
date_counts = df.date.value_counts().to_frame("Count").reset_index()

date_counts["index"] = date_counts["index"].dt.strftime("%Y-%m-%d")



px.bar(

    date_counts, x="index", y="Count", template="plotly_white",

    labels=dict(Count="Number of Entries", index="Date"), 

    width=800, height=400, title="Article Counts by Date"

)
date_counts = df[df.source == "NYTimes"].date.value_counts().to_frame("Count").reset_index()

date_counts["index"] = date_counts["index"].dt.strftime("%Y-%m-%d")



px.bar(

    date_counts, x="index", y="Count", template="plotly_white",

    labels=dict(Count="Number of Entries", index="Date"), 

    width=800, height=400, title="New York Times (CN) Article Counts by Date"

)
df["trump"] = (

    df.title.str.contains("川普") |

    df.title.str.contains("特朗普")

)

f'% of titles mentioning Trump: {df["trump"].sum() / df.shape[0] * 100:.2f}%'
trump_perc_by_source = df.groupby("source")["trump"].mean().sort_values() * 100

trump_perc_by_source = trump_perc_by_source.to_frame("Perc").reset_index()



px.bar(

    trump_perc_by_source, x="Perc", y="source", template="plotly_white",

    labels=dict(Perc="%", source="Source"), 

    width=800, height=400, title="Percentage of Titles mentioning Trump by Sources",

    orientation="h"

)
trump_perc_by_date = df.groupby("date")["trump"].mean().sort_values() * 100

trump_perc_by_date = trump_perc_by_date.to_frame("Perc").reset_index()

trump_perc_by_date["date"] = trump_perc_by_date["date"].dt.strftime("%Y-%m-%d")



px.bar(

    trump_perc_by_date, x="date", y="Perc", template="plotly_white",

    labels=dict(Perc="%", date="Date"), 

    width=800, height=400, title="Percentage of Titles mentiong Trump by Date",

    orientation="v"

)
df["xi"] = (

    df.title.str.contains("習近平") |

    df.title.str.contains("习近平")

)

f'% of titles mentioning Xi: {df["xi"].sum() / df.shape[0] * 100:.2f}%'
xi_perc_by_source = df.groupby("source")["xi"].mean().sort_values() * 100

xi_perc_by_source = xi_perc_by_source.to_frame("Perc").reset_index()



px.bar(

    xi_perc_by_source, x="Perc", y="source", template="plotly_white",

    labels=dict(Perc="%", source="Source"), 

    width=800, height=400, title="Percentage of Titles mentioning Xi by Sources",

    orientation="h"

)
xi_perc_by_date = df.groupby("date")["xi"] .mean().sort_values() * 100

xi_perc_by_date = xi_perc_by_date.to_frame("Perc").reset_index()

xi_perc_by_date["date"] = xi_perc_by_date["date"].dt.strftime("%Y-%m-%d")



px.bar(

    xi_perc_by_date, x="date", y="Perc", template="plotly_white",

    labels=dict(Perc="%", date="Date"), 

    width=800, height=400, title="Percentage of Titles mentiong Xi by Date",

    orientation="v"

)
df[(df.date == "2019-01-03") & df.xi][["title", "desc", "source"]].sample(5)
xi_perc_by_source["poi"] = "Xi"

trump_perc_by_source["poi"] = "Trump"

combined = pd.concat([trump_perc_by_source, xi_perc_by_source], axis=0)

px.bar(

    combined, x="Perc", y="source", color="poi", template="plotly_white",

    labels=dict(Perc="%", source="Source"),

    width=800, height=400, title="Percentage of Title Mentions by Sources",

    orientation="h", barmode="group"

)
xi_perc_by_source["poi"] = "Xi"

trump_perc_by_source["poi"] = "Trump"

combined = pd.concat([trump_perc_by_source, xi_perc_by_source], axis=0)

px.bar(

    combined, x="Perc", y="source", template="plotly_white",

    labels=dict(Perc="%", source="Source"), facet_col="poi",

    width=800, height=600, title="Percentage of Title Mentions by Sources",

    orientation="h"

)