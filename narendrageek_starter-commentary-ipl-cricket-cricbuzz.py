import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





import plotly.express as px

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



from PIL import Image

import requests

from io import BytesIO



DATA_DIR = "/kaggle/input/can-generate-automatic-commentary-for-ipl-cricket/"



for dirname, _, filenames in os.walk(DATA_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ipl_schedule_df = pd.read_csv(os.path.join(DATA_DIR, "IPL_SCHEDULE_2008_2020.csv"))

ipl_highlights_df = pd.read_csv(os.path.join(DATA_DIR, "IPL_Match_Highlights_Commentary.csv"))

print(f"IPL Schedule: {ipl_schedule_df.shape}, IPL Highlights {ipl_highlights_df.shape}")
ipl_schedule_df.head(5)
ipl_schedule_df.head(5)
match_distribution = ipl_schedule_df["IPL_year"].value_counts() 



fig = px.bar(match_distribution, title="No of match held on every Year")

fig.update_layout(

    xaxis_title="IPL Year",

    yaxis_title="No of match")

fig.show()
stadium_distribution = ipl_schedule_df["Stadium"].value_counts() 



fig = px.bar(stadium_distribution, title="Stadium Vs no of match")

fig.update_layout(

    xaxis_title="Stadium",

    yaxis_title="No of match held")



fig.show()
ipl_schedule_df.groupby(["Location"])["Stadium"].nunique().reset_index().sort_values("Stadium", ascending=False).reset_index(drop=True)
## Mumbai stadiums

ipl_schedule_df[ipl_schedule_df["Location"] == "Mumbai"]["Stadium"].unique()
stadium_distribution = ipl_schedule_df["Location"].value_counts() 



fig = px.bar(stadium_distribution, title="Location Vs no of match")

fig.update_layout(

    xaxis_title="Location",

    yaxis_title="No of match held")



fig.show()
ipl_schedule_df["Highlights_available"].value_counts()
match_highlights_date = ipl_schedule_df[ipl_schedule_df["Highlights_available"] == True]["Match_Date"]

print(f"The match highlights available from {match_highlights_date.min()} to {match_highlights_date.max()}")
## Join the Schedule info with Highlights info using match_id
ipl_schedule_highlights_available = ipl_schedule_df[ipl_schedule_df["Highlights_available"] == True]
ipl_schedule_commentary_df = pd.merge(ipl_highlights_df, ipl_schedule_highlights_available, on="Match_id")

ipl_schedule_commentary_df.columns
stopwords = set(STOPWORDS)





wordcloud = WordCloud(stopwords=stopwords, contour_width=3, contour_color='steelblue', background_color="white", max_words=1000)

wordcloud.generate(",".join(ipl_schedule_commentary_df["Commentary"].tolist()))



plt.figure(figsize = (20,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title("English word distribution",fontdict={"fontsize":20}, pad=2)

plt.show()
score_distribution = ipl_schedule_commentary_df["Run_scored"].value_counts() 



fig = px.bar(score_distribution, title="Run Distribution")

fig.update_layout(

    xaxis_title="Run",

    yaxis_title="Frequency")



fig.show()