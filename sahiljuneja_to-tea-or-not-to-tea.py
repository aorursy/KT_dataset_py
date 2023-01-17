import pandas as pd

from IPython.display import display

pd.set_option('display.max_rows', 76)

pd.set_option('display.max_columns', 40)

pd.set_option('display.max_colwidth', 1000)

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



#plotly

# thanks to https://www.kaggle.com/parulpandey/how-to-explore-the-ctds-show-data

!pip install dexplot

!pip install chart_studio

import dexplot as dxp

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot
predictions = pd.read_csv("../input/ctds-audio-emotions/predictions.csv")

predictions = predictions.drop(columns=["Unnamed: 0"])
predictions.iloc[4803:4808]
predictions.info()
predictions[~predictions["pred_num"].isnull() & predictions["pred_label"].isnull()].head(5)
data_dict = {0.0: 'angry', 1.0: 'calm', 2.0: 'disgust', 3.0: 'fearful', 4.0: 'happy', 5.0: 'neutral', 6.0: 'sad', 7.0: 'surprised'}



def pred_label(row):

    if row in data_dict:

        return data_dict[row]

    return np.nan

        

predictions["pred_label"] = predictions["pred_num"].apply(pred_label)

predictions[predictions["pred_label"] == "angry"].sample(4)
predictions.info()
predictions[predictions["pred_num"].isnull()]["speaker"].value_counts().nlargest(3)
predictions["pred_num_new"] = predictions["pred_num"].fillna(5.0, axis=0)

predictions["pred_label_new"] = predictions["pred_label"].fillna("neutral", axis=0)
predictions.info()
predictions = predictions[(predictions["episode_num"] != 46) & (predictions["episode_num"] != 4)]

predictions.info()
sanyam_emotion = predictions[predictions["speaker"]=="Sanyam Bhutani"].groupby("episode_num")["pred_label"].apply(lambda val: val.value_counts())

color_map = {'angry': 'red', 'calm': 'cyan', 'disgust': 'pink', 'fearful': 'black', 

             'happy': 'orange', 'neutral': 'green', 'sad': 'grey', 'surprised': 'black'}

fig = px.bar(sanyam_emotion.unstack() , color_discrete_map=color_map, title="Count of Predicted Emotions from Sanyam's Audio Clips")

fig.update_xaxes(title_text='Episode Number')

fig.update_yaxes(title_text='Number of Predictions per Emotion')

fig.update_layout(legend_title_text='Emotions')

fig.update_layout(barmode='stack')

fig.show()
sanyam_emotion = predictions[predictions["speaker"]=="Sanyam Bhutani"].groupby("episode_num")["pred_label"].apply(lambda val: 100*val.value_counts()/val.value_counts().sum())

fig = px.bar(sanyam_emotion.unstack(), color_discrete_map=color_map,  title="Percentage of Predicted Emotions from Sanyam's Audio Clips")

fig.update_xaxes(title_text='Episode Number')

fig.update_yaxes(title_text='Proportion of each Emotion')

fig.update_layout(legend_title_text='Emotions')

fig.update_layout(barmode='stack')

fig.show()
emotion_percent = predictions[predictions["speaker"]=="Sanyam Bhutani"].groupby("episode_num")["pred_label"].apply(lambda val: 100*val.value_counts().nlargest(1)/val.value_counts().sum()).sort_values(ascending=False)



emotion_percent.head(10)
episodes = pd.read_csv("../input/ctds-audio-emotions/Episodes_cleaned.csv")





columns_to_use = ["episode_id","flavour_of_tea","recording_time", "episode_duration"]

episodes[columns_to_use].head()
episodes = episodes[columns_to_use]

episodes["episode_num"] = episodes["episode_id"].str[1:].astype(int)

episodes = episodes.drop(columns="episode_id")

pred_merged = predictions.merge(episodes, how="inner", on="episode_num")

pred_merged.head(5)
# nlargest is 2 to avoid NaN values for "happy"

happiness_recording_time = pred_merged[pred_merged["speaker"]=="Sanyam Bhutani"].groupby(["episode_num", "recording_time"])["pred_label"].apply(lambda val: 100*val.value_counts().nlargest(2)/val.value_counts().sum())

happiness_recording_time = happiness_recording_time.unstack().reset_index()

fig = px.bar(happiness_recording_time, x="recording_time", title="Number of Episodes per Time of Day")

fig.update_xaxes(title_text='Time of Day')

fig.update_yaxes(title_text='Number of Episodes')

fig.show()
fig = px.scatter(happiness_recording_time, x="recording_time", color="happy",

                 size="happy", hover_data=["happy"],

                 title="Relationship between Time of Day and how happy the host was")

fig.update_xaxes(title_text='Time of Day')

fig.update_yaxes(title_text='Episode Number')

fig.show()
happiness_tea_flavour = pred_merged[pred_merged["speaker"]=="Sanyam Bhutani"].groupby(["episode_num", "flavour_of_tea"])["pred_label"].apply(lambda val: 100*val.value_counts().nlargest(2)/val.value_counts().sum())

happiness_tea_flavour = happiness_tea_flavour.unstack().reset_index()

happiness_tea_flavour = happiness_tea_flavour[["episode_num", "flavour_of_tea", "happy"]]
happiness_tea_flavour.sort_values(by="happy", ascending=False).head(10)
happiness_tea_flavour.sort_values(by="happy").head(5)
fig = px.scatter(happiness_tea_flavour, x="flavour_of_tea", color="happy",

                 size="happy", hover_data=["happy"],

                 title="Relationship between flavour of tea and how happy the host was")

fig.update_xaxes(title_text='Tea Flavour')

fig.update_yaxes(title_text='Episode Number')

fig.show()
happiness_tea_flavour.groupby("flavour_of_tea").apply(lambda val: val[val["happy"] > 75.0])["flavour_of_tea"].value_counts().head(3)
happiness_tea_flavour.groupby("flavour_of_tea").apply(lambda val: val[val["happy"] < 75.0])["flavour_of_tea"].value_counts().head(3)
pred_merged_sanyam = pred_merged[(pred_merged["speaker"]=="Sanyam Bhutani")]





fig = px.scatter(pred_merged_sanyam, x="episode_clip", y="pred_label_new", color="pred_label_new", color_discrete_map=color_map,

                 title="Emotional Journey through an Episode", labels={"pred_label_new": "Prediction Label"},

               animation_frame='episode_num')

fig.update_xaxes(title_text='Episode Clips')

fig.update_yaxes(title_text='Prediction Label')

# https://community.plotly.com/t/how-to-slow-down-animation-in-plotly-express/31309/6

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

fig.show()

# emotion_episode_clip

happiness_episode_duration = pred_merged[pred_merged["speaker"]=="Sanyam Bhutani"].groupby(["episode_num", "episode_duration"]).apply(lambda val: 100*val["pred_label"].value_counts().nlargest(2)/val["pred_label"].value_counts().sum())

happiness_episode_duration = happiness_episode_duration.unstack().reset_index()

happiness_episode_duration = happiness_episode_duration[["episode_num", "episode_duration", "happy"]]

fig = px.scatter(happiness_episode_duration, x="episode_num",y="episode_duration", color="happy",

                 size="episode_duration", hover_data=["happy", "episode_duration"],

                 title="Relationship between episode duration and how happy the host was")

fig.update_xaxes(title_text='Episode Number')

fig.update_yaxes(title_text='Episode Duration')

fig.show()
happiness_episode_duration.sort_values(by=["happy"], ascending=False).head(5)