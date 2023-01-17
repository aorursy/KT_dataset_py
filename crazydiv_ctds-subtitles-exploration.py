!pip install chart_studio --quiet
import pandas as pd

import os

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

import chart_studio.plotly as py

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import cufflinks as cf

from sklearn.cluster import KMeans

import tensorflow_hub as hub

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



from plotly.offline import iplot

pd.options.plotting.backend = "plotly"
HARDCODE_LAST_SPEAK_DURATION = 40

BASE_PATH = "../input/chai-time-data-science/"

BASE_CSV_PATH = "../input/generating-cleaner-sublititles-file/"

df_episodes = pd.read_csv(BASE_PATH+"Episodes.csv")
def merge_consecutive_data(df):

    """If same speaker content is in succession, merge it into a single row to make further analysis simpler"""

    new_idx = 0

    rows = []

    for idx,row in df.iterrows():

        if new_idx == 0:

            prev_row = row

            new_idx += 1

            continue

        if prev_row["Speaker"] == row["Speaker"]:

            prev_row["Text"] += row["Text"]

        else:

            rows.append(prev_row)

            prev_row = row

            prev_row["relative_index"] = new_idx

            new_idx += 1

        if idx == df.shape[0] - 1:

            rows.append(prev_row)

    df = pd.DataFrame(rows)

    return df



def get_time(t):

    """hh:mm:ss to seconds. Since time isn't formatted nicely in cleaned data, using hacks"""

    ts = t.split(":")

    total = 0

    for idx, data in enumerate(ts[::-1]):

        total += int(data) * 60**idx

    return total        



def calculate_speak_duration(df, df_episodes, episode_id):

    """Calculate speaker duration in seconds and also in time elapsed since 0"""

    df = df.reset_index()

    df["duration"] = None

    df["timestamp_relative"] = None

    prev_time = None

    for idx, row in df.iterrows():

        if idx == 0:

            prev_time = get_time(row["Time"])

            df.loc[idx, "timestamp_relative"] = prev_time

            continue

        curr_time = get_time(row["Time"])

        df.loc[idx-1, "duration"] = curr_time - prev_time

        prev_time = curr_time

        df.loc[idx, "timestamp_relative"] = curr_time

    # Hardcoding because for some of the cases subtracting from episode duration is producing too large numbers than practically expected

#     df.loc[idx, "duration"] = df_episodes[df_episodes["episode_id"] == episode_id]["episode_duration"].values[0] - prev_time

    df.loc[idx, "duration"] = HARDCODE_LAST_SPEAK_DURATION

    return df
df = pd.DataFrame()

for f in os.listdir(BASE_PATH+"Cleaned Subtitles"):

    if f == "E69.csv":

        continue

    df_e = pd.read_csv(BASE_PATH+'Cleaned Subtitles/'+f)

    

    df_e["relative_index"] = df_e.index

    df_e["episode_id"] = f.split(".csv")[0]

    df_e = merge_consecutive_data(df_e)

    df_e = calculate_speak_duration(df_e, df_episodes, f.split(".csv")[0])

    df = df.append(df_e)

del df["index"]


print("Shape of all data:", df.shape)
df.head(2)
def get_words_list(x):

    """For an input text, returns list of words used"""

    x_list = x.replace(".", " ").replace(",", " ").lower().split()

    return x_list



def get_num_words(x):

    """For an input text, returns number of words used"""

    x_list = x.replace(".", " ").replace(",", " ").lower().split()

    return len(x_list)



def get_num_sentences(x):

    """For an input text, gets number of sentences"""

    num_sentences = len(x.split("."))

    return num_sentences



def get_avg_sentence_len(x):

    """For an input text containing multiple sentences, returns the average length of sentences used"""

    sentences = x.split(".")

    sentences_splitted = [len(s.split(" ")) for s in sentences]

    return sum(sentences_splitted) / len(sentences_splitted)

def plot_single_episode_timeline(episode_id, df, df_episodes, max_tick=4000):

    """Plots a timeline for an individual episode"""

    df_e = df[df["episode_id"] == episode_id]

    data = []

    colors = []

    for idx, row in df_e.iterrows():

        color = "tab:blue"

        if row["Speaker"] == "Sanyam Bhutani":

            color = "tab:orange"

        data_tuple = (row["timestamp_relative"], row["duration"])

        data.append(data_tuple)

        colors.append(color)

    fig, ax = plt.subplots(figsize=(20,3))

    ax.broken_barh(data, (10, 9),

                   facecolors=colors)

    # ax.set_ylim(0, 1)

#     ax.set_xlim(0, 200)

    ax.set_xlabel('seconds since start')

    ax.set_yticks([15])

    ax.set_xticks(range(0, max_tick, 200))

    ax.set_yticklabels(['Speaker'])

    blue_patch = mpatches.Patch(color='tab:blue', label=df_episodes[df_episodes["episode_id"] == episode_id]["heroes"].values[0])

    orange_patch = mpatches.Patch(color='tab:orange', label="Sanyam Bhutani")

    ax.legend(handles=[orange_patch, blue_patch])

    fig.suptitle(df_episodes[df_episodes["episode_id"] == episode_id]["episode_name"].values[0], fontsize=14)

    plt.show()
df["words_used"] = df["Text"].apply(get_words_list)

df["num_words"] = df["Text"].apply(get_num_words)

df["num_sentences"] = df["Text"].apply(get_num_sentences)

df["avg_sentence_len"] = df["Text"].apply(get_avg_sentence_len)
df.to_csv("subtitles_aggregated.csv", index=False)

df.head(2)

df_temp = df.groupby(["Speaker", "episode_id"])["num_words"].sum().reset_index().sort_values(by="num_words", ascending=False)

df_temp["derived"] = df_temp["Speaker"] + " - "+ df_temp["episode_id"]


df_temp["num_words"].hist()


px.bar(x="derived",y="num_words",data_frame=df_temp[:10],title="10 speakers with highest Number of words spoken", labels={"derived": "Speaker", "num_words": "Number of words"})
px.bar(x="derived",y="num_words",data_frame=df_temp[(df_temp["Speaker"] != "Unknown Speaker") & (df_temp["Speaker"] != "Sanyam Bhutani")][-10:],title="10 speakers with lowest Number of words spoken", labels={"derived": "Speaker", "num_words": "Number of words"})
df_temp2 = df.groupby(["Speaker", "episode_id"])["duration"].sum().reset_index().sort_values(by="duration", ascending=False)

df_temp2["duration"] /= 60

df_temp2["derived"] = df_temp2["Speaker"] + " - "+ df_temp2["episode_id"]
df_temp2["duration"].hist()
px.bar(x="derived",y="duration",data_frame=df_temp2[:10],title="10 Speakers with highest speak duration", labels={"derived": "Speaker", "duration": "Speaker Duration(minutes)"})
plot_single_episode_timeline("E23", df, df_episodes, max_tick=3600*2+1200)
df_temp3 = df_temp.merge(df_temp2, on="derived")

df_temp3["wpm"] = df_temp3["num_words"] / df_temp3["duration"]

df_temp3 = df_temp3.sort_values("wpm", ascending=False)
df_temp3["wpm"].hist()
px.bar(x="derived",y="wpm",data_frame=df_temp3[(df_temp3["Speaker_x"] != "Sanyam Bhutani") & (df_temp3["Speaker_x"] != "Unknown Speaker")][:20],title="20 speakers with highest words per minute", labels={"derived": "Speaker", "wpm": "WPM(Words per minute)"})
px.bar(x="derived",y="wpm",data_frame=df_temp3[(df_temp3["Speaker_x"] != "Sanyam Bhutani") & (df_temp3["Speaker_x"] != "Unknown Speaker")][-20:],title="20 speakers with lowest words per minute", labels={"derived": "Speaker", "wpm": "WPM(Words per minute)"})
df_temp3.rename(columns={"episode_id_x": "episode_id"}, inplace=True)

df_temp4 = df_temp3.merge(df_episodes, on="episode_id").sort_values("wpm", ascending=False)

df_temp4["derived"] = df_temp4["heroes"] + " - "  + df_temp4["episode_id"]
fig = px.bar(x="derived",y="wpm",data_frame=df_temp4[df_temp4["Speaker_x"] == "Sanyam Bhutani"][:10],title="10 highest wpm episodes of Host(Sanyam)", labels={"derived": "Episode", "wpm": "Sanyam's WPM(Words per minute)"})

fig.update_traces(marker_color='#ff7f0e')

fig.show()
fig = px.bar(x="derived",y="wpm",data_frame=df_temp4[df_temp4["Speaker_x"] == "Sanyam Bhutani"][-10:],title="10 lowest wpm episodes of Host(Sanyam)", labels={"derived": "Episode", "wpm": "Sanyam's WPM(Words per minute)"})

fig.update_traces(marker_color='#ff7f0e')

fig.show()
plot_single_episode_timeline("E49", df, df_episodes, max_tick=3400)
plot_single_episode_timeline("E63", df, df_episodes, max_tick=4200)
plot_single_episode_timeline("E36", df, df_episodes, max_tick=4200)
plot_single_episode_timeline("E35", df, df_episodes, max_tick=4200)
plot_single_episode_timeline("E1", df, df_episodes, max_tick=4200)
plot_single_episode_timeline("E74", df, df_episodes, max_tick=4200)
df_temp5 = df.groupby(["Speaker", "episode_id"])["num_words"].max().reset_index().sort_values(by="num_words", ascending=False)

df_temp5["derived"] = df_temp5["Speaker"] + " - "+ df_temp5["episode_id"]
px.bar(x="derived",y="num_words",data_frame=df_temp5[df_temp5["Speaker"] != "Sanyam Bhutani"][:10],title="10 episodes containing longest monologue by guest speaker", labels={"derived": "Episode", "num_words": "Count of consecutive words spoken"})
plot_single_episode_timeline("E57", df, df_episodes, max_tick=4200)
rindex = df[(df["episode_id"] == "E57")&(df["num_words"] == 1682)]["relative_index"].values[0]

previous_text = df[(df["episode_id"] == "E57")&(df["relative_index"] == rindex-1)]["Text"].values[0]

current_text = df[(df["episode_id"] == "E57")&(df["relative_index"] == rindex)]["Text"].values[0]
print("Sanyam's Question:", previous_text)
fig = px.bar(x="derived",y="num_words",data_frame=df_temp5[df_temp5["Speaker"] == "Sanyam Bhutani"][:10],title="10 episodes containing longest monologue by host", labels={"derived": "Episode", "num_words": "Count of consecutive words spoken by host"})

fig.update_traces(marker_color='#ff7f0e')

fig.show()
df["text_stripped"] = df["Text"].apply(lambda x: x.strip().replace(".", "").replace(","," "))

df[(df["Speaker"] == "Sanyam Bhutani") &(df["num_words"] < 3)].groupby(["text_stripped"]).count().reset_index().sort_values("Text", ascending=False)[["text_stripped", "Text"]].rename(columns={"text_stripped": "text", "Text": "Times used"})[:15]
embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
df_episode_content = df.groupby("episode_id")['Text'].apply(lambda x: ','.join(x)).reset_index()

df_episode_content["embedding"] = None

for idx, row in df_episode_content.iterrows():

    df_episode_content.loc[idx, "embedding"] = list(np.array(embedder([row["Text"]])[0])) # Get embeddings per episode

    

kmeans = KMeans(n_clusters=8, random_state=0).fit(list(df_episode_content["embedding"].values))



df_episode_content["cluster_id"] = kmeans.labels_



df_temp7 = df_episode_content.merge(df_episodes, on="episode_id")

df_temp7 = df_temp7[["episode_id", "episode_name", "cluster_id"]]
df_temp7[df_temp7["cluster_id"] == 7]
df_temp7[df_temp7["cluster_id"] == 4]
df_temp7[df_temp7["cluster_id"] == 3]
df_temp7[df_temp7["cluster_id"] == 1]
df2 = pd.read_csv("subtitles_aggregated.csv")

df2["embedding"] = None

for idx, row in df2.iterrows():

    df2.at[idx, "embedding"] = list(np.array(embedder([row["Text"]])[0]))

    

df2["is_question"] = df2["Text"].apply(lambda x: "?" in x)



# df_questions = df2[(df2["is_question"] == True) & (df2["Speaker"] == "Sanyam Bhutani")]

# kmeans = KMeans(n_clusters=25, random_state=0).fit(list(df_questions["embedding"].values))

# df_questions["cluster_id"] = kmeans.labels_

# df_questions = df_questions.merge(df_episodes, on="episode_id")

# df_temp8 = df_questions[["episode_id", "episode_name", "cluster_id", "Text", "timestamp_relative", "relative_index"]]

df_temp8 = pd.read_csv("../input/ctds-kmeans-clustered/ctds_kmeans_clustered_df.csv") # Kmeans output cache
df_temp8["answer"] = None

for idx, row in df_temp8.iterrows():

    try:

        df_temp8.at[idx, "answer"] = df2[(df2["episode_id"] == row["episode_id"]) & (df2["relative_index"] == row["relative_index"] + 1)]["Text"].values[0]    

    except:

        pass
df_temp8[df_temp8["cluster_id"] == 1][["episode_name", "Text", "answer"]].sample(frac=1)[:3].style.set_properties(subset=['Text', "episode_name"], **{'width': '200px'})
df_temp8[df_temp8["cluster_id"] == 17][["episode_name", "Text", "answer"]].sample(frac=1)[:3].style.set_properties(subset=['Text', "episode_name"], **{'width': '200px'})
df_temp8[df_temp8["cluster_id"] == 23][["episode_name", "Text", "answer"]].sample(frac=1)[:3].style.set_properties(subset=['Text', "episode_name"], **{'width': '200px'})
from transformers import pipeline

summarizer = pipeline("summarization")
df_temp9 = df_temp8[df_temp8["cluster_id"] == 17].copy()

all_answers = " ".join(df_temp9[df_temp9["cluster_id"] == 17]["answer"].values)
df_temp9["summarized_answer"] = None

ctr = 0

questions, answers, summaries = [], [], []



for idx, row in df_temp9.sample(frac=1).iterrows():

    if len(row["answer"]) < 200:

        continue

    print("*"*10)

    print(row["episode_name"])

    print("*"*10)

    print("Question:")

    print(row["Text"])

    questions.append(row["Text"])

    print("-"*10)

    print("Answer:")

    print(row["answer"])

    answers.append(row["answer"])

    print("-"*10)

    print("Summarized Answer:")

    summary = summarizer(row["answer"], max_length=100, min_length=10, do_sample=False)[0]["summary_text"]

    print(summary)

    summaries.append(summary)

    print("~"*20)

    if ctr == 0:

        break

    ctr+=1