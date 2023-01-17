#eda analysis on bird sounds dataset

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
import librosa #module to analyse audio signals

from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.layouts import column, row

from bokeh.models.tools import HoverTool
from bokeh.palettes import BuGn4, PuRd, Reds, Set3
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import cumsum

output_notebook()


train = train = pd.read_csv('../input/birdsong-recognition/train.csv')
test = pd.read_csv('../input/birdsong-recognition/test.csv')
audio_path = "../input/birdsong-recognition/train_audio"

train_extend = pd.read_csv("../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv")
train.columns
train.country

train["type"].values
df_bird_map = train[['ebird_code', 'species']].drop_duplicates()

for ebird_code in os.listdir(audio_path)[:5]:
    species = df_bird_map[df_bird_map['ebird_code'] == ebird_code].species.values[0]
    audio_file = os.listdir(f"{audio_path}/{ebird_code}")[0]
    path_to_audio = f"{audio_path}/{ebird_code}/{audio_file}"
    ipd.display(ipd.HTML(f"<h2>{ebird_code} ({species})</h2>"))
    ipd.display(ipd.Audio(path_to_audio))
#target variable is ebird code. After training, the audios should be classisfied 
#to the correct ebird label
#looking at the distribution of ebird

bird_type = train.groupby("ebird_code")["filename"].count().reset_index().rename(columns = {"filename": "recordings"}).sort_values("recordings")

source = ColumnDataSource(bird_type)
tooltips = [("Bird code", "@ebird_code"),("Recordings", "@recordings")]

fig = figure(plot_width = 650, plot_height = 3000, y_range = bird_type['ebird_code'].values, tooltips = tooltips, title = "Count of Bird Species")

fig.hbar("ebird_code", right="recordings", source = source, height = 0.75, color = "blue", alpha = 0.6)

fig.xaxis.axis_label = "Count"
fig.yaxis.axis_label = "Ebird code"

show(fig)
bird_type.columns
"""
About half of the bird species have 100 plus recordings
"""
#looking at the times when the recordings were taken

df_date = train.groupby("date")["ebird_code"].count().reset_index().rename(columns = {'ebird_code': "recordings"})
df_date.date = pd.to_datetime(df_date.date, errors = 'coerce')
#drop missing values 
df_date.dropna(inplace = True)

df_date["weekday"] = df_date["date"].dt.day_name()

source_1 = ColumnDataSource(df_date)

tooltips_1 = [("Date", "@date"), ("Recordings", "@recordings")]

formatters = {"@date": "datetime"}

fig1 = figure(plot_width = 700, plot_height = 400, x_axis_type = "datetime", title = "Date of recording")
fig1.line("date", "recordings", source = source_1, width = 2, color = "orange", alpha =0.6)

fig1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))
fig1.xaxis.axis_label = "Date"
fig1.yaxis.axis_label = "Recordings"

show(fig1)

'''
Most of the recordings have been recorded since 2010
There is an uusual spike of recordings around the year 2003

'''





train["hour"] = pd.to_numeric(train["time"].str.split(":", expand = True)[0], errors = "coerce")

df_hour = train[train["hour"].notna()].groupby("hour")["ebird_code"].count().reset_index().rename(columns ={"ebird_code" : "recordings"})

source_2 = ColumnDataSource(df_hour)
tooltips_2 = [("Hour", "@hour"),("Recordings", "@recordings")]

fig2 = figure( plot_width = 500, plot_height = 400, tooltips = tooltips_2, title = "Hour of recordings")
fig2.vbar("hour", top = "recordings", source = source_2, width = 0.75, color = "tomato", alpha = 0.6)


fig2.xaxis.axis_label = "Hour of day"
fig2.yaxis.axis_label = "Recordings"

show(fig2)
'''
Most recordings happen from around 6am to 7pm
low number of recordings are during the night
high number of recordings happen in between 6am and 12am
'''
df_weekday = df_date.groupby("weekday")["recordings"].sum().reset_index().sort_values("recordings", ascending=False)

source_3 = ColumnDataSource(df_weekday)

tooltips_3 =[("Weekday", "@weekday"),("Recordings", "@recordings")]

fig3 = figure(plot_width = 500, plot_height = 400, x_range = df_weekday["weekday"].values, tooltips = tooltips_3, title = "Day of the week and no of recordings")
fig3.vbar("weekday", top="recordings", source = source_3, width = 0.75, color="limegreen", alpha=0.6 )

fig3.xaxis.axis_label = "Day of the week"
fig3.yaxis.axis_label = "Recordings"

show(fig3)
'''
Most recordings happen during the weekends
'''
#countries with the highest number of recordings

df_country = train.groupby("country")["ebird_code"].count().reset_index().rename(columns = {"ebird_code": "recordings"}).sort_values("recordings", ascending=False).head(20).sort_values("recordings")

source_4 = ColumnDataSource(df_country)

tooltips_4 = [("country", '@country'), ("recordings", "@recordings")]

fig4 = figure(plot_width = 650, plot_height = 700, y_range = df_country["country"].values, tooltips = tooltips_1, title = "Country of recording")
fig4.hbar("country", right = "recordings", source = source_4, height = 0.75, color = "coral", alpha = 0.6)


fig4.xaxis.axis_label = "Country"
fig4.yaxis.axis_label = "Recordings"
show(fig4)
#distribution of bird recordings by their location
df_location = train.groupby("location")['ebird_code'].count().reset_index().rename(columns = {"ebird_code":"recordings"}).sort_values("recordings", ascending = False).head(20).sort_values("recordings")

source_5 = ColumnDataSource(df_location)

tooltips_5 = [("Location", "@location"), ("Recoedings", "@recordings")]

fig5 = figure(plot_width = 650, plot_height = 700, y_range = df_location['location'].values, tooltips = tooltips_5, title = "Top 20 locations for recordings")
fig5.hbar("location", right = "recordings", source = source_5, height = 0.75, color = "red", alpha = 0.6)
show(fig5)
#distribution of bird calls in training set
df_calltype = train.groupby("type")["ebird_code"].count().reset_index().rename(columns = {"ebird_code": "records"}).sort_values("records", ascending = False).head(15)

source_6 = ColumnDataSource(df_calltype)

tooltips_6 = [("type", "@type"), ("records", "@records")]

fig6 = figure(plot_width = 650, plot_height = 700, x_range = df_calltype['type'].values, tooltips = tooltips_6, title = "Top song types" )
fig6.vbar("type", top="records", source = source_6, width = 0.75, color="purple", alpha=0.6 )

fig6.xaxis.axis_label = "Bird song type"
fig6.yaxis.axis_label = "Recordings"
show(fig6)
train_extend.head()
df_train_original = train.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings_original"})

df_train_extend = train_extend.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings_extended"})

df_bird = df_train_original.merge(df_train_extend, on = "species", how = "left").fillna(0)
df_bird["recordings_total"] = df_bird["recordings_original"] + df_bird["recordings_extended"]

df_bird = df_bird.sort_values("recordings_total").reset_index()

source_6 = ColumnDataSource(df_bird)
tooltips_6 = [("Species", "@species"), ("original recordings", "recordings_original"), ("extended recordings", "@recordings_extended")]

fig6 = figure(plot_width = 750, plot_height = 3000, y_range = df_bird.species.values, tooltips = tooltips_6, title = "Count of Bird Species")
fig6.hbar_stack(["recordings_original", "recordings_extended"], y = "species", source = source_6, height = 0.75, color = ["blue", "orange"], alpha = 0.65)
              
fig6.xaxis.axis_label = "Count"
fig6.yaxis.axis_label = "Species"

show(fig6)
              
sample_audio_path = "../input/birdsong-recognition/train_audio/aldfly/XC135455.mp3"
import librosa.display
#analysing the audio
x , sr = librosa.load(sample_audio_path)
print(x.shape, sr)
#visualising audio
plt.figure(figsize = (14,5))
librosa.display.waveplot(x)
