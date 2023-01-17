## importing packages

import bokeh

import os



import numpy as np

import pandas as pd

import seaborn as sns

import IPython.display as ipd



from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, LinearAxis, Range1d

from bokeh.models.tools import HoverTool

from bokeh.palettes import BuGn4

from bokeh.plotting import figure, output_notebook, show

from bokeh.transform import cumsum



output_notebook()





## configuring setup, constants and parameters

PATH_TRAIN = "../input/birdsong-recognition/train.csv"

PATH_TEST = "../input/birdsong-recognition/test.csv"



PATH_TRAIN_EXTENDED = "../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv"



PATH_AUDIO = "../input/birdsong-recognition/train_audio"

## reading data

df_train = pd.read_csv(PATH_TRAIN)

df_test = pd.read_csv(PATH_TEST)



df_train_extended = pd.read_csv(PATH_TRAIN_EXTENDED)

df_train.shape
df_train.head()
df_train.columns
df_test.shape
df_test.columns
df_bird_map = df_train[["ebird_code", "species"]].drop_duplicates()



for ebird_code in os.listdir(PATH_AUDIO)[:20]:

    species = df_bird_map[df_bird_map.ebird_code == ebird_code].species.values[0]

    audio_file = os.listdir(f"{PATH_AUDIO}/{ebird_code}")[0]

    audio_path = f"{PATH_AUDIO}/{ebird_code}/{audio_file}"

    ipd.display(ipd.HTML(f"<h2>{ebird_code} ({species})</h2>"))

    ipd.display(ipd.Audio(audio_path))

df_bird = df_train.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings"}).sort_values("recordings")



source = ColumnDataSource(df_bird)

tooltips = [

    ("Bird Species", "@species"),

    ("Recordings", "@recordings")

]



v = figure(plot_width = 650, plot_height = 3000, y_range = df_bird.species.values, tooltips = tooltips, title = "Count of Bird Species")

v.hbar("species", right = "recordings", source = source, height = 0.75, color = "steelblue", alpha = 0.6)



v.xaxis.axis_label = "Count"

v.yaxis.axis_label = "Species"



show(v)

df_train_extended.head()
df_bird_original = df_train.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings_original"})

df_bird_extended = df_train_extended.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings_extended"})



df_bird = df_bird_original.merge(df_bird_extended, on = "species", how = "left").fillna(0)

df_bird["recordings_total"] = df_bird.recordings_original + df_bird.recordings_extended

df_bird = df_bird.sort_values("recordings_total").reset_index()



source = ColumnDataSource(df_bird)

tooltips = [

    ("Bird Species", "@species"),

    ("Recordings Original", "@recordings_original"),

    ("Recordings Extended", "@recordings_extended"),

]



v = figure(plot_width = 650, plot_height = 3000, y_range = df_bird.species.values, tooltips = tooltips, title = "Count of Bird Species")

v.hbar_stack(["recordings_original", "recordings_extended"], y = "species", source = source, height = 0.75, color = ["steelblue", "crimson"], alpha = 0.6)



v.xaxis.axis_label = "Count"

v.yaxis.axis_label = "Species"



show(v)

df_date = df_train.groupby("date")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_date.date = pd.to_datetime(df_date.date, errors = "coerce")

df_date.dropna(inplace = True)

df_date["weekday"] = df_date.date.dt.day_name()



source_1 = ColumnDataSource(df_date)



tooltips_1 = [

    ("Date", "@date{%F}"),

    ("Recordings", "@recordings")

]



formatters = {

    "@date": "datetime"

}



v1 = figure(plot_width = 700, plot_height = 400, x_axis_type = "datetime", title = "Date of recording")

v1.line("date", "recordings", source = source_1, width = 2, color = "orange", alpha = 0.6)



v1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))



v1.xaxis.axis_label = "Date"

v1.yaxis.axis_label = "Recordings"





df_train["hour"] = pd.to_numeric(df_train.time.str.split(":", expand = True)[0], errors = "coerce")



df_hour = df_train[~df_train.hour.isna()].groupby("hour")["species"].count().reset_index().rename(columns = {"species": "recordings"})



source_2 = ColumnDataSource(df_hour)



tooltips_2 = [

    ("Hour", "@hour"),

    ("Recordings", "@recordings")

]



v2 = figure(plot_width = 450, plot_height = 400, tooltips = tooltips_2, title = "Hour of recording")

v2.vbar("hour", top = "recordings", source = source_2, width = 0.75, color = "maroon", alpha = 0.6)



v2.xaxis.axis_label = "Hour of day"

v2.yaxis.axis_label = "Recordings"





df_weekday = df_date.groupby("weekday")["recordings"].sum().reset_index().sort_values("recordings", ascending = False)



source_3 = ColumnDataSource(df_weekday)



tooltips_3 = [

    ("Weekday", "@weekday"),

    ("Recordings", "@recordings")

]



v3 = figure(plot_width = 250, plot_height = 400, x_range = df_weekday.weekday.values, tooltips = tooltips_3, title = "Weekday of recording")

v3.vbar("weekday", top = "recordings", source = source_3, width = 0.75, color = "maroon", alpha = 0.6)



v3.xaxis.axis_label = "Day of week"

v3.yaxis.axis_label = "Recordings"



v3.xaxis.major_label_orientation = np.pi / 2





show(column(v1, row(v2, v3)))

df_duration = df_train.groupby("duration")["species"].count().reset_index().rename(columns = {"species": "recordings"})



source = ColumnDataSource(df_duration)



tooltips = [

    ("Duration", "@duration"),

    ("Recordings", "@recordings")

]



v = figure(plot_width = 700, plot_height = 200, tooltips = tooltips, title = "Duration of recording")

v.line("duration", "recordings", source = source, width = 2, color = "green", alpha = 0.6)



v.xaxis.axis_label = "Duration"

v.yaxis.axis_label = "Recordings"



show(v)

df_country = df_train.groupby("country")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings")



source = ColumnDataSource(df_country)



tooltips_1 = [

    ("Country", "@country"),

    ("Recordings", "@recordings")

]



v1 = figure(plot_width = 650, plot_height = 1000, y_range = df_country.country.values, tooltips = tooltips_1, title = "Country of recording")

v1.hbar("country", right = "recordings", source = source, height = 0.75, color = "coral", alpha = 0.6)



show(v1)

df_location = df_train.groupby("location")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False).head(20).sort_values("recordings")



source = ColumnDataSource(df_location)



tooltips_2 = [

    ("Location", "@location"),

    ("Recordings", "@recordings")

]



v2 = figure(plot_width = 650, plot_height = 400, y_range = df_location.location, tooltips = tooltips_2, title = "Top-20 Locations of recording")

v2.hbar("location", right = "recordings", source = source, height = 0.75, color = "coral", alpha = 0.6)



show(v2)

df_train["elevation_clean"] = pd.to_numeric(df_train.elevation.str.replace("[^0-9]", ""), errors = "coerce")

df_elevation = df_train[~df_train.elevation_clean.isna()].groupby("elevation_clean")["species"].count().reset_index().rename(columns = {"elevation_clean": "elevation", "species": "recordings"})



source_3 = ColumnDataSource(df_elevation[df_elevation.elevation < 4700])



tooltips_3 = [

    ("Elevation", "@elevation"),

    ("Recordings", "@recordings")

]



v3 = figure(plot_width = 650, plot_height = 300, tooltips = tooltips_3, title = "Elevation of Recording")

v3.line("elevation", "recordings", source = source_3, width = 3, color = "lightseagreen", alpha = 0.6)



v3.xaxis.axis_label = "Elevation (in metres)"

v3.yaxis.axis_label = "Recordings"



show(v3)

df_bird_country = df_train.groupby(["species", "country"])["ebird_code"].count().reset_index()

df_bird_country = df_bird_country.merge(df_bird, on = "species")

df_bird_country = df_bird_country.merge(df_country, on = "country")

df_bird_country.rename(columns = {"ebird_code": "recordings", "recordings_original": "recordings_species", "recordings": "recordings_country"}, inplace = True)

df_bird_country["alpha"] = 0.2 + (0.8 * (df_bird_country.recordings - min(df_bird_country.recordings)) / (max(df_bird_country.recordings) - min(df_bird_country.recordings)))



df_bird_country = df_bird_country.sort_values(["recordings"], ascending = False).reset_index(drop = True)



source_4 = ColumnDataSource(df_bird_country)



tooltips_4 = [

    ("Species", "@species"),

    ("Country", "@country"),

    ("Recordings", "@recordings")

]



species = list(df_bird_country.species.unique())

species.reverse()



v4 = figure(

    plot_width = 700,

    plot_height = 2100,

    x_range = list(df_bird_country.country.unique()),

    y_range = species,

    x_axis_location = "above",

    tooltips = tooltips_4,

    title = "Species by Country"

)



v4.rect("country", "species", 0.9, 0.9, source = source_4, color = "purple", alpha = "alpha", line_color = None, hover_line_color = "black")



v4.grid.grid_line_color = None

v4.axis.axis_line_color = None

v4.axis.major_tick_line_color = None

v4.xaxis.major_label_text_font_size = "6px"

v4.yaxis.major_label_text_font_size = "7px"

v4.axis.major_label_standoff = 0

v4.xaxis.major_label_orientation = np.pi / 2



show(v4)

df_rating = df_train.groupby("rating")["species"].count().reset_index().rename(columns = {"species": "recordings"})



source_1 = ColumnDataSource(df_rating)



tooltips_1 = [

    ("Rating", "@rating{0.0}"),

    ("Recordings", "@recordings")

]



v1 = figure(plot_width = 350, plot_height = 300, tooltips = tooltips_1, title = "Distribution of Rating")

v1.vbar("rating", top = "recordings", source = source_1, width = 0.4, color = "lightseagreen", alpha = 0.6)



v1.xaxis.axis_label = "Rating"

v1.yaxis.axis_label = "Recordings"





df_playback = df_train.groupby("playback_used")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_playback["percentage"] = df_playback.recordings * 100 / df_playback.recordings.sum()

df_playback["angle"] = df_playback.recordings / df_playback.recordings.sum() * 2 * np.pi

df_playback["color"] = ["mediumseagreen", "lightseagreen"]



source_2 = ColumnDataSource(df_playback)



tooltips_2 = [

    ("Playback Used", "@playback_used"),

    ("Recordings", "@recordings"),

    ("Percentage", "@percentage{0}%")

]



v2 = figure(plot_width = 350, plot_height = 300, tooltips = tooltips_2, title = "Distribution of Playback Used")

v2.wedge(x = 0, y = 1, radius = 0.4, start_angle = cumsum("angle", include_zero = True), end_angle = cumsum("angle"), line_color = "white", fill_color = "color", legend_field = "playback_used", source = source_2)



v2.axis.axis_label = None

v2.axis.visible = False

v2.grid.grid_line_color = None





df_pitch = df_train.groupby("pitch")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False)



source_3 = ColumnDataSource(df_pitch)



tooltips_3 = [

    ("Pitch", "@pitch"),

    ("Recordings", "@recordings")

]



v3 = figure(plot_width = 350, plot_height = 300, x_range = df_pitch.pitch.values, tooltips = tooltips_3, title = "Distribution of Pitch")

v3.vbar("pitch", top = "recordings", source = source_3, width = 0.4, color = "lightseagreen", alpha = 0.6)



v3.xaxis.axis_label = "Pitch"

v3.yaxis.axis_label = "Recordings"





df_channels = df_train.groupby("channels")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_channels["percentage"] = df_channels.recordings * 100 / df_channels.recordings.sum()

df_channels["angle"] = df_channels.recordings / df_channels.recordings.sum() * 2 * np.pi

df_channels["color"] = ["mediumseagreen", "lightseagreen"]



source_4 = ColumnDataSource(df_channels)



tooltips_4 = [

    ("Channel", "@channels"),

    ("Recordings", "@recordings"),

    ("Percentage", "@percentage{0}%")

]



v4 = figure(plot_width = 350, plot_height = 300, tooltips = tooltips_4, title = "Distribution of Channel")

v4.wedge(x = 0, y = 1, radius = 0.4, start_angle = cumsum("angle", include_zero = True), end_angle = cumsum("angle"), line_color = "white", fill_color = "color", legend_field = "channels", source = source_4)



v4.axis.axis_label = None

v4.axis.visible = False

v4.grid.grid_line_color = None





df_speed = df_train.groupby("speed")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False)



source_5 = ColumnDataSource(df_speed)



tooltips_5 = [

    ("Speed", "@speed"),

    ("Recordings", "@recordings")

]



v5 = figure(plot_width = 350, plot_height = 300, x_range = df_speed.speed.values, tooltips = tooltips_5, title = "Distribution of Speed")

v5.vbar("speed", top = "recordings", source = source_5, width = 0.4, color = "lightseagreen", alpha = 0.6)



v5.xaxis.axis_label = "Speed"

v5.yaxis.axis_label = "Recordings"





df_bird_seen = df_train.groupby("bird_seen")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_bird_seen["percentage"] = df_bird_seen.recordings * 100 / df_bird_seen.recordings.sum()

df_bird_seen["angle"] = df_bird_seen.recordings / df_bird_seen.recordings.sum() * 2 * np.pi

df_bird_seen["color"] = ["mediumseagreen", "lightseagreen"]



source_6 = ColumnDataSource(df_bird_seen)



tooltips_6 = [

    ("Bird Seen", "@bird_seen"),

    ("Recordings", "@recordings"),

    ("Percentage", "@percentage{0}%")

]



v6 = figure(plot_width = 350, plot_height = 300, tooltips = tooltips_6, title = "Distribution of Bird Seen")

v6.wedge(x = 0, y = 1, radius = 0.4, start_angle = cumsum("angle", include_zero = True), end_angle = cumsum("angle"), line_color = "white", fill_color = "color", legend_field = "bird_seen", source = source_6)



v6.axis.axis_label = None

v6.axis.visible = False

v6.grid.grid_line_color = None





df_volume = df_train.groupby("volume")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False)



source_7 = ColumnDataSource(df_volume)



tooltips_7 = [

    ("Volume", "@volume"),

    ("Recordings", "@recordings")

]



v7 = figure(plot_width = 350, plot_height = 300, x_range = df_volume.volume.values, tooltips = tooltips_7, title = "Distribution of Volume")

v7.vbar("volume", top = "recordings", source = source_7, width = 0.4, color = "lightseagreen", alpha = 0.6)



v7.xaxis.axis_label = "Volume"

v7.yaxis.axis_label = "Recordings"





df_train["filetype"] = "mp3"

df_train.loc[df_train.file_type != "mp3", "filetype"] = "other"



df_file_type = df_train.groupby("filetype")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_file_type["percentage"] = df_file_type.recordings * 100 / df_file_type.recordings.sum()

df_file_type["angle"] = df_file_type.recordings / df_file_type.recordings.sum() * 2 * np.pi

df_file_type["color"] = ["mediumseagreen", "lightseagreen"]



source_8 = ColumnDataSource(df_file_type)



tooltips_8 = [

    ("File Type", "@filetype"),

    ("Recordings", "@recordings"),

    ("Percentage", "@percentage{0.000}%")

]



v8 = figure(plot_width = 350, plot_height = 300, tooltips = tooltips_8, title = "Distribution of File Type")

v8.wedge(x = 0, y = 1, radius = 0.4, start_angle = cumsum("angle", include_zero = True), end_angle = cumsum("angle"), line_color = "white", fill_color = "color", legend_field = "filetype", source = source_8)



v8.axis.axis_label = None

v8.axis.visible = False

v8.grid.grid_line_color = None





df_sampling_rate = df_train.groupby("sampling_rate")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False)



source_9 = ColumnDataSource(df_sampling_rate)



tooltips_9 = [

    ("Sampling Rate", "@sampling_rate"),

    ("Recordings", "@recordings")

]



v9 = figure(plot_width = 350, plot_height = 300, x_range = df_sampling_rate.sampling_rate.values, tooltips = tooltips_9, title = "Distribution of Sampling Rate")

v9.vbar("sampling_rate", top = "recordings", source = source_9, width = 0.4, color = "lightseagreen", alpha = 0.6)



v9.xaxis.axis_label = "Sampling Rate"

v9.yaxis.axis_label = "Recordings"



v9.xaxis.major_label_orientation = np.pi / 4





license_map = {

    "Creative Commons Attribution-NonCommercial-ShareAlike 3.0": "CC BY-NC-SA 3.0",

    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0": "CC BY-NC-SA 4.0",

    "Creative Commons Attribution-ShareAlike 3.0": "CC BY-SA 3.0",

    "Creative Commons Attribution-ShareAlike 4.0": "CC BY-SA 4.0"

}



df_train["license_abbr"] = df_train.license.map(license_map)



df_license = df_train.groupby("license_abbr")["species"].count().reset_index().rename(columns = {"license_abbr": "license", "species": "recordings"}).sort_values("recordings", ascending = False)

df_license["percentage"] = df_license.recordings * 100 / df_license.recordings.sum()

df_license["angle"] = df_license.recordings / df_license.recordings.sum() * 2 * np.pi

df_license["color"] = BuGn4



source_10 = ColumnDataSource(df_license)



tooltips_10 = [

    ("License", "@license"),

    ("Recordings", "@recordings"),

    ("Percentage", "@percentage{0}%")

]



v10 = figure(plot_width = 350, plot_height = 300, tooltips = tooltips_10, title = "Distribution of License")

v10.wedge(x = 0, y = 1, radius = 0.4, start_angle = cumsum("angle", include_zero = True), end_angle = cumsum("angle"), line_color = "white", fill_color = "color", legend_field = "license", source = source_10)



v10.axis.axis_label = None

v10.axis.visible = False

v10.grid.grid_line_color = None



v10.legend.label_text_font_size = "6pt"





show(column(row(v1, v2), row(v3, v4), row(v5, v6), row(v7, v8), row(v9, v10)))

df_recordist = df_train.groupby("recordist")["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False).head(20)



source_1 = ColumnDataSource(df_recordist)



tooltips_1 = [

    ("Recordist", "@recordist"),

    ("Recordings", "@recordings")

]



v1 = figure(plot_width = 650, plot_height = 400, x_range = df_recordist.recordist.values, tooltips = tooltips_1, title = "Top-20 Recordists")

v1.vbar("recordist", top = "recordings", source = source_1, width = 0.75, color = "olive", alpha = 0.8)



v1.xaxis.axis_label = "Recordist"

v1.yaxis.axis_label = "Recordings"



v1.xaxis.major_label_orientation = np.pi / 4



df_recordist_country = df_train.groupby(["recordist", "country"])["species"].count().reset_index().rename(columns = {"species": "recordings"}).sort_values("recordings", ascending = False).drop_duplicates("country").head(20)

df_recordist_country["recordist_country"] = df_recordist_country.recordist + " (" + df_recordist_country.country + ")"



source_2 = ColumnDataSource(df_recordist_country)



tooltips_2 = [

    ("Recordist", "@recordist"),

    ("Country", "@country"),

    ("Recordings", "@recordings")

]



v2 = figure(plot_width = 650, plot_height = 400, x_range = df_recordist_country.recordist_country.values, tooltips = tooltips_2, title = "Top-20 Recordists across Countries")

v2.vbar("recordist_country", top = "recordings", source = source_2, width = 0.75, color = "olive", alpha = 0.8)



v2.xaxis.axis_label = "Recordist"

v2.yaxis.axis_label = "Recordings"



v2.xaxis.major_label_orientation = np.pi / 4





show(column(v1, v2))
