'''
# File location at hdf5_getters.py
import hdf5_getters

subset_path = 'MillionSongSubset/AdditionalFiles/subset_msd_summary_file.h5' # location of the summary file
summary = hdf5_getters.open_h5_file_read(dataset_path) # hdf5 object to be passed into getters
num_songs = hdf5_getters.get_num_songs(summary) # iterate over this many songs

# Create pandas DataFrame from the collected data
df_cols = ('artist_name','title','duration','song_hotttnesss','year','tempo','key','loudness','song_mode','time_signature')
df = pd.DataFrame(columns = df_cols, index = range(num_songs))

for k in range(num_songs):
	df.artist_name[k] = hdf5_getters.get_artist_name(summary,k)
	df.title[k] = hdf5_getters.get_title(summary,k)
	df.danceability[k] = hdf5_getters.get_danceability(summary,k)
	df.duration[k] = hdf5_getters.get_duration(summary,k)
	df.song_hotttnesss[k] = hdf5_getters.get_song_hotttnesss(summary,k)
	df.year[k] = hdf5_getters.get_year(summary,k)
	df.tempo[k] = hdf5_getters.get_tempo(summary,k)
	df.energy[k] = hdf5_getters.get_energy(summary,k)
	df.key[k] = hdf5_getters.get_key(summary,k)
	df.loudness[k] = hdf5_getters.get_loudness(summary,k)
	df.song_mode[k] = hdf5_getters.get_mode(summary,k)
	df.time_signature[k] = hdf5_getters.get_time_signature(summary,k)
	if k%1000 == 0:
		print ("Processed {} / {}".format(k,num_songs))

# save as csv for easy access later
subset_filename = 'subset_msd_summary_file.csv'
''';
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting tools
import seaborn as sns # more plotting tools
import scipy.stats as sp
data_path = '../input/subset_msd_summary_file.csv'
df = pd.read_csv(data_path)
df.drop(df.columns[0], axis = 1, inplace = True) # drop the index column
print ("Size of the dataset: {}\n".format(df.shape))
df.columns
print ("'" + df.title[0] + "' by " + df.artist_name[0])
print (str(df.duration[0]) + " seconds")
print (str(df.year[0]) + " year released")
print (str(df.loudness[0]) + " decibels")
print (str(df.tempo[0]) + " beats per minute")
print (str(df.key[0]) + " key")
print (str(df.song_mode[0]) + " mode")
print (str(df.time_signature[0]) + " beats per segment")
print (str(df.danceability[0]) + " danceability")
print (str(df.song_hotttnesss[0]) + " song hotttnesss")
print (str(df.energy[0]) + " energy")
df_zeros = (df==0).sum()
danceability_missing = df_zeros.danceability / 10000.0 * 100
energy_missing = df_zeros.energy / 10000.0 * 100
hotttnesss_missing = (df_zeros.song_hotttnesss + df.song_hotttnesss.isnull().sum()) / 10000.0 * 100
year_missing = df_zeros.year / 10000.0 * 100
print("Percent missing\t\tFeature\n{}\t\t\tdanceability\n{}\t\t\tenergy\n{}\t\t\thotttnesss\n{}\t\t\tyear".format(danceability_missing,energy_missing,hotttnesss_missing, year_missing))
df.drop(['danceability','energy','song_hotttnesss'], axis = 1, inplace = True) # drop danceability, energy, and song_hotttnesss
# number of missing values: tempo, time_signature, and title
missing_null = df.isnull().sum()
missing_zeros = (df==0).sum()
these_values = [missing_null.tempo + missing_zeros.tempo, missing_null.time_signature + missing_zeros.time_signature, missing_null.title + missing_zeros.title]
fig_missing_values, ax_missing_values = plt.subplots()
ax_missing_values.bar(range(3), these_values);
ax_missing_values.set_xticks(range(3))
ax_missing_values.set_xticklabels(['tempo','time_signature','title']);
ax_missing_values.set_title("Other Features with Missing Values");
ax_missing_values.set_ylabel('Number of Missing Values');
# impute tempo and time_signature
df_imputed = df.copy()
def impute_with_median(df,series):
    df_imputed[series] = df[series].replace(to_replace = 0,value = np.nan)
    median = df_imputed[series].median()
    df_imputed[series].fillna(value = median, inplace = True)
impute_with_median(df_imputed,'tempo')
impute_with_median(df_imputed,'time_signature')
# get a clean dataframe of year
# df_imputed now has nans for year
year_nans = df_imputed.year.replace(to_replace = 0, value = np.nan)
clean_year = year_nans.dropna()
clean_year.reset_index(drop = True, inplace = True)
df_imputed.year = year_nans
df_imputed.describe()
correlation = df_imputed.corr()
sns.heatmap(correlation);
fig_duration_dist, ax_duration_dist = plt.subplots()
ax_duration_dist.boxplot(df_imputed.duration);
ax_duration_dist.set_title("Distribution of Duration");
ax_duration_dist.set_ylabel('Seconds');
year_counts = clean_year.value_counts()
year_counts.sort_index(inplace = True)
fig_year_dist, ax_year_dist = plt.subplots()
ax_year_dist.plot(year_counts.index, year_counts);
ax_year_dist.set_title('Distribution of Year');
ax_year_dist.set_ylabel('Count');
ax_year_dist.set_xlabel('Year');
# tempo histogram with normal curve overlay
ax_tempo = sns.distplot(df_imputed.tempo); # shows density with kernal density estimate (kde)
plt.title('Distribution of Tempo');
plt.ylabel('Density Value');
plt.xlabel('Beats per Minute');
# key sorted horizontal bar chart (color coded)
key_counts = df_imputed.key.value_counts()
df_key_counts = pd.DataFrame(key_counts)
df_key_counts.reset_index(inplace = True)
# ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
fig_key_counts, ax_key_counts = plt.subplots()
bars = ax_key_counts.barh(df_key_counts.index, df_key_counts.key)
ax_key_counts.set_yticks(df_key_counts.index)
ax_key_counts.set_yticklabels(['G','C','D','A','C#','E','F','B','A#','F#','G#','D#']);
ax_key_counts.set_title('Key Count');
ax_key_counts.set_xlabel('Count');
for x,bar in zip(range(len(bars)),bars):
    bar.set_color(str(2.0/(x+2.3)))
# loudness histogram (talk about skew)
sns.distplot(df_imputed.loudness);
plt.title('Distribution of Loudness');
plt.ylabel('Density Value');
plt.xlabel('dB');
# song_mode bar chart (colors)
song_mode_counts = df_imputed.song_mode.value_counts()
fig_song_mode_counts, ax_song_mode_counts = plt.subplots()
song_mode_bars = ax_song_mode_counts.bar(song_mode_counts.index, song_mode_counts);
ax_song_mode_counts.set_xticks(song_mode_counts.index)
ax_song_mode_counts.set_xticklabels(['Major', 'Minor']);
ax_song_mode_counts.set_title('Song Mode Counts');
ax_song_mode_counts.set_ylabel('Count');
song_mode_bars[0].set_color('r')
percent_major = song_mode_counts[1] / 10000.0 * 100
print ("{}% of songs are Major".format(percent_major))
# time_signature (line/area plot)
time_signature_counts = df_imputed.time_signature.value_counts()
time_signature_counts.sort_index(inplace = True)
fig_time_signature_counts, ax_time_signature_counts = plt.subplots()
ax_time_signature_counts.bar(time_signature_counts.index, time_signature_counts);
ax_time_signature_counts.set_title('Time Signature Counts');
ax_time_signature_counts.set_xlabel('Beats per Segment');
ax_time_signature_counts.set_ylabel('Count');
df_c = df_imputed[df_imputed.key == 0]
df_d = df_imputed[df_imputed.key == 2]
df_g = df_imputed[df_imputed.key == 7]
percent_major_df = song_mode_counts[1] / 10000 * 100.0
song_mode_counts_g = df_g.song_mode.value_counts()
percent_major_g = song_mode_counts_g[1] / len(df_g) * 100.0
song_mode_counts_c = df_c.song_mode.value_counts()
percent_major_c = song_mode_counts_c[1] / len(df_c) * 100.0
song_mode_counts_d = df_d.song_mode.value_counts()
percent_major_d = song_mode_counts_d[1] / len(df_d) * 100.0
print ("Percent Major\t\tKey")
print ("{}\t\t\tdataset\n{}\tG\n{}\tC\n{}\tD".format(percent_major_df,percent_major_g,percent_major_c,percent_major_d))

g_mode_counts = df_g.song_mode.value_counts()
c_mode_counts = df_c.song_mode.value_counts()
major = [g_mode_counts[1], c_mode_counts[1]]
minor = [g_mode_counts[0], c_mode_counts[0]]
g_c_mode_index = range(len(major))
# plot the stacked bar charts
fig_g_c_mode, ax_g_c_mode = plt.subplots()
ax_g_c_mode.bar(g_c_mode_index, major);
ax_g_c_mode.bar(g_c_mode_index, minor, bottom = major);
ax_g_c_mode.set_title("Mode Distribution for G and C")
ax_g_c_mode.set_xticks(g_c_mode_index)
ax_g_c_mode.set_xticklabels(["G","C"])
# print the t-test results
print (sp.ttest_ind(g_mode_counts, c_mode_counts, equal_var=False))
# find linear regresion of tempo vs duration
slope, intercept, r_value, p_value, std_err = sp.linregress(df_imputed.tempo,df_imputed.duration)
# print the r_value
print ("r-value: {}".format(r_value))
# make list of regression line values
regress_length = int(df_imputed.tempo.max())
regress = range(regress_length) * (np.ones(regress_length) * float(slope)) + np.ones(regress_length) * float(intercept)
# plot scatter over linear regression line
fig_duration_tempo, ax_duration_tempo = plt.subplots()
ax_duration_tempo.scatter(df_imputed.tempo, df_imputed.duration, s = .5);
ax_duration_tempo.plot(regress, 'r')
ax_duration_tempo.set_title('Duration vs Tempo');
ax_duration_tempo.set_xlabel('Tempo');
ax_duration_tempo.set_ylabel('Duration');
artist_name_counts = df_imputed.artist_name.value_counts()
df_artist_name_counts = artist_name_counts.value_counts()
plt.bar(df_artist_name_counts.index, df_artist_name_counts);
plt.title('Number of Songs by an Artist');
plt.ylabel('Number of Artists');
plt.xlabel('Number of Songs by the same Artist');
print (artist_name_counts[:][:4])
df_mario = df_imputed[df_imputed.artist_name == 'Mario Rosenstock']
df_mario.describe()
