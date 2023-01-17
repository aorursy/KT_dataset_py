! pip install -q pydub
import os





import random

import seaborn as sns

import cv2

# General packages

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

import IPython.display as ipd

import glob

import h5py

import plotly.graph_objs as go

import plotly.express as px

from scipy import signal

from scipy.io import wavfile

from PIL import Image

from scipy.fftpack import fft

from pydub import AudioSegment

from tempfile import mktemp



from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, LinearAxis, Range1d

from bokeh.models.tools import HoverTool

from bokeh.palettes import BuGn4

from bokeh.plotting import figure, output_notebook, show

from bokeh.transform import cumsum

from math import pi



output_notebook()





from IPython.display import Image, display

import warnings

warnings.filterwarnings("ignore")
os.listdir('../input/birdsong-recognition/')

BASE_PATH = '../input/birdsong-recognition'



# image and mask directories

train_data_dir = f'{BASE_PATH}/train_audio'

test_data_dir = f'{BASE_PATH}/example_test_audio'



print('Reading data...')

test_audio_metadata = pd.read_csv(f'{BASE_PATH}/example_test_audio_metadata.csv')

test_audio_summary = pd.read_csv(f'{BASE_PATH}/example_test_audio_summary.csv')



train = pd.read_csv(f'{BASE_PATH}/train.csv')

test = pd.read_csv(f'{BASE_PATH}/test.csv')

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')





print('Reading data completed')
display(train.head())

print("Shape of train_data :", train.shape)
display(test.head())

print("Shape of test :", test.shape)
display(test_audio_metadata.head())

print("Shape of test_audio_metadata :", test_audio_metadata.shape)
display(test_audio_summary.head())

print("Shape of test_audio_metadata :", test_audio_summary.shape)
def check_null_values(df):

    # checking missing data

    total = df.isnull().sum().sort_values(ascending = False)

    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data
check_null_values(train).head(10)
check_null_values(test)
check_null_values(test_audio_metadata).head(10)
check_null_values(test_audio_summary)
sample_audio = [

    'aldfly/XC134874.mp3',

    'amegfi/XC109299.mp3',

    'brebla/XC104521.mp3',

    'lewwoo/XC161334.mp3',

    'macwar/XC125970.mp3',

    'norwat/XC124175.mp3',

    'pinjay/XC153392.mp3',

    'rufhum/XC133552.mp3',

    'weskin/XC124287.mp3',

    'yetvir/XC120867.mp3'    

]
for audio in sample_audio:

    print("Audio sample of bird", audio.split('/')[0])

    display(ipd.Audio(f"{train_data_dir}/{audio}"))
fig = px.scatter(data_frame=train, x='longitude', y='latitude', color='ebird_code')

fig.show()
sample_audio = []

total = 0



bird_audio_folders = [ folder for folder in glob.glob(f'{train_data_dir}/*')]

birds_data = []



for folder in bird_audio_folders:

    # get all the wave files

    all_files = [y for y in os.listdir(folder) if '.mp3' in y]

    total += len(all_files)

    # collect the first file from each dir

    sample_audio.append(folder + '/'+ all_files[0])

    birds_data.append({'bird_name': folder.split('/')[-1], 'num_audio_samples': len(all_files)})
birds_sample_df = pd.DataFrame(data= birds_data)

# taking first 25 samples from birds_sample_df

birds_sample_df_top30 = birds_sample_df.sample(30)
import plotly.express as px

# df = px.data.tips()

fig = px.bar(birds_sample_df_top30, x="num_audio_samples", y="bird_name",color='bird_name', orientation='h',

             hover_data=["num_audio_samples", "bird_name"],

             height=800,

             title='Number of audio samples in tarin data')

fig.show()
train.ebird_code.value_counts()
# displaying only the top 30 countries

country = train.country.value_counts()

country_df = pd.DataFrame({'country':country.index, 'frequency':country.values}).head(30)



fig = px.bar(country_df, x="frequency", y="country",color='country', orientation='h',

             hover_data=["country", "frequency"],

             height=1000,

             title='Number of audio samples besed on country of recording')

fig.show()
##datetime feature section is inspired from this notebook, olease upvote it too

## https://www.kaggle.com/rohanrao/birdcall-eda-chirp-hoot-and-flutter



## let's create some datafremes 



df_date = train.groupby("date")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_date.date = pd.to_datetime(df_date.date, errors = "coerce")

df_date.dropna(inplace = True)

df_date["weekday"] = df_date.date.dt.day_name()





train["hour"] = pd.to_numeric(train.time.str.split(":", expand = True)[0], errors = "coerce")

df_hour = train[~train.hour.isna()].groupby("hour")["species"].count().reset_index().rename(columns = {"species": "recordings"})





df_weekday = df_date.groupby("weekday")["recordings"].sum().reset_index().sort_values("recordings", ascending = False)

# source 1

source_1 = ColumnDataSource(df_date)

tooltips_1 = [ ("Date", "@date{%F}"), ("Recordings", "@recordings")]

formatters = { "@date": "datetime" }



v1 = figure(plot_width = 800, plot_height = 450, x_axis_type = "datetime", title = "Date of recording")

v1.line("date", "recordings", source = source_1, color = "red", alpha = 0.6)



v1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))



v1.xaxis.axis_label = "Date"

v1.yaxis.axis_label = "Recordings"





# source 2

source_2 = ColumnDataSource(df_hour)



tooltips_2 = [

    ("Hour", "@hour"),

    ("Recordings", "@recordings")

]



v2 = figure(plot_width = 400, plot_height = 400, tooltips = tooltips_2, title = "Hour of recording")

v2.vbar("hour", top = "recordings", source = source_2, width = 0.75, color = "blue", alpha = 0.6)



v2.xaxis.axis_label = "Hour of day"

v2.yaxis.axis_label = "Recordings"





# source 3

source_3 = ColumnDataSource(df_weekday)



tooltips_3 = [

    ("Weekday", "@weekday"),

    ("Recordings", "@recordings")

]



v3 = figure(plot_width = 400, plot_height = 400, x_range = df_weekday.weekday.values, tooltips = tooltips_3, title = "Weekday of recording")

v3.vbar("weekday", top = "recordings", source = source_3, width = 0.75, color = "blue", alpha = 0.6)



v3.xaxis.axis_label = "Day of week"

v3.yaxis.axis_label = "Recordings"



v3.xaxis.major_label_orientation = pi / 2





show(column(v1, row(v2, v3)))
def log_specgram(audio, sample_rate, window_size=20,

                 step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))

    noverlap = int(round(step_size * sample_rate / 1e3))

    

    freqs, _, spec = signal.spectrogram(audio,

                                    fs=sample_rate,

                                    window='hann',

                                    nperseg=nperseg,

                                    noverlap=noverlap,

                                    detrend=False)

    return freqs, np.log(spec.T.astype(np.float32) + eps)
spect_samples = [

    'aldfly/XC134874.mp3',

    'ameavo/XC133080.mp3',

    'amecro/XC109768.mp3',

    'amepip/XC111040.mp3',

    'amewig/XC150063.mp3',

    'astfly/XC109920.mp3',

    'balori/XC101614.mp3',

    'bkbmag1/XC114081.mp3',

    'bkpwar/XC133993.mp3',

    'bnhcow/XC113821.mp3',

    'btnwar/XC101591.mp3',

    'carwre/XC109026.mp3',

    'chswar/XC101586.mp3',

    'evegro/XC110121.mp3',

    'greegr/XC109029.mp3',

    'hamfly/XC122665.mp3',

    'hoomer/XC134692.mp3',

    'horlar/XC113144.mp3',

    'lesgol/XC116239.mp3',

    'macwar/XC113825.mp3',

    'norfli/XC104536.mp3',

    'orcwar/XC113131.mp3',

    'pibgre/XC109907.mp3',

    'rebnut/XC104516.mp3',

    'ruckin/XC127130.mp3'    

]
fig = plt.figure(figsize=(22,22))

plt.suptitle('comparing spectograms for different birds', fontsize=20)





for i, filepath in enumerate(spect_samples):

    # Make subplots

    plt.subplot(5,5,i+1)

    bird_name, file_name = filepath.split('/')

    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")

    # create spectogram

    mp3_audio = AudioSegment.from_file(f'{train_data_dir}/{filepath}', format="mp3")  # read mp3

    wname = mktemp('.wav')  # use temporary file

    mp3_audio.export(wname, format="wav")  # convert to wav

    

    samplerate, test_sound  = wavfile.read(wname)

    _, spectrogram = log_specgram(test_sound, samplerate)

    plt.imshow(spectrogram.T, aspect='auto', origin='lower')

    plt.axis('off')
aldfly_samples = [

 'aldfly/XC157462.mp3',

 'aldfly/XC318444.mp3',

 'aldfly/XC374636.mp3',

 'aldfly/XC189268.mp3',

 'aldfly/XC296725.mp3',

 'aldfly/XC167789.mp3',

 'aldfly/XC373885.mp3',

 'aldfly/XC188432.mp3',

 'aldfly/XC189264.mp3',

 'aldfly/XC154449.mp3',

 'aldfly/XC189269.mp3',

 'aldfly/XC2628.mp3',

 'aldfly/XC420909.mp3',

 'aldfly/XC179600.mp3',

 'aldfly/XC188434.mp3',

 'aldfly/XC264715.mp3',

 'aldfly/XC189262.mp3',

 'aldfly/XC139577.mp3',

 'aldfly/XC16967.mp3',

 'aldfly/XC189263.mp3',

 'aldfly/XC318899.mp3',

 'aldfly/XC193116.mp3',

 'aldfly/XC269063.mp3',

 'aldfly/XC180091.mp3',

 'aldfly/XC381871.mp3',

]
fig = plt.figure(figsize=(22,22))

plt.suptitle('comparing spectograms for same bird', fontsize=20)



for i, filepath in enumerate(aldfly_samples):

    # Make subplots

    plt.subplot(5,5,i+1)

    bird_name, file_name = filepath.split('/')

    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")

    

    # create spectogram

    mp3_audio = AudioSegment.from_file(f"{train_data_dir}/" + filepath, format="mp3")  # read mp3

    wname = mktemp('.wav')  # use temporary file

    mp3_audio.export(wname, format="wav")  # convert to wav

    

    samplerate, test_sound  = wavfile.read(wname)

    _, spectrogram = log_specgram(test_sound, samplerate)

    

    plt.imshow(spectrogram.T, aspect='auto', origin='lower')

    plt.axis('off')
fig = plt.figure(figsize=(22,22))

plt.suptitle('comparing waveforms for different bird', fontsize=20)





for i, filepath in enumerate(spect_samples):

    # Make subplots

    plt.subplot(5,5,i+1)

    bird_name, file_name = filepath.split('/')

    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")

    # create spectogram

    mp3_audio = AudioSegment.from_file(f'{train_data_dir}/{filepath}', format="mp3")  # read mp3

    wname = mktemp('.wav')  # use temporary file

    mp3_audio.export(wname, format="wav")  # convert to wav

    

    samplerate, test_sound  = wavfile.read(wname)

    plt.plot(test_sound, '-', )

    plt.axis('off')
fig = plt.figure(figsize=(22,22))

plt.suptitle('comparing waveforms for aldfly bird', fontsize=20)



for i, filepath in enumerate(aldfly_samples):

    # Make subplots

    plt.subplot(5,5,i+1)

    bird_name, file_name = filepath.split('/')

    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")

    

    # create spectogram

    mp3_audio = AudioSegment.from_file(f"{train_data_dir}/" + filepath, format="mp3")  # read mp3

    wname = mktemp('.wav')  # use temporary file

    mp3_audio.export(wname, format="wav")  # convert to wav

    

    samplerate, test_sound  = wavfile.read(wname)

    plt.plot(test_sound, '-', )

    plt.axis('off')
duplicate_samples = []

for val in spect_samples[:5]:

    duplicate_samples.append(val)

    duplicate_samples.append(val)
fig = plt.figure(figsize=(22,22))

plt.suptitle('comparing spectograms with waveforms for same bird', fontsize=20)





for i, filepath in enumerate(duplicate_samples):

    # Make subplots    

    plt.subplot(5,2,i+1)

    bird_name, file_name = filepath.split('/')

    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")

    # create spectogram

    mp3_audio = AudioSegment.from_file(f'{train_data_dir}/{filepath}', format="mp3")  # read mp3

    wname = mktemp('.wav')  # use temporary file

    mp3_audio.export(wname, format="wav")  # convert to wav

    

    samplerate, test_sound  = wavfile.read(wname)

    _, spectrogram = log_specgram(test_sound, samplerate)



    if i % 2 == 0:

        plt.imshow(spectrogram.T, aspect='auto', origin='lower')  

    else:

        plt.plot(test_sound, '-', )

    

    plt.axis('off')
submission.to_csv('submission.csv', index=False)
submission.head()