# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.io.wavfile as sci_wav # Open wav files
import os  # Manipulate files
import matplotlib.pyplot as plt # Plotting
import librosa
%matplotlib inline

#from utils.py
fs = 16000 # 16kHz sampling rate
ROOT_DIR = '../input/cats_dogs/'
def read_wav_files(wav_files):
    '''Returns a list of audio waves
    Params:
        wav_files: List of .wav paths

    Returns:
        List of audio signals
    '''
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data_path = os.listdir(ROOT_DIR)
print(len(data_path)) # check the number of recordings
print(data_path[:20]) # Peek at the first few file paths
data = read_wav_files(data_path)
print(data[0]) # display the first recording's values
df = pd.DataFrame()
df['Path'] = data_path
is_cat = []
for path in data_path:
    if path[:3] == "cat":
        is_cat.append(1)
    else:
        is_cat.append(0)
df['Cat'] = is_cat
print (df.head())
# find length of each recording
length = []
for x in range(len(data_path)):
    length.append(len(data[x]))
df['Length'] = length
# aggregate mean from Length Series
mean_length = df.Length.mean()
mean_length_seconds = mean_length / fs
print ("{}\tMean length in Samples".format(mean_length))
print ("{}\tMean length in Seconds".format(mean_length_seconds))
# find standard deviations from Length Series
std_length = np.power((np.sum(np.power(np.subtract(length, mean_length), 2)) / (len(data_path)-1)),0.5)
std_length_seconds = np.power((np.sum(np.power(np.subtract(np.true_divide(length,fs), mean_length_seconds), 2)) / (len(data_path)-1)),0.5)
print ("{}\tStandard deviation in Samples".format(std_length))
print ("{}\tStandard deviation in Seconds".format(std_length_seconds))
# find range from minimum and maximum of Length Series
range_length = np.max(length) - np.min(length)
range_length_seconds = range_length / fs
print ("{}\t\t\tRange in Samples".format(range_length))
print ("{}\t\tRange in Seconds".format(range_length_seconds))

# Verify all of the above with .describe()
print ("\nVerifying with Pandas describe function below:")
print (df.describe())
cat_df = df[df.Cat == 1.0]
dog_df = df[df.Cat == 0.0]
num_cats = len(cat_df)
num_dogs = len(dog_df)
total_length_cat = cat_df.Length.sum()
total_length_dog = dog_df.Length.sum()
nums = (num_cats, num_dogs)
total_lengths = (total_length_cat, total_length_dog)
# Plot the variables
fig, (ax_nums_4, ax_total_lengths_4) = plt.subplots(1, 2)
ind = np.arange(len(nums))
ax_nums_4.bar(ind, nums)
ax_nums_4.set_xticks(ind)
ax_nums_4.set_xticklabels(('Cats', 'Dogs'))
ax_nums_4.set_title('Number of Cat and Dog Labels')
ax_total_lengths_4.bar(ind, total_lengths)
ax_total_lengths_4.set_title('Total Length of Cat and Dog Recordings')
ax_total_lengths_4.set_xticks(ind)
ax_total_lengths_4.set_xticklabels(('Cats','Dogs'))
plt.subplots_adjust(right = 1.75)
plt.show()
print ("{}\tPercentage of Cat labels".format(num_cats/(num_cats+num_dogs)*100))
print ("{}\tPercentage of Dog labels".format(num_dogs/(num_cats+num_dogs)*100))
print ("{}\tPercentage of Cat recordings".format(total_length_cat/(total_length_cat+total_length_dog)*100))
print ("{}\tPercentage of Dog recordings".format(total_length_dog/(total_length_cat+total_length_dog)*100))
cat_path = 'cat_20.wav'
dog_path = 'dog_barking_103.wav'
inspect_cat = np.array(librosa.load(ROOT_DIR + cat_path)[0])
inspect_dog = np.array(librosa.load(ROOT_DIR + dog_path)[0])
fig, (ax_cat_5, ax_dog_5) = plt.subplots(2, 1) # subplot for section 5
ax_cat_5.plot(inspect_cat)
ax_cat_5.set_title(cat_path)
ax_cat_5.set_xlabel("Samples")
ax_dog_5.plot(inspect_dog)
ax_dog_5.set_title(dog_path)
ax_dog_5.set_xlabel("Samples")
plt.subplots_adjust(hspace = 0.75)
plt.show()
fig_6, (ax_cat_6, ax_dog_6) = plt.subplots(1, 2) # subplot for section 6
Pxx_cat, freqs_cat, bins_cat, im_cat = ax_cat_6.specgram(inspect_cat, Fs = fs)
ax_cat_6.set_title(cat_path)
ax_cat_6.set_xlabel("Time [sec]")
ax_cat_6.set_ylabel("Frequency [Hz]")
Pxx_dog, freqs_dog, bins_dog, im_dog = ax_dog_6.specgram(inspect_dog, Fs = fs)
ax_dog_6.set_title(dog_path)
ax_dog_6.set_xlabel("Time [sec]")
ax_dog_6.set_ylabel("Frequency [Hz]")
plt.subplots_adjust(right = 2)
plt.show()
mfcc_cat = librosa.feature.mfcc(y = inspect_cat, sr = fs)
mfcc_dog = librosa.feature.mfcc(y = inspect_dog, sr = fs)
fig_7, (ax_cat_7, ax_dog_7) = plt.subplots(1, 2)
ax_cat_7.imshow(mfcc_cat, cmap = 'hot', interpolation = 'nearest')
ax_cat_7.set_title(cat_path)
ax_dog_7.imshow(mfcc_dog, cmap = 'hot', interpolation = 'nearest')
ax_dog_7.set_title(dog_path)
plt.subplots_adjust(right = 1.75)
plt.show()