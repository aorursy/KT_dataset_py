import IPython.display as ipd  # To play sound in the notebook

from scipy.io import wavfile # for reading wave files as numpy arrays

import wave # opening .wav files

import struct # for padding

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualizations

%matplotlib inline

import os # operating system

from os.path import join

import time

print(os.listdir("../input"))
RATE = 16000 # KHz

DATA_DIR = "../input/data"

TRAIN_CSV_FILE = "../input/train_data.csv"

TEST_CSV_FILE = "../input/test_data.csv"

FIRST_TRAIN_SAMPLE_DIR = join(DATA_DIR, "TRAIN", "DR1", "FCJF0")
# Given the test_data.csv filename or train_data.csv filename or both. 

# Output a Pandas DataFrame of the files that have `is_converted_audio == True`

def get_good_audio_files(filename):

    df = pd.read_csv(filename)

    return df[df['is_converted_audio'] == True]
# Given the test_data.csv filename or train_data.csv filename or both. 

# Output a Pandas DataFrame of the files that have `is_word_file == True`

def get_word_files(filename):

    df = pd.read_csv(filename)

    return df[df['is_word_file'] == True]
# Given a file path to the .WAV.wav file

# Output the result of SciPy’s wavefile.read()

def read_audio(wave_path, verbose=False):

    rate, data = wavfile.read(wave_path)

    # make sure the rate of the file is the RATE that we want

    assert rate == RATE

    print("Sampling (frame) rate = ", rate) if verbose else None

    print("Total samples (frames) = ", data.shape) if verbose else None

    return data
# Given a row (...,'TRAIN','DR4','MMDM0', 'SI681.WAV.wav',...)

# return the os.path.join of the relevant dirs

# '../input/data/TRAIN/DR4/MMDM0/SI681.WAV.wav'

def join_dirs(row):

    return os.path.join(DATA_DIR,

                       row['test_or_train'],

                       row['dialect_region'],

                       row['speaker_id'],

                       row['filename'])
# Given a file path to the .WRD file

# Output a list of tuples containing (start, end, word, speaker_id, sentence_id)

def parse_wrd_timestamps(wrd_path, verbose=False):

    print('wrd_path', wrd_path) if verbose else None

    speaker_id = wrd_path.split('/')[-2]

    sentence_id = wrd_path.split('/')[-1].replace('.WRD', '')

    wrd_file = open(wrd_path)

    content = wrd_file.read()

    content = content.split('\n')

    # print('content b4 tuple', content) if verbose else None

    content = [tuple(foo.split(' ') + [speaker_id, sentence_id]) for foo in content if foo != '']

    wrd_file.close()

    return content
# Given both a time_aligned_words file && the output of read_audio() 

# Output the another list of tuples containing (audio_data, label)

# e.g.

# [(array([ 2, 2, -3, ... , 3, 6, 1], dtype=int16), critical), 

#   ... ((array([ 5, -6, 4, ... , 1, 3, 3], dtype=int16),maintenance)]

def parse_word_waves(time_aligned_words, audio_data, verbose=False):

    return [align_data(data, words, verbose) for data, words in zip(audio_data, time_aligned_words)]

    

# given numpy wave array and time alignment details

# output a list of each data with its word

def align_data(data, words, verbose=False):

    aligned = []

    print('len(data)', len(data)) if verbose else None

    print('len(words)', len(words)) if verbose else None

    print('data', data) if verbose else None

    print('words', words) if verbose else None

    for tup in words:

        print('tup',tup) if verbose else None

        start = int(tup[0])

        end = int(tup[1])

        word = tup[2]

        speaker_id = tup[3]

        sentence_id = tup[4]

        assert start >= 0

        assert end <= len(data)

        aligned.append( (data[start:end], word, speaker_id, sentence_id) )

    assert len(aligned) == len(words)

    return aligned
df = get_good_audio_files(TRAIN_CSV_FILE)

df.head()
df['filepath'] = df.apply(lambda row: join_dirs(row), axis=1)
df['filepath'][0]
waves = df['filepath']
audio_data = [read_audio(wave) for wave in waves]
audio_data[0]
assert len(waves) == len(audio_data)
wrd_path = waves[0].replace('.WAV.wav', '') + '.WRD'

wrd_path
print(parse_wrd_timestamps('../input/data/TRAIN/DR4/MMDM0/SI681.WRD', verbose=True))
foo = parse_wrd_timestamps('../input/data/TRAIN/DR4/MMDM0/SI681.WRD')

print(align_data(audio_data[0], foo))
time_aligned_words = [parse_wrd_timestamps(w.replace('.WAV.wav', '') + '.WRD') for w in waves]
word_aligned_audio = parse_word_waves(time_aligned_words, audio_data)

word_aligned_audio[0]
timestamp = time.strftime("%m%d%Y%H%M%S", time.localtime())

timestamp
"catch_m_monotone_20-a-classroom_l_ewenike_chidi_05312019150206_ewenike_pitch_down_50".split('_')
!mkdir waves

os.chdir(path='waves')

os.getcwd()
i = 1

for sentence in word_aligned_audio:

    for word_tup in sentence:

        timestamp = time.strftime("%m%d%Y%H%M%S", time.localtime())

        data, word, speaker, sentence = word_tup

        gender = 'gender-speaker-id'

        location = 'unknown-location'

        loudness = 'unknown-loudness'

        lastname = 'lastname-speaker-id'

        firstname = 'firstname-speaker-id'

        nametag = 'timit'

        description = speaker + '-' + sentence + '-' + str(i)

        filename = word + '_' + gender + '_' +  description + '_' + location

        filename += '_' + loudness + '_' + lastname + '_' + firstname

        filename += '_' + timestamp + '_' + nametag

        filename += '.wav'

        

        # filenames cannot have single quotes

        filename = filename.replace("'", '')

        

        wavfile.write(data=data,filename=filename,rate=RATE)

        # print(data, filename)

        i += 1



print("done")

stuff = os.listdir('.')

stuff.remove('.ipynb_checkpoints') if '.ipynb_checkpoints' in stuff else None

stuff.remove('__notebook_source__.ipynb') if '__notebook_source__.ipynb' in stuff else None

print('Saved',len(stuff),'wave files')
# archive & compress

!tar -zcvf ../waves.tar.gz .
os.chdir(path='..')

os.getcwd()
word_audio_files = np.array(stuff)
waf = pd.DataFrame(word_audio_files)

waf.columns = ['filenames']

waf.dataframeName = 'word_audio.csv'
waf.head()
def parse_word(row):

    feature_list = row.split("_")

    return feature_list[0]



def parse_gender(row):

    feature_list = row.split("_")

    return feature_list[1]



def parse_description(row):

    feature_list = row.split("_")

    return feature_list[2]



def parse_location(row):

    feature_list = row.split("_")

    return feature_list[3]



def parse_loudness(row):

    feature_list = row.split("_")

    return feature_list[4]



def parse_full_name(row):

    feature_list = row.split("_")

    return feature_list[5] + "-" + feature_list[6] 



def parse_timestamp(row):

    feature_list = row.split("_")

    return feature_list[7]



def parse_nametag(row):

    feature_list = row.split("_")

    return feature_list[8]
waf['word'] = waf['filenames'].apply(lambda row: parse_word(row))

waf['gender'] = waf['filenames'].apply(lambda row: parse_gender(row))

waf['description'] = waf['filenames'].apply(lambda row: parse_description(row))

waf['location'] = waf['filenames'].apply(lambda row: parse_location(row))

waf['loudness'] = waf['filenames'].apply(lambda row: parse_loudness(row))

waf['full_name'] = waf['filenames'].apply(lambda row: parse_full_name(row))

waf['timestamp'] = waf['filenames'].apply(lambda row: parse_timestamp(row))

waf['nametag'] = waf['filenames'].apply(lambda row: parse_nametag(row))
waf.head()
# return a tuple of the description column details

def parse_extra_description(row):

    speaker_id, sentence_id, iteration = row.split('-')

    return (speaker_id, sentence_id, iteration)
waf['speaker_id'] = waf['description'].apply(lambda row: parse_extra_description(row)[0])

waf['sentence_id'] = waf['description'].apply(lambda row: parse_extra_description(row)[1])

# iteration is just the `i` number in the for loop when the file was written to disk

waf['iteration'] = waf['description'].apply(lambda row: parse_extra_description(row)[2])
waf.head()
waf.to_csv("word_audio.csv")
NUM_FRAMES = 22528



def normalize_bytes(fname):

    with wave.open(fname, 'rb') as f:

        data = f.readframes(NUM_FRAMES)

        



def normalize(file1):

    input = wave.open(file1, 'r')

    norm_value = min(22528, input.getnframes())

    data = input.readframes(norm_value)

    params = list(input.getparams())

    input.close()

    filename = file1[:-4] + "_norm.wav"

    output = wave.open(filename, 'w')

    output.setparams(params)

    output.writeframes(data)

    while(output.getnframes() < 22528):

        padding = struct.pack('<h', 0)

        output.writeframesraw(padding)

    output.close()

    return filename
def white_noise(file1):

    input = wave.open(file1, 'r')

    norm_value = min(22528, input.getnframes())

    data = np.fromstring(input.readframes(input.getnframes()), dtype=np.int16)

    params = list(input.getparams())

    input.close()

    amplitude = np.random.randint(250, 1000)

    wn_data = np.random.randint(0, amplitude, 22528, dtype=np.int16)

    white_noise_data = wn_data + data

    white_noise_data = white_noise_data.tostring()



    filename = file1[:-4] + "_wn.wav"

    output = wave.open(filename, 'w')

    output.setparams(params)

    output.writeframes(white_noise_data)

    output.close()
# FFT on audio

# Pitch shift

# INFFT on audio



def pitch_change(name, write_loc, amount):

    dir = -1 if amount < 0 else 1

    filepath = write_loc + name

    wr = wave.open(filepath, 'r')

    par = list(wr.getparams())

    par[3] = 0  # The number of samples will be set by writeframes.

    par = tuple(par)

    dir_str = 'down_' if dir == -1 else 'up_'

    fname = write_loc + name[:-4] + '_pitch_' + dir_str + str(abs(amount)) + '.wav'

    ww = wave.open(fname, 'w')

    ww.setparams(par)

    fr = 10

    sz = wr.getframerate()//fr  # Read and process 1/fr second at a time.

    c = int(wr.getnframes()/sz)  # count of the whole file

    shift = amount//fr  # shifting 100 Hz

    for num in range(c):

        da = np.fromstring(wr.readframes(sz), dtype=np.int16)

        # split channels

        left, right = da[0::2], da[1::2]

        lf, rf = np.fft.rfft(left), np.fft.rfft(right) # run fft

        # shift frequency values by desired amount

        lf, rf = np.roll(lf, shift), np.roll(rf, shift)

        # clear incorrectly shifted values

        if(dir == -1):

            lf[shift:], rf[shift:] = 0, 0

        else:

            lf[0:shift], rf[0:shift] = 0, 0



        # run inverse fft

        nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)

        # integrate channels

        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)

        ww.writeframes(ns.tostring())

    ww.close()

    wr.close()

    return fname



def run_pitch_change(name, write_loc):

    fname_list = []

    for i in range(10):

        val = -200 + (50 * i)

        fname = pitch_change(name, write_loc, val)

        fname_list.append(fname)

    return fname_list
!mkdir normalized

!mkdir augmented

!ls -d */
!ls | head
aug_files = []

norm_files = []



# first augment

for f in waf['filenames']:

    res1 = run_pitch_change(f, 'waves/')

    res1_renamed = []



    for r in res1:

        new_aug_name = r.replace('waves/','augmented/')

        aug_files.append(new_aug_name)

        res1_renamed.append(new_aug_name)

        os.rename(r, new_aug_name)



    for x in aug_files:

        res = normalize(x)

        old_name = res

        new_name = res.replace('augmented/', 'normalized/')

        norm_files.append(new_name)

        os.rename(old_name, new_name)



    for r in res1_renamed:

        os.remove(r)



    aug_files = []
# aug_files = []



# # first augment

# for f in waf['filenames']:

#     res = run_pitch_change(f, 'waves/')

#     for r in res:

#         new_name = r.replace('waves/','augmented/')

#         aug_files.append(new_name)

#         os.rename(r, new_name)
!ls augmented | head
aug_files[:5]
# norm_files = []



# # now normalize

# for f in aug_files:

#     res = normalize(f)

#     old_name = res

#     new_name = res.replace('augmented/', 'normalized/')

#     norm_files.append(new_name)

#     os.rename(old_name, new_name)
!ls normalized | head
!ls -l
# archive & compress

!tar -zcvf waves_aug.tar.gz augmented
# archive & compress

!tar -zcvf waves_norm.tar.gz normalized
!ls -l
# lastly, let's clean up the working directory so that Kaggle does not yell at us

!rm -rf waves

!rm -rf augmented

!rm -rf normalized
nRow, nCol = waf.shape

print(f'There are {nRow} wave files and {nCol} columns')
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()



# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()





# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
plotPerColumnDistribution(waf, 12, 5)
valueCounts = waf['speaker_id'].value_counts()

valueCounts.plot.bar()
valueCounts = waf['word'].value_counts()

valueCounts.plot.bar()