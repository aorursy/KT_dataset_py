import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import gc



#thanks to EEK: https://www.kaggle.com/theupgrade/creating-breaks-as-start-ends-of-bins

ttf_resets = np.array([5656575,50085879,104677357,138772454,187641821,

                  218652631,245829586,307838918,338276288,375377849,

                  419368881,461811624,495800226,528777116,585568145,

                  621985674]) - 1



train = pd.read_csv('../input/train.csv', nrows=ttf_resets[0])

#rename columns - quicker to type!

train.columns = ['signal', 'ttf']

print(train.describe())

train.plot(kind='scatter',x='ttf',y='signal', s=1)

plt.xlim(1.5, -0.05)

plt.title('Acoustic signal/ttf')

plt.show()
train.signal.hist(bins=200)
train.loc[(train.signal < 40) & (train.signal > -40), ].signal.hist(bins=200)
train.ttf.hist()
import decimal as dec

t1 = dec.Decimal(train.iloc[0, 1])

t2 = dec.Decimal(train.iloc[1, 1])

t3 = dec.Decimal(train.iloc[2, 1])

print('First TTF: ', t1, '(s)')

print('Second TTF: ', t2, '(s)')

print('Third TTF: ', t3, '(s)')



print('First delta t: ', t1-t2, '(s)')

print('Second delta t: ', t2-t3, '(s)')
from astropy.stats import LombScargle



t = train.ttf.values

y = train.signal.values



#astropy implementation works directly on frequency, not angular frequency

#autopower() calculates a frequency grid automatically based on mean time separation

frequency, power = LombScargle(t[0:10000], y[0:10000]).autopower(nyquist_factor=2)
plt.plot(frequency, power) 

plt.title('Noise LSP Frequency-Amplitude Spectrum')

plt.xlabel('Frequency (Hz)')

plt.ylabel('Amplitude')

plt.show()
#power multiplied for graphical visibility

freq_df = pd.DataFrame({'freq': frequency.round(),

                       'amp': (power*1e6).round(1)})
import matplotlib.pyplot as plt

freq_df.loc[freq_df.freq < 1000000, :].plot(kind='scatter',x='freq',y='amp', s=2)

plt.title('Noise LSP Frequency-Amplitude Spectrum')

plt.show()
freq_df.loc[freq_df.freq < 150000, :].plot(kind='scatter',x='freq',y='amp', s=2)

plt.title('Noise LSP Frequency-Amplitude Spectrum')

plt.show()
freq_df.sort_values('amp', ascending=False).head(10)
freq_df.loc[(freq_df.freq > 125000) & (freq_df.freq < 1000000), :].plot(kind='scatter',x='freq',y='amp', s=2)

plt.show()
def arr_dist(arr, sep, n=5):

    output = []

    for x in arr:

        keep=True

        for y in output:

            if abs(y-x)<sep:

                keep=False

                break

        if(keep):

            output.append(x)

            if len(output)==n:

                return(np.asarray(output))



ROWS_PER_SEGMENT = 10000

TIME_PER_SEGMENT = train.iloc[0, 1] - train.iloc[ROWS_PER_SEGMENT, 1]

MIN_FREQ = round(1/TIME_PER_SEGMENT)

MAX_FREQ = 1e8

FREQ_SEP = 0.2e6



def LSP_freq(df, signal_col='signal', time_col='ttf',

             nrows=ROWS_PER_SEGMENT, min_freq=MIN_FREQ, max_freq=MAX_FREQ):

    print('Lomb-Scargle Periodogram analysis commencing.')

    print('Minimum detection frequency: {}Hz'.format(MIN_FREQ))

    print('Manual maximum frequency cutoff: {}Hz'.format(MAX_FREQ))

    print('Number of segments: ', round(len(df)/ROWS_PER_SEGMENT))

    #initialise empty arrays for frequency outputs to be concatenated to DataFrame 

    freq_1 = np.zeros(len(df))

    freq_2 = np.zeros(len(df))

    freq_3 = np.zeros(len(df))

    freq_4 = np.zeros(len(df))

    freq_5 = np.zeros(len(df))

    segment_num = np.zeros(len(df))

    #loop through input DataFrame in chunks of length=nrows

    init_id = 0

    segment_id =1

    while init_id < len(df):

        if segment_id==1:

            print('Processing segment {:d}...'.format(segment_id))

        if segment_id%25==0:

            print('Processing segment {:d}...'.format(segment_id))

        end_id = min(init_id + nrows, len(df))

        ids = range(init_id, end_id)

        df_chunk = df.iloc[ids]

        #np arrays of amplitude and time columns

        signal = df_chunk[signal_col].values

        ttf = df_chunk[time_col].values

        #clear memory

        del df_chunk

        gc.collect()

        #calulate Lomb-Scargle periodograms for spectral analysis

        freq, amp = LombScargle(ttf, signal).autopower(nyquist_factor=2)

        freq_df = pd.DataFrame({'freq': freq.round(),

                               'amp': amp})

        #obtain frequencies sorted by highest amplitude as np.array

        top_freqs = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).freq.values

        del freq_df, freq, amp

        gc.collect()

        #obtain top 5 values that do not lie within 1kHz of eachother

        top_freqs = arr_dist(top_freqs, sep=FREQ_SEP)

        #sort principal frequencies from highest to lowest

        top_freqs = -np.sort(-top_freqs)

        #update main frequency component arrays

        freq_1[ids] = top_freqs[0]

        freq_2[ids] = top_freqs[1]

        freq_3[ids] = top_freqs[2]

        freq_4[ids] = top_freqs[3]

        freq_5[ids] = top_freqs[4]

        segment_num[ids] = segment_id

        del top_freqs

        init_id += nrows

        segment_id += 1

    print('...Done. Adding main component frequencies as DataFrame columns...')

    df['Freq_1'] = freq_1

    df['Freq_2'] = freq_2

    df['Freq_3'] = freq_3

    df['Freq_4'] = freq_4

    df['Freq_5'] = freq_5

    df['Segment'] = segment_num

    df['Freq_MinMax'] = df['Freq_1'] - df['Freq_5']

    print('...Done.')

    

LSP_freq(train)
train.head()
cols = ['Freq_1', 'Freq_2','Freq_3','Freq_4','Freq_5']

for x in cols:

    train[x + '_RM'] = train[x].rolling(window=100000,center=False).mean()

cols_rm = []

for x in range(0, 5):

    cols_rm.append(cols[x] + '_RM')



ax = plt.gca()

train.plot(kind='line',x='ttf',y=cols_rm ,ax=ax, figsize=(12, 6))

plt.xlim(1.5, -0.05)

plt.title('5 Main Noise Component Frequencies (LSP, min separation 200kHz)')

plt.xlabel('TTF(s)')

plt.ylabel('Frequency (Hz)')

plt.show()
train['FREQ_MEAN'] = train[cols].mean(axis=1)

train['FREQ_MEAN_RM'] = train['FREQ_MEAN'].rolling(window=250000, center=False).mean()

train.plot(kind='line',x='ttf',y='FREQ_MEAN_RM', figsize=(12, 6))

plt.xlim(1.5, -0.05)

plt.show()
train['FREQ_RATIO'] = train['Freq_1']/train['Freq_2']

train['FREQ_RATIO_RM'] = train['FREQ_RATIO'].rolling(window=500000, center=False).mean()

train.plot(kind='line',x='ttf',y='FREQ_RATIO_RM', figsize=(12, 6))

plt.xlim(1.5, -0.05)

plt.title('Ratio of Highest Main Frequency/Second Highest: Rolling-Mean')

plt.show()
#false-alarm filter - unused in this kernel

def LSP_filter(p, M):

    threshold = -np.log(1-((1-p)**(1/M)))

    return(threshold)
ROWS_PER_SEGMENT = 1000

TIME_PER_SEGMENT = train.iloc[0, 1] - train.iloc[ROWS_PER_SEGMENT, 1]

MIN_FREQ = round(1/TIME_PER_SEGMENT)

MAX_FREQ = 1e10

THRESHOLD = 0.1



def LSP_freq_filtered(df, signal_col='signal', time_col='ttf',

             nrows=ROWS_PER_SEGMENT, min_freq=MIN_FREQ, max_freq=MAX_FREQ):

    print('Lomb-Scargle Periodogram analysis commencing.')

    print('Minimum detection frequency: {}Hz'.format(MIN_FREQ))

    print('Manual maximum frequency cutoff: {}Hz'.format(MAX_FREQ))

    #initialise empty arrays for frequency outputs to be concatenated to DataFrame 

    freq_1 = np.zeros(len(df))

    amps_1 = np.zeros(len(df))

    segment_num = np.zeros(len(df))

    #loop through input DataFrame in chunks of length=nrows

    init_id = 0

    segment_id =1

    while init_id < len(df):

        if segment_id==1:

            print('Processing segment {:d}...'.format(segment_id))

        if segment_id%500==0:

            print('Processing segment {:d}...'.format(segment_id))

        end_id = min(init_id + nrows, len(df))

        ids = range(init_id, end_id)

        df_chunk = df.iloc[ids]

        #np arrays of amplitude and time columns

        signal = df_chunk[signal_col].values

        ttf = df_chunk[time_col].values

        #clear memory

        del df_chunk

        gc.collect()

        #calulate Lomb-Scargle periodograms for spectral analysis

        freq, amp = LombScargle(ttf, signal).autopower(nyquist_factor=2)

        freq_df = pd.DataFrame({'freq': freq.round(),

                               'amp': amp})

        freq_df = freq_df.loc[freq_df['amp'] >= THRESHOLD] 

        #obtain frequencies sorted by highest amplitude as np.array

        top_freqs = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).freq.values

        amps = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).amp.values

        del freq_df, freq, amp, signal, ttf

        gc.collect()

        #update main frequency component arrays

        try:

            freq_1[ids] = top_freqs[0]

            amps_1[ids] = amps[0]

        except:

            freq_1[ids] = 0

        segment_num[ids] = segment_id

        del top_freqs

        init_id += nrows

        segment_id += 1

    print('...Done. Adding main component frequencies as DataFrame columns...')

    df['Freq_periodic'] = freq_1

    df['Amps_periodic'] = amps_1

    df['Segment'] = segment_num

    print('...Done.')

    

LSP_freq_filtered(train)
train.head()
train['Freq_RM'] = train['Freq_periodic'].rolling(window=200000,center=False).std()

train.plot(kind='scatter',x='ttf',y='Freq_RM', figsize=(12, 6), s=1)

plt.xlim(1.5, -0.05)

plt.axvline(0.32)

plt.text(0.3,2e8,'Main Quake')

plt.title('Lomb-Scargle Main Component Frequency Standard Deviation (100MHz): Rolling-Mean ')

plt.show()
train['Amp_RM'] = train['Amps_periodic'].rolling(window=200000,center=False).mean()

train.plot(kind='scatter',x='ttf',y='Amp_RM', figsize=(12, 6), s=1)

plt.xlim(1.5, -0.05)

plt.title('Lomb-Scargle Main Component Frequency Amplitude: Rolling-Mean')

plt.axvline(0.32)

plt.text(0.3,0.01,'Main Quake')

plt.show()