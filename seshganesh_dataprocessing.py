import numpy as np

import pandas as pd

import numpy as np

import librosa.display

from scipy import signal

import scipy.io.wavfile

import IPython.display as ipd

import matplotlib.pyplot as plt

import os

from IPython.display import Audio, IFrame, display
train_metadata = pd.read_csv('../input/birdsong-recognition/train.csv')
train_metadata.head(1)
train_metadata['duration'].plot.box()

plt.show()

print(train_metadata['duration'].describe())

list_first=['buggna', 'brespa', 'rebwoo', 'comyel', 'bulori', 'y00475', 'hergul', 'bnhcow', 'whcspa', 'stejay', 'lesgol', 'houfin', 'pinwar', 'yebfly', 'easmea', 'cangoo', 'grtgra', 'westan', 'savspa', 'hamfly', 'veery', 'gnwtea', 'tuftit', 'aldfly', 'ovenbi1', 'warvir', 'blujay', 'winwre3', 'caster1', 'yerwar', 'orcwar', 'bkhgro', 'mouchi', 'amered', 'herthr', 'btnwar', 'osprey', 'wlswar', 'perfal', 'rocpig', 'whtspa', 'norpar', 'astfly', 'canwar', 'pilwoo', 'hoowar', 'amerob', 'swathr', 'normoc', 'whbnut', 'killde', 'greegr', 'louwat', 'haiwoo', 'marwre', 'wesmea', 'pibgre', 'houspa', 'boboli', 'eawpew', 'bushti', 'wilfly', 'balori', 'gockin', 'bewwre', 'grcfly', 'fiespa', 'norfli', 'wewpew', 'magwar', 'evegro', 'chswar', 'carwre', 'robgro', 'redcro']

list_second=['amegfi', 'bkbwar', 'chispa', 'woothr', 'moudov', 'vesspa', 'comrav', 'mallar3', 'pinsis', 'comter', 'bkcchi', 'purfin', 'norcar', 'barswa', 'indbun', 'greyel', 'reevir1', 'eastow', 'rewbla', 'annhum', 'horlar', 'swaspa', 'olsfly', 'houwre', 'ruckin', 'pasfly', 'buhvir', 'amepip', 'spotow', 'macwar', 'eucdov', 'canwre', 'dowwoo', 'sora', 'scoori', 'foxspa', 'bkpwar', 'wilsni1', 'blugrb1', 'daejun', 'norwat', 'gadwal', 'banswa', 'sonspa', 'lesyel', 'amecro', 'brdowl', 'logshr', 'comgra', 'comred', 'grhowl', 'brncre', 'gnttow', 'brnthr', 'snobun', 'linspa', 'rebnut', 'blkpho', 'cacwre', 'brthum', 'dusfly', 'yetvir', 'lazbun', 'buwwar', 'btywar', 'easblu', 'leafly', 'larspa', 'solsan', 'bktspa', 'scatan', 'rethaw', 'pingro', 'sposan', 'plsvir']

list_third=['easpho', 'leabit', 'casvir', 'greroa', 'calqua', 'coohaw', 'wooscj2', 'cedwax', 'yelwar', 'comgol', 'fiscro', 'comnig', 'labwoo', 'treswa', 'weskin', 'amewoo', 'commer', 'easkin', 'grycat', 'gryfly', 'clanut', 'rocwre', 'snogoo', 'belspa2', 'casfin', 'camwar', 'amekes', 'btbwar', 'lobdow', 'renpha', 'bawwar', 'leasan', 'amtspa', 'brwhaw', 'reshaw', 'grbher3', 'semsan', 'gocspa', 'prawar', 'juntit1', 'belkin1', 'cowscj1', 'bkchum', 'horgre', 'ribgul', 'lobcur', 'pecsan', 'comloo', 'grnher', 'cliswa', 'norsho', 'tunswa', 'semplo', 'yebsap', 'pygnut', 'lotduc', 'phaino', 'bkbmag1', 'rufhum', 'rusbla', 'merlin', 'sheowl', 'yehbla', 'baisan', 'saypho', 'buwtea', 'vigswa', 'pinjay', 'goleag', 'brebla', 'bkbcuc', 'palwar', 'wiltur', 'nutwoo', 'amebit', 'lesnig', 'bongul', 'wooduc', 'eursta', 'calgul', 'norpin', 'wesgre', 'truswa', 'ameavo', 'eargre', 'whtswi', 'sagthr', 'baleag', 'wesblu', 'doccor', 'amewig', 'whfibi', 'rebsap', 'rthhum', 'gcrfin', 'moublu', 'sagspa1', 'wessan', 'nrwswa', 'chukar', 'norhar2', 'chiswi', 'rebmer', 'lewwoo', 'swahaw', 'rinduc', 'rufgro', 'rudduc', 'shshaw', 'lecthr', 'hoomer', 'coshum', 'buffle', 'redhea']
new_data=train_metadata[train_metadata["ebird_code"].isin(list_first)].reset_index()

new_data.drop(['index'],axis=1,inplace=True)

subset=new_data[(new_data['duration']>10) & (new_data['duration']>4.0)]
subset.shape
bird_names=list(subset.loc[:,'ebird_code'].unique())

#bird_names
len(train_metadata.species.unique()),len(subset.species.unique())
bird_indx=2 # bird index

sample_no=10

#print(len(subset.loc[subset['ebird_code']==bird_names[bird_indx]]))



audio_filename=subset.loc[subset['ebird_code']==bird_names[bird_indx]][sample_no:sample_no+1].filename.values[0]

fold_name=str(subset.loc[subset['filename']==audio_filename].ebird_code.values[0])

inp_file='../input/birdsong-recognition/train_audio/'+str(fold_name)+'/'+str(audio_filename)

print(inp_file,audio_filename)



# Load the audio file in librosa

librosa_audio,librosa_sample_rate=librosa.load(inp_file)

librosa.display.waveplot(librosa_audio, sr=librosa_sample_rate)

display(Audio(librosa_audio, rate=librosa_sample_rate))



# Finding Amplitude array from the audio

db = librosa.core.amplitude_to_db(librosa_audio)



# Finding the mean & std of the amplitude

mean_db = np.abs(db).mean()

std_db = db.std()

print(mean_db,std_db)



# Splitting the audio into non-silent intervls

audio_split_intervals= librosa.effects.split(y=librosa_audio, top_db = mean_db - std_db)



# removes silences from clip

silence_removed = []

for inter in audio_split_intervals:

    silence_removed.extend(librosa_audio[inter[0]:inter[1]])

silence_removed = np.array(silence_removed)

librosa.display.waveplot(silence_removed, sr=librosa_sample_rate)

display(Audio(silence_removed, rate=librosa_sample_rate))





'''

# using those silences from clip

silence_clip = librosa_audio[0:audio_split_intervals[0][0]]

for i in range(len(audio_split_intervals)-1):

    silence_clip=np.append(silence_clip,librosa_audio[audio_split_intervals[i][1]:audio_split_intervals[i+1][0]])



display(Audio(silence, rate=librosa_sample_rate))

'''





#print(subset.loc[subset['filename']==audio_filename].duration.values[0])

librosa.display.waveplot(silence_removed, sr=librosa_sample_rate)
librosa.display.waveplot(librosa_audio, sr=librosa_sample_rate)
fft=np.fft.fft(silence_removed)

magnitude=np.abs(fft)

frequency=np.linspace(0,librosa_sample_rate,len(magnitude))

left_frequency=frequency[:int(len(frequency)/2)]

left_magnitude=magnitude[:int(len(magnitude)/2)]

plt.plot(left_frequency,left_magnitude)

plt.xlabel('frequency')

plt.ylabel("Magnitude")
orig_duration=librosa.get_duration(silence_removed)

print(orig_duration)
n_fft=2048

hop_length=512

stft=librosa.core.stft(silence_removed,hop_length=hop_length,n_fft=n_fft)

spectogram=np.abs(stft)



log_spectogram=librosa.amplitude_to_db(spectogram) # Converting amplitude to decibels for clear visuals



librosa.display.specshow(log_spectogram,sr=librosa_sample_rate,hop_length=hop_length)

plt.xlabel("Time")

plt.ylabel("Frequency")

plt.colorbar()

plt.show()
MFCCs=librosa.feature.mfcc(silence_removed,hop_length=hop_length,n_fft=n_fft,n_mfcc=13)



librosa.display.specshow(MFCCs,sr=librosa_sample_rate,hop_length=hop_length)

plt.xlabel("Time")

plt.ylabel("MFCCs")

plt.colorbar()

plt.show()

MFCCs.shape
import math

from scipy.io.wavfile import write

from numba import jit, prange

from tqdm import tqdm





#print(subset.loc[subset['filename']==audio_filename].duration.values[0])













# Parameters for MFCC

hop_length=512

n_fft=2048

n_mfcc=13

#expected_no_mfccvec_per_segment=700



# Storing the data

data = {"mapping":[], "mfcc":[],"labels":[]}



SAMPLE_RATE=librosa_sample_rate

print(SAMPLE_RATE)



#DURATION=subset.loc[subset['filename']==audio_filename].duration.values[0]# seconds

DURATION=librosa.get_duration(silence_removed)

SAMPLES_PER_SIGNAL=SAMPLE_RATE*DURATION # Varies as a function of duration if sample rate is fixed

segment_duration=5 # seconds

total_samples_reqd_segment=SAMPLE_RATE*segment_duration

num_samples_per_segment=total_samples_reqd_segment

expected_no_mfccvec_per_segment=math.ceil(num_samples_per_segment/hop_length)

#num_samples_per_segment=math.ceil(expected_no_mfccvec_per_segment*hop_length)

num_segments=math.ceil(int(SAMPLES_PER_SIGNAL)/num_samples_per_segment)

print(num_segments,SAMPLES_PER_SIGNAL,num_samples_per_segment,DURATION)



# Sampling for multiple MFCCs





#@autojit

for s in tqdm(prange(num_segments)):

    start_sample=int(num_samples_per_segment*s) # For s=0: start

    end_sample=start_sample+num_samples_per_segment # for s=0; start+no_of_samples/segment

    if end_sample>SAMPLES_PER_SIGNAL:

        break

    print(start_sample,end_sample,int((end_sample-start_sample)/SAMPLE_RATE))



    plt.figure(figsize=(12,3))

    plt.plot(silence_removed,'r')

    plt.plot(silence_removed[start_sample:end_sample],'g')

    plt.show()

    

    mfcc=librosa.feature.mfcc(silence_removed[start_sample:end_sample],hop_length=hop_length,n_fft=n_fft,n_mfcc=n_mfcc) # Analyzing a slice of a signal 

    mfcc=mfcc.T 

    print(mfcc.shape)

    librosa.display.specshow(mfcc,sr=SAMPLE_RATE,hop_length=hop_length)

    plt.xlabel('mfcc')

    plt.ylabel('time')

    plt.colorbar()

    plt.show()

    

    write("example"+str(s)+".wav", librosa_sample_rate, librosa_audio[start_sample:end_sample])

    #ipd.Audio("example.wav")

    

    #print("segment_"+str(s+1),mfcc.shape)

    # Some times audio samples dont have expected length: so different shape of MFCC , so need to use expected_no_mfcvec_per_segment

    # Store mfcc for segment if it has the expected length

    if len(mfcc)==expected_no_mfccvec_per_segment+1:

        print("pass")

        data["mfcc"].append(mfcc.tolist())





'''

# Dividing audio into different segments

num_segments=2  # Lesser the no of segments, better capturing of the entire signal without miss

SAMPLE_RATE=librosa_sample_rate

DURATION=duration# seconds

SAMPLES_PER_SIGNAL=SAMPLE_RATE*DURATION # Varies as a function of duration if sample rate is fixed





num_samples_per_segment=int(SAMPLES_PER_SIGNAL/num_segments) # Varies as a function of duration 

expected_no_mfccvec_per_segment=math.ceil(num_samples_per_segment/hop_length) # hop_length is the measure of overlapping window--> so vector size will increase if hop_length is small (more overlapping)

print(f'sample/signal:{SAMPLES_PER_SIGNAL},no_samples/segment:{num_samples_per_segment},hop_length:{hop_length},expected_mfccs:{expected_no_mfccvec_per_segment}')

'''



'''

# Sampling for multiple MFCCs

for s in range(num_segments):

    start_sample=num_samples_per_segment*s # For s=0: start

    end_sample=start_sample+num_samples_per_segment # for s=0; start+no_of_samples/segment



    plt.figure(figsize=(12,3))

    plt.plot(librosa_audio,'r')

    plt.plot(librosa_audio[start_sample:end_sample],'g')

    plt.show()

    

    mfcc=librosa.feature.mfcc(librosa_audio[start_sample:end_sample],hop_length=hop_length,n_fft=n_fft,n_mfcc=n_mfcc) # Analyzing a slice of a signal 

    mfcc=mfcc.T 

    librosa.display.specshow(mfcc,sr=SAMPLE_RATE,hop_length=hop_length)

    plt.xlabel('mfcc')

    plt.ylabel('time')

    plt.colorbar()

    plt.show()

    

    write("example"+str(s)+".wav", librosa_sample_rate, librosa_audio[start_sample:end_sample])

    #ipd.Audio("example.wav")

    

    #print("segment_"+str(s+1),mfcc.shape)

    # Some times audio samples dont have expected length: so different shape of MFCC , so need to use expected_no_mfcvec_per_segment

    # Store mfcc for segment if it has the expected length

    if len(mfcc)==expected_no_mfccvec_per_segment+1:

        print("pass")

        data["mfcc"].append(mfcc.tolist())

    

'''
bird_classes={}



for ind,val in enumerate(bird_names):

    bird_classes[val]=ind
bird_classes
from tqdm import tqdm

import json







def silenceremoval(audio_clip):



    # Load the audio file in librosa

    librosa_audio,librosa_sample_rate=librosa.load(audio_clip)



    # Finding Amplitude array from the audio

    db = librosa.core.amplitude_to_db(librosa_audio)



    # Finding the mean & std of the amplitude

    mean_db = np.abs(db).mean()

    std_db = db.std()

    print(mean_db,std_db)



    # Splitting the audio into non-silent intervls

    audio_split_intervals= librosa.effects.split(y=librosa_audio, top_db = mean_db - std_db)



    # removes silences from clip

    silence_removed = []

    for inter in audio_split_intervals:

        silence_removed.extend(librosa_audio[inter[0]:inter[1]])

    silence_removed = np.array(silence_removed)

    return silence_removed,librosa_sample_rate











features=[]

ind=0

Par_Fold='../input/birdsong-recognition'

data={"features":[],"classes":[]}

json_file="data.json"



ind1=0









for index,row in tqdm(subset.iterrows()):

    if ind1==0:

        try:

            #print(ind1)

            file_name=os.path.join(os.path.abspath(Par_Fold),str("train_audio"),str(row['ebird_code']),str(row['filename']))

            clas_name=row['species'] # Bird name

            clas_no=bird_classes[row['ebird_code']] # get the number for the class 

            #print(clas_no)





            # Parameters for MFCC

            hop_length=512

            n_fft=2048

            n_mfcc=13

            #expected_no_mfccvec_per_segment=700



            # Storing the data

            librosa_audio,librosa_sample_rate=silenceremoval(file_name)

            #librosa_audio, librosa_sample_rate = librosa.load(file_name) # Librosa load . Constant Sample rate ~ 22000 Hz

            SAMPLE_RATE=librosa_sample_rate # Defining the Sample rate



            DURATION=librosa.get_duration(librosa_audio)

            #DURATION=row['duration']# Duration in seconds

            SAMPLES_PER_SIGNAL=SAMPLE_RATE*DURATION # Varies as a function of duration if sample rate is fixed

            segment_duration=5 # seconds

            total_samples_reqd_segment=SAMPLE_RATE*segment_duration

            num_samples_per_segment=total_samples_reqd_segment

            expected_no_mfccvec_per_segment=math.ceil(num_samples_per_segment/hop_length)

            #num_samples_per_segment=math.ceil(expected_no_mfccvec_per_segment*hop_length)

            num_segments=math.ceil(int(SAMPLES_PER_SIGNAL)/num_samples_per_segment)

            #print(num_segments,SAMPLES_PER_SIGNAL,num_samples_per_segment,DURATION)

            ind=0

            #if ind<1:

                # Sampling for multiple MFCCs

            for s in range(0,(num_segments)):

                #print(ind)

                if ind > 5:

                    break



                start_sample=int(num_samples_per_segment*s) # For s=0: start

                end_sample=start_sample+num_samples_per_segment # for s=0; start+no_of_samples/segment

                if end_sample>SAMPLES_PER_SIGNAL:

                    break



                #plt.figure(figsize=(12,3))

                #plt.plot(librosa_audio,'r')

                #plt.plot(librosa_audio[start_sample:end_sample],'g')

                #plt.show()



                mfcc=librosa.feature.mfcc(librosa_audio[start_sample:end_sample],hop_length=hop_length,n_fft=n_fft,n_mfcc=n_mfcc) # Analyzing a slice of a signal 

                mfcc=mfcc.T 

                #librosa.display.specshow(mfcc,sr=SAMPLE_RATE,hop_length=hop_length)

                #plt.xlabel('mfcc')

                #plt.ylabel('time')

                #plt.colorbar()

                #plt.show()



                #write("example"+str(s+1)+".wav", librosa_sample_rate, librosa_audio[start_sample:end_sample])

                #ipd.Audio("example.wav")

                ind=ind+1

                if len(mfcc)==expected_no_mfccvec_per_segment:

                    #print(s,"pass",mfcc.shape)

                    #features.append([mfcc.tolist(),clas_no])

                    data["features"].append(mfcc.tolist())

                    data["classes"].append(clas_no)

        

        except:

            print("Something went wrong: next entry")

            pass

            #break

        

    else:

        break

    #ind1+=1





with open(json_file,"w") as fp:

    json.dump(data, fp, indent=4)

    







len(data['classes'])
print(1)