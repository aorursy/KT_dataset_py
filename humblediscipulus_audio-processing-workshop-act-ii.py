!apt install -y ffmpeg #needed by librosa to function correctly



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Audio # used for playing audio samples

import tensorflow as tf # tensorflow for all the ML magic

import librosa # for audio signal processing

import librosa.display # for displaying waveforms

from sklearn.preprocessing import StandardScaler

import sys



%matplotlib inline

import matplotlib.pyplot as plt # for plotting graphs

import seaborn as sns # for prettier graphs

sns.set(style="dark")
AUDIO_INPUT_FILE = '../input/audio-conversation/conversation.wav'
y, sr = librosa.load(AUDIO_INPUT_FILE)
print(type(y), type(sr))

print(y.shape, sr)
# calculated duration based on array length and sample rate

print(y.shape[0] / sr * 1.0)



# native duration value directly from the file

print(librosa.get_duration(filename=AUDIO_INPUT_FILE))
Audio(data=y, rate=sr)
plt.figure(figsize=(22, 4))

librosa.display.waveplot(y, sr, color='#7B1FA2', alpha=0.8)

plt.title('Lost in Translation - "Does it get easier?"')

plt.show()
X = librosa.stft(y)

Xdb = librosa.amplitude_to_db(X)
plt.figure(figsize=(27, 5))

librosa.display.specshow(Xdb, sr = sr, x_axis='time', y_axis='linear', cmap="magma")

plt.colorbar(format='%+2.0f dB')

plt.show()
plt.figure(figsize=(27, 5))

librosa.display.specshow(Xdb, sr = sr, x_axis='time', y_axis='log', cmap="magma")

plt.colorbar(format='%+2.0f dB')

plt.show()
division_per_second = 8



chunk_time = 1. / division_per_second

chunk_size = sr // division_per_second

print(chunk_size, chunk_time)
left_over = y.shape[0] % chunk_size

num_of_chunks = y[:-left_over].shape[0]/chunk_size



y_split = np.split(y[:-left_over], num_of_chunks)

len(y_split)
feature_vector_mfcc = np.array([ librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc = 40) for chunk in y_split ])

feature_vector_mfcc.shape
feature_vector_spec_rolloff = np.array([librosa.feature.spectral_rolloff(y=chunk, sr=sr) for chunk in y_split])

feature_vector_spec_rolloff.shape
feature_vector_spec_centroid = np.array([librosa.feature.spectral_centroid(y=chunk, sr=sr) for chunk in y_split])

feature_vector_spec_centroid.shape
feature_vector_zcr = np.array([librosa.feature.zero_crossing_rate(y=chunk) for chunk in y_split])

feature_vector_zcr.shape
feature_vector_mfcc_mean = np.mean(feature_vector_mfcc, axis = 2)

feature_vector_mfcc_mean.shape
feature_vector_spec_rolloff_mean = np.mean(feature_vector_spec_rolloff, axis = 2)

feature_vector_spec_rolloff_mean.shape
feature_vector_spec_centroid_mean = np.mean(feature_vector_spec_centroid, axis = 2)

feature_vector_spec_centroid_mean.shape
feature_vector_zcr_mean = np.mean(feature_vector_zcr, axis = 2)

feature_vector_zcr_mean.shape
feature_matrix = np.hstack((

    feature_vector_mfcc_mean,

    feature_vector_spec_rolloff_mean,

    feature_vector_spec_centroid_mean,

    feature_vector_zcr_mean,

))

feature_matrix.shape
scaler = StandardScaler()



normalized_feature_matrix = scaler.fit_transform(feature_matrix)

normalized_feature_matrix = feature_matrix

normalized_feature_matrix.shape
feature_matrix_df = pd.DataFrame(normalized_feature_matrix)

feature_matrix_df.columns = [

               "MFCC-1", "MFCC-2", "MFCC-3", "MFCC-4", 

               "MFCC-5", "MFCC-6", "MFCC-7", "MFCC-8", 

               "MFCC-9", "MFCC-10", "MFCC-11", "MFCC-12", 

               "MFCC-13", "MFCC-14", "MFCC-15", "MFCC-16", 

               "MFCC-17", "MFCC-18", "MFCC-19", "MFCC-20",

               "MFCC-21", "MFCC-22", "MFCC-23", "MFCC-24", 

               "MFCC-25", "MFCC-26", "MFCC-27", "MFCC-28", 

               "MFCC-29", "MFCC-30", "MFCC-31", "MFCC-32", 

               "MFCC-33", "MFCC-34", "MFCC-35", "MFCC-36", 

               "MFCC-37", "MFCC-38", "MFCC-39", "MFCC-40",

               "Spec Roll-off", 

               "Spec Centroid", 

               "ZCR"

              ]



# take a peak in the data

feature_matrix_df.head()
# Number of cluster we wish to divide the data into( user tunable )

k = 3



# Max number of allowed iterations for the algorithm( user tunable )

epochs = 10000
# return an np-array of k points based on k-means++

def selectCentroids(k, dummy_data):

    centroids = [] 

    

    # pick the first centroid at random

    centroids.append(dummy_data[np.random.randint( 

            dummy_data.shape[0]), :]) 

    

    # compute remaining k - 1 centroids 

    for centroid_index in range(k - 1):  

        # points from nearest centroid 

        distance_array = [] 

        

        # iterate over the data points for each centroid

        # to find the distance from nearest chosen centroids

        for i in range(dummy_data.shape[0]): 

            point = dummy_data[i, :]

            distance = sys.maxsize 

              

            ## compute distance of 'point' from each of the previously 

            ## selected centroid and store the minimum distance 

            for j in range(len(centroids)): 

                temp_distance = np.linalg.norm(point - centroids[j]) 

                distance = min(distance, temp_distance) 

            distance_array.append(distance) 

              

        ## select data point with maximum distance as our next centroid 

        distance_array = np.array(distance_array) 

        next_centroid = dummy_data[np.argmax(distance_array), :]

        centroids.append(next_centroid) 

    return np.array(centroids)



# utility to assign centroids to data points

def assignCentroids(X, C):  

    expanded_vectors = tf.expand_dims(X, 0)

    expanded_centroids = tf.expand_dims(C, 1)

    distance = tf.math.reduce_sum( tf.math.square( tf.math.subtract( expanded_vectors, expanded_centroids ) ), axis=2 )

    return tf.math.argmin(distance, 0)

                                              

# utility to recalculate centroids

def reCalculateCentroids(X, X_labels):

    sums = tf.math.unsorted_segment_sum( X, X_labels, k )

    counts = tf.math.unsorted_segment_sum( tf.ones_like( X ), X_labels, k  )

    return tf.math.divide( sums, counts )
data = feature_matrix_df



X = tf.Variable(data.values, name="X")

X_labels = tf.Variable(tf.zeros(shape=(X.shape[0], 1)),name="C_lables")

C = tf.Variable(selectCentroids(k, normalized_feature_matrix), name="C")



for epoch in range( epochs ):

    X_labels =  assignCentroids( X, C )

    C = reCalculateCentroids( X, X_labels )
X_labels
def frame_condencer( X_labels ):

    condenced_frame_vectors = []

    current_label = -1

    

    for label in X_labels:

        if current_label != label:

            condenced_frame_vectors.append((label, 1))

        else:

            condenced_frame_vectors[-1][1] += 1

    return condenced_frame_vectors



def rect_generator( condenced_frame_vectors, plt, time = 0.25 ):

    rect_height = plt.ylim()[1] - plt.ylim()[0]

    rect_range = plt.xlim()[1]

    

    color_map = {

        0:"r",

        1:"b",

        2:"g",

        3:"y"

    }

    

    rect_list = []

    current_count = 0

    for vector in condenced_frame_vectors:

        rect_list.append( patches.Rectangle(

            (current_count * time,plt.ylim()[0]),

            vector[1] * time, 

            rect_height, 

            linewidth=1, 

            edgecolor='none', 

            facecolor=color_map[vector[0]], 

            alpha=0.5

        ))

        current_count += vector[1]

    return rect_list
import matplotlib.patches as patches



plt.figure(figsize=(32, 8))

librosa.display.waveplot(y, sr, color='#7B1FA2', alpha=0.8)

rect = patches.Rectangle((0,0),5,0.75,linewidth=1,edgecolor='none',facecolor='r', alpha=0.5)

ax = plt.gca()



condenced_frame_vectors = frame_condencer(X_labels.numpy().tolist())

rect_list = rect_generator(condenced_frame_vectors, plt, time = chunk_time)



for rect in rect_list:

    ax.add_patch(rect)

plt.title('Lost in Translation - "Does it get easier?" - Speaker diarization')

plt.show()
Audio(data=y, rate=sr)