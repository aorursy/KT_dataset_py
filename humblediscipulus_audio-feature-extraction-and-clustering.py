!apt install -y ffmpeg

!pip install eyed3

!pip install pydub

!pip install pyAudioAnalysis



import os

import tensorflow as tf

import numpy as np

import pandas as pd

from pyAudioAnalysis import audioBasicIO

from pyAudioAnalysis import audioFeatureExtraction

import matplotlib.pyplot as plt
def preProcess( fileName ):

    # Extracting wav file data

    [Fs, x] = audioBasicIO.readAudioFile(fileName);



    # If double channel data then take mean

    if( len( x.shape ) > 1 and  x.shape[1] == 2 ):

        x = np.mean( x, axis = 1, keepdims = True )

    else:

        x = x.reshape( x.shape[0], 1 )

    

    # Extract the raw chromagram data, expected dimention is [ m,  ] not [ m, 1 ]

    F, f_names = audioFeatureExtraction.stFeatureExtraction(

        x[ :, 0 ], 

        Fs, 0.050*Fs, 

        0.025*Fs

    )

    

    return (f_names, F)
def getChromagram( audioData ):

    # chronograph_1

    temp_data =  audioData[ 21 ].reshape( 1, audioData[ 21 ].shape[0] )

    chronograph = temp_data

    

    # looping through the next 11 stacking them vertically

    for i in range( 22, 33 ):

        temp_data =  audioData[ i ].reshape( 1, audioData[ i ].shape[0] )

        chronograph = np.vstack( [ chronograph,  temp_data ] )

    

    return chronograph
def getNoteFrequency( chromagram ):

    

    # Total number of time frames in the current sample

    numberOfWindows = chromagram.shape[1]

    

    # Taking the note with the highest amplitude

    freqVal = chromagram.argmax( axis = 0 )

    

    # Converting the freqVal vs time to freq count

    histogram, bin = np.histogram( freqVal, bins = 12 ) 

    

    # Normalizing the distribution by the number of time frames

    normalized_hist = histogram.reshape( 1, 12 ).astype( float ) / numberOfWindows

    

    return normalized_hist
def plotHeatmap( chromagraph, smallSample = True ):

    

    notesLabels = [ "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C", "B", "A#", "A" ]

    

    fig, axis = plt.subplots()

    

    if smallSample:

        im = axis.imshow( chromagram[ :, 0 : 25 ], cmap = "YlGn" )

    else:

        im = axis.imshow( chromagram )

        

    cbar = axis.figure.colorbar(im, ax = axis,  cmap = "YlGn")

    cbar.ax.set_ylabel("Amplitude", rotation=-90, va="bottom")

    

    axis.set_yticks( np.arange( len(notesLabels) ) )

    axis.set_yticklabels(notesLabels)

    

    axis.set_title( "chromagram" )

    

    fig.tight_layout()

    _ = plt.show()
def noteFrequencyPlot( noteFrequency, smallSample = True ):

    

    fig, axis = plt.subplots(1, 1, sharey=True )

    

    axis.plot( np.arange( 1, 13 ), noteFrequency[0, :] )

    

    _ = plt.show()
feature_name, features = preProcess( "../input/c1_2.wav" )
chromagram = getChromagram( features )

plotHeatmap( chromagram )
noteFrequency = getNoteFrequency( chromagram )

noteFrequencyPlot( noteFrequency )
fileList = []



def getDataset( filePath ):

    X = pd.DataFrame(  )

    

    columns=[ "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C", "B", "A#", "A" ]

    

    for root, dirs, filenames in os.walk( filePath ):

        for file in filenames:

            fileList.append( file )

            feature_name, features = preProcess(filePath + file )

            chromagram = getChromagram( features )

            noteFrequency = getNoteFrequency( chromagram )

            x_new =  pd.Series(noteFrequency[ 0, : ])

            X = pd.concat( [ X, x_new ], axis = 1 )

        

    data = X.T.copy()

    data.columns = columns

    data.index = [ i for i in range( 0, data.shape[ 0 ] ) ]

            

    return data
data = getDataset( "../input/" )
data
# Number of cluster we wish to divide the data into( user tunable )

k = 3



# Max number of allowed iterations for the algorithm( user tunable )

epochs = 2000
def initilizeCentroids( data, k ):

    '''

    Initilize cluster centroids( assuming random k data points to be centroid return them )

    '''

    centroids = data.values[ 0 : k ]

    return centroids
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
X = tf.Variable(data.values, name="X")

X_labels = tf.Variable(tf.zeros(shape=(X.shape[0], 1)),name="C_lables")

C = tf.Variable(initilizeCentroids( data, k ), name="C")



for epoch in range( epochs ):

    X_labels =  assignCentroids( X, C )

    C = reCalculateCentroids( X, X_labels )
final_labels = pd.DataFrame( { "Labels": X_labels, "File Names": fileList } )

final_labels