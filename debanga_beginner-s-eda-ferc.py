import sys, os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import cv2 

import plotly.graph_objs as go

from plotly.offline import iplot

def plot(x,y,type='scatter',title="title",xlabel="x", ylabel="y"):

    if type=='scatter':

        data = go.Scatter(x=x, y=y)

    elif type=='bar':

        data = go.Bar(x=x, y=y)

        

    layout = go.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))

    fig = go.Figure(data=[data], layout=layout)

    iplot(fig)
import tarfile



#simple function to extract the train data

#tar_file : the path to the .tar file

#path : the path where it will be extracted

def extract(tar_file, path):

    opened_tar = tarfile.open(tar_file)

     

    if tarfile.is_tarfile(tar_file):

        opened_tar.extractall(path)

    else:

        print("The tar file you entered is not a tar file")



# Extract data

extract("/kaggle/input/facial-expression-recognition-challenge/fer2013.tar",".")

# Read Metadata

meta_df = pd.read_csv('./fer2013/fer2013.csv')



emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



meta_df = meta_df.rename(columns={"emotion":"label","pixels":"image"})

meta_df['emotion'] = meta_df['label'].apply(lambda x: emotions[int(x)])

meta_df = meta_df.drop(columns=['Usage'])
meta_df
# Check data

id = 6

image = np.reshape(np.array(meta_df.image[id].split(' ')).astype(int),(48,48))

plt.imshow(image)

print(meta_df.emotion[id])
plot(x=meta_df.emotion.unique().tolist(),

     y=meta_df.groupby('emotion').count().label.tolist(), 

     type='bar',

     title='Emotion Distribution',

     xlabel='Emotions',

     ylabel='Count')
meta_df
# Find the average of all emotion counts

m = meta_df.groupby('label').count().mean().values[0]

print("Mean of all emotion counts: " + str(m))



oversampled = pd.DataFrame()

for emotion in emotions:

    print('\n' + emotion)

    l = len(meta_df[meta_df.emotion==emotion])

    print('Before sampling: ' + str(l))

    

    if (l>=m):

        dft = meta_df[meta_df.emotion==emotion].sample(int(m))

        oversampled = oversampled.append(dft)

        print('Ater sampling: ' + str(len(dft)))

    else:

        frac = int(m/l)

        dft = pd.DataFrame()

        for i in range(frac+1):

            dft = dft.append(meta_df[meta_df.emotion==emotion])

        dft = dft[dft.emotion==emotion].sample(int(m))

        oversampled = oversampled.append(dft)

        print('Ater sampling: ' + str(len(dft)))

        

oversampled = oversampled.sample(frac=1).reset_index().drop(columns=['index'])
plot(x=oversampled.emotion.unique().tolist(),

     y=oversampled.groupby('emotion').count().label.tolist(), 

     type='bar',

     title='Emotion Distribution',

     xlabel='Emotions',

     ylabel='Count')
oversampled.to_csv('metadata_processed.csv', index=False)