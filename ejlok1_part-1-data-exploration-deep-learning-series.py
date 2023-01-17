import pandas as pd       
import os 
import math 
import numpy as np
import matplotlib.pyplot as plt  
import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import os
os.chdir("../input")
print(os.listdir("../input"))
#load the data 
df = pd.read_csv("speakers_all.csv", header=0)

# Check the data
print(df.shape, 'is the shape of the dataset') 
print('------------------------') 
print(df.head())
df.drop(df.columns[9:12],axis = 1, inplace = True)
print(df.columns)
df.describe()
# Very rough plot
df['country'].value_counts().plot(kind='bar')
# Ok so that plot wasn't very good for that category. Lets try another category... 
df['native_language'].value_counts().plot(kind='bar')
# That's lots of categories too! Ok so maybe lets try a different way...
df.groupby("native_language")['age'].describe().sort_values(by=['count'],ascending=False)
# Check country of origin again...
df.groupby("country")['age'].describe().sort_values(by=['count'],ascending=False)
# Create DTM of counts 
df.groupby("sex")['age'].describe()
# birthplace
df.groupby("birthplace")['age'].describe().sort_values(by=['count'],ascending=False)
# file_missing
df.groupby("file_missing?")['age'].describe().sort_values(by=['count'],ascending=False)
# Count the total audio files given
print (len([name for name in os.listdir('../input/recordings/recordings') if os.path.isfile(os.path.join('../input/recordings/recordings', name))]))
# filename column. This time we just print out the first 10 records. 
df.groupby("filename")['age'].describe().sort_values(by=['count'],ascending=False).head(10)
# Cross-tab. Again, just print the first 10 record 
df.groupby("filename")['file_missing?'].describe().sort_values(by=['count'],ascending=False).head(10)
# pd.crosstab(df['filename'],df['file_missing?']) as an alternative method 
# Play afrikaans1
fname1 = 'recordings/recordings/' + 'afrikaans1.mp3'
ipd.Audio(fname1)
# Play mandarin46
fname2 = 'recordings/recordings/' + 'mandarin46.mp3'
ipd.Audio(fname2)
# lets have a listen to a male voice. 
print(df.groupby("filename")['sex'].describe().head(10))
fname3 = 'recordings/recordings/' + 'agni1.mp3'   
ipd.Audio(fname3)
print(df[df['birthplace'].str.contains("kentucky",na=False)])
fname4 = 'recordings/recordings/' + 'english385.mp3'   
ipd.Audio(fname4)
fname5 = 'recordings/recordings/' + 'english462.mp3'   
ipd.Audio(fname5)
fname6 = 'recordings/recordings/' + 'english381.mp3'   # An older male 
ipd.Audio(fname6)