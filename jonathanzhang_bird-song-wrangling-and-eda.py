# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import librosa
import librosa.display
from IPython import display
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
ex_file = ('/kaggle/input/birdsong-recognition/train_audio'+ '/' + 
           train['ebird_code']+ '/' + 
           train['filename']).iloc[4423] #4423
x, sr = librosa.load(ex_file)
bird_species_count = train['species'].value_counts(ascending=False)
plt.xticks([])
bird_species_count.plot()
top_15 = list(train['country'].value_counts().head(15).reset_index()['index'])
data = train[train['country'].isin(top_15)]

plt.figure(figsize=(16,6))
ax = sns.countplot(data['country'],palette = 'muted', order = data['country'].value_counts().index)

plt.title('top 15 Countries with most Recordings', fontsize=15)
plt.ylabel('Frequency',fontsize=13)
plt.yticks(fontsize=12)
plt.xticks(rotation=45,fontsize=12)
plt.xlabel('')
df = px.data.gapminder().query('year==2007')[['country','iso_alpha']]

data = pd.merge(left=train, right = df,how = 'inner',on='country')

data=data.groupby(by=['country','iso_alpha']).count()['species'].reset_index()

fig = px.choropleth(data,locations = 'iso_alpha'
                    ,color='species'
                    ,hover_name = 'country'
                    , color_continuous_scale = px.colors.sequential.Teal
                    ,title='World: Recordings Per Country')

fig.show()
birdcall_meta_samp.columns
#quality rating for audio files

rating = list(train['rating'].value_counts().reset_index()['index'])
rating_data = train['rating']

plt.figure(figsize=(16, 6))
ax = sns.countplot(train['rating'], palette="muted", order = rating_data.value_counts().index.sort_values(ascending=False))
#distribution of audio duration

plt.figure(figsize=(16, 6))
duration = sns.distplot(train['duration'])
duration_adjusted = train['duration'][train['duration'].between(train['duration'].quantile(.05), train['duration'].quantile(.95))] 
plt.figure(figsize=(16, 6))
duration = sns.distplot(duration_adjusted)
speed = sns.countplot(train['speed'])
speed = sns.countplot(train['pitch'])
speed = sns.countplot(train['number_of_notes'])
#sample songs

files = ('/kaggle/input/birdsong-recognition/train_audio'+ '/' + 
           train['ebird_code']+ '/' + 
           train['filename'])
files_samp = files.sample(5)
x = []
sr = []
for i in files_samp:
    a,b = librosa.load(i)
    x.append(a)
    sr.append(b)
display.Audio(data=x[0],rate=sr[0])
#norpar/XC235682.mp3
display.Audio(data=x[1],rate=sr[1])
#bkbwar/XC217955.mp3
display.Audio(data=x[2],rate=sr[2])
#logshr/XC192339.mp3
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

le = preprocessing.LabelEncoder()
birdcall_meta_samp['class_code'] = le.fit_transform(birdcall_meta_samp['ebird_code'])
birdcall_train, birdcall_test = train_test_split(birdcall_meta_samp, test_size=0.2, random_state=0, stratify=birdcall_meta_samp[['ebird_code']])

from scipy.ndimage.morphology import binary_erosion,binary_dilation
classes_size = birdcall_train['ebird_code'].nunique()

sec_split = 3
obs_train = birdcall_train['chunks'].sum()
obs_test = birdcall_test['chunks'].sum()
classes_size
X_train = np.zeros((obs_train, 128, 130))
Y_train = np.zeros((obs_train, classes_size))
X_test = np.zeros((obs_test,128,130))
y_test = np.zeros((obs_test,classes_size))
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
#minmaxscaler = MinMaxScaler()
def erosion(m):
    norm_m=(m-m.min())/(m.max()-m.min())
    column_medians=np.median(norm_m,axis=0)
    row_medians=np.median(norm_m)
    eroded_spectrogram = binary_erosion(np.greater(norm_m,column_medians*3)&np.greater(norm_m.T,row_medians*3).T*1)
    return eroded_spectrogram

def dilation(x,e):
    dilated = binary_dilation(e.sum(axis=0)>0,  iterations=3)
    x = x[np.round(np.interp(np.arange(x.shape[0]),
                             np.arange(dilated.shape[0])*x.shape[0]/dilated.shape[0],
                             dilated)).astype(bool)]
    return x
i=0

for r in birdcall_train[['path','class_code']].iterrows():
    # loading to lr
    x, sr = librosa.load(r[1]['path'])
    
    S = librosa.feature.melspectrogram(x, sr=sr, n_fft=1028, hop_length=512, n_mels=128)
    
    #de-noising
    
    eroded_spec = erosion(S)
    x = dilation(x,eroded_spec)
    
    window_slice = np.floor(x.shape[0]/sr/sec_split) # dividing the full length of the array divided by the sampling rate and pre-set sectional split of 3

    x=x[:int(window_slice*sec_split*sr)] 
       
