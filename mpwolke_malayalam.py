#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRUAI6bV2uhlzKIzHO04gwdcfrPcXV3UBKazXo3KngevkdiQy4x',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/malayalam-multispeaker-speech-data-set/line_index_female.tsv', sep='\t', error_bad_lines=False)
df.head()
malay = df['കൂടുതൽ വിവരങ്ങൾ വരുമ്പോൾ തിരിച്ചു ചേർക്കാം'].unique()

malay
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('കൂടുതൽ വിവരങ്ങൾ വരുമ്പോൾ തിരിച്ചു ചേർക്കാം').size()/df['mlf_02879_01795762363'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
sns.countplot(df["mlf_02879_01795762363"])

plt.xticks(rotation=90)

plt.show()
import librosa

import librosa.display

import IPython.display as ipd

import warnings

warnings.filterwarnings('ignore')
# Use one audio file in previous parts again

fname = '/kaggle/input/malayalam-multispeaker-speech-data-set/ml_in_male/mlm_08777_01770418244.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)
# Use one audio file in previous parts again

fname = '/kaggle/input/malayalam-multispeaker-speech-data-set//ml_in_female/mlf_07754_00409559679.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSnD3521QaBc3vhLhUMCnnRyMdPX4UvV-A2-ABz2mi_fjnGSwE_',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQS9tu9acPY-ywDT5e-CVug1sN4wPQ8wllTg85n-sRROBTRex86',width=400,height=400)