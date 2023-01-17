#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQxItufa3tAIZtMvf4JqJU7hlGGpVtz8Ti-JDF83tQ1HSvRMUFT&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/metadata.csv')

df.shape
df.head()
title = df.copy()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRRtwq7VwNmlnYqOSU_DM-5N6XTMai8Vx93YnBy1lpP6m43FurP&usqp=CAU',width=400,height=400)
title = title.dropna(subset=['title'])
title.head()
title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)

title['title'] = title['title'].str.lower()
title.tail()
title['keyword_interleukin'] = title['title'].str.find('interleukin')
title.head()
included_interleukin = title.loc[title['keyword_interleukin'] != -1]

included_interleukin
import json

file_path = '/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/document_parses/pdf_json/5726297380b86a67cf694c0483d546051d1e8be6.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
IL = pd.read_csv('../input/cusersmarildownloadsinterleukincsv/interleukin.csv', sep=';')

IL
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTO67Jx2Ys2OT7N7G5mNIqCp6xVenXoCZgt0_IozHN9cGW-Xqw8&usqp=CAU',width=400,height=400)
IL.describe()
IL['204252_at'].hist()

plt.show()
ax = IL.groupby('Samples')['204252_at'].max().sort_values(ascending=True).plot(kind='barh', figsize=(12,8),

                                                                                   title='Maximum Interleukin-6 Samples')

plt.xlabel('204252_at')

plt.ylabel('Samples')

plt.show()
ax = IL.groupby('Samples')['211804_s_at', '211803_at'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='211804_s_at x 211803_at')

plt.xlabel('Samples')

plt.ylabel('Log Scale 211804_s_at')

plt.show()
ax = IL.groupby('Samples')['211804_s_at', '211803_at'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='211804_s_at x 211803_at', logx=True, linewidth=3)

plt.xlabel('Log Scale 211804_s_at')

plt.ylabel('Samples')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSfKM9ny6lvxiXRkLxDps9NFRuWdpSy2Bt3THOwDJTWSSPoVoqy&usqp=CAU',width=400,height=400)