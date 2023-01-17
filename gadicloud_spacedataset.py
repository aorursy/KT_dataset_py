# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import regularizers, Sequential
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import pickle
from keras.activations import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('------------------')
print('Space Data Details')
print('------------------')
df = pd.read_excel("/kaggle/input/spacedata/spaceDataSheet.xlsx", sheet_name=0, index_col=0)
print(df)
df1 = pd.read_excel("/kaggle/input/spacedata/spaceDataSheet.xlsx", sheet_name=0, index_col=0)


df = df[["Company Name","Datum"]]
df['Datum'] = df['Datum'].apply(lambda x : pd.to_datetime(str(x), utc=True))
df['Datum'] = df['Datum'].dt.date
df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
df['Datum'] = df['Datum'].dt.year
df.head()
df_allCount = df.groupby(['Datum'])['Datum'].count().reset_index(name="count")
#https://seaborn.pydata.org/generated/seaborn.scatterplot.html
fig = plt.gcf()
fig.set_size_inches(12, 8)

sns.scatterplot(x='Datum', 
                y='count', 
                data=df_allCount)
#This timeline of artificial satellites and space probes includes unmanned spacecraft including technology demonstrators, 
#observatories, lunar probes, and interplanetary probes. The details show, the launch operations were at high during 1965-1975, 
# then gradually decreasing around 1980-2000. with initiatives like spaceX, one could see rise in space explorations after 2000.
df.head()
df_countryWiseCount = df.groupby(['Company Name','Datum'])['Datum'].count().reset_index(name="count")
df_countryWiseCount.columns = ['Organization', 'Year', 'Count']
df_countryWiseCount
df_countryWiseCount.columns = ['Organization', 'Year', 'Count']

#fig = plt.gcf()
#fig.set_size_inches( 50, 10)
sns.catplot(x="Organization", kind="count", palette="ch:.25", data=df_countryWiseCount, height=20, aspect=3);
df
#df_statusRatio = df.groupby(['Company Name', 'Status Mission'])['Status Mission'].count().reset_index(name="Ratiocount")
df1
#Company Name     Status Mission  Ratiocount
df_statusRatio = df1.groupby(['Company Name', 'Status Mission'])['Status Mission'].count().reset_index(name="Ratiocount")

sns.catplot(x="Company Name", y="Ratiocount", hue="Status Mission", kind="bar", data=df_statusRatio, height=20, aspect=3);
print('Mission Status wrt to Success, Failure etc based on Organisations')
print('Working on the Descriptive Statistics part to describe the problem')
df_stats1 = (df_statusRatio['Status Mission'].value_counts()/df_statusRatio['Status Mission'].count())*100
df_stats1
print('41% of the times the launch missions have been successful')
print('36% of the times the launch missions have been failure')
print('19% of the times the launch missions have been partial failure')
print('3% of the times the launch missions have been prelaunch failure')
print('Success/Failure Criteria based on organizations')
df_statusRatio.head(5)
df_stats2 =  df_statusRatio
s = df_stats2.groupby('Company Name')['Ratiocount'].transform('sum')
df_stats2['Percentage'] = df_stats2['Ratiocount'].div(s).mul(100).round()

snsstat2 = sns.catplot(x="Company Name", y="Percentage", palette="ch:.75", hue="Status Mission", kind="bar", data=df_stats2, height=15, aspect=2);
plt.xlabel("Company")
plt.ylabel("Percent")
plt.title("Companies vs Launch Criterias") # You can comment this line out if you don't need title
plt.show(snsstat2)
# HEAT MAP BASED ON COUNTRIES AND YEAR WISE
df_stats3 = df_stats2
Index= df_stats3['Company Name'].tolist()
Cols = df_stats3['Status Mission'].unique().tolist()

heatmap1_data = pd.pivot_table(df_stats3, values='Ratiocount', 
                     index='Company Name', 
                     columns='Status Mission')
plt.figure(figsize=(15, 17))
plt.title('Heatmap view of Mission Launches for different organizations and their counts')
sns.heatmap(heatmap1_data, cmap="BuGn", annot=True, linewidths=0.30)
Index= df_stats3['Company Name'].tolist()
Cols = df_stats3['Status Mission'].unique().tolist()

heatmap1_data = pd.pivot_table(df_stats3, values='Percentage', 
                     index='Company Name', 
                     columns='Status Mission')
plt.figure(figsize=(15, 17))
plt.title('Heatmap view of Mission Launches for different organizations percentage wise')
sns.heatmap(heatmap1_data, cmap="BuGn", annot=True, linewidths=0.30)
df2 = df1
df2
df2['Country']= [x.rsplit(",", 1)[-1] for x in df2["Location"]]
df2
df_countries = df2.groupby(['Country', 'Status Mission'])['Status Mission'].count().reset_index(name="countryWiseCount")

sns.catplot(x="Country", y="countryWiseCount", hue="Status Mission", kind="bar", palette="ch:.25", data=df_countries, height=20, aspect=3);

#sns.catplot(x="Company Name", y="countryWiseCount", hue="Status Mission", kind="bar", data=df_statusRatio, height=20, aspect=3);
print('Country wise Mission Status wrt to Success, Failure')
print("OTHER NOTEBOOKS TO FOLLOW.")