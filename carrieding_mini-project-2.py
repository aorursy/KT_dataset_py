# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import csv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Fitbit = pd.read_csv("../input/appreview/Fitbit.csv")

GarminConnect = pd.read_csv("../input/appreview/GarminConnect.csv")

MiFit = pd.read_csv("../input/appreview/MiFit.csv")

Misfit = pd.read_csv("../input/appreview/Misfit.csv")

Fitbit_Sentiment = pd.read_csv("../input/appreview/processed_batch_Fitbit_Sentiment.csv")

GarminConnect_Sentiment = pd.read_csv("../input/appreview/processed_batch_GarminConnect_Sentiment.csv")

MiFit_Sentiment = pd.read_csv("../input/appreview/processed_batch_MiFit_Sentiment.csv")

Misfit_Sentiment = pd.read_csv("../input/appreview/processed_batch_Misfit_Sentiment.csv")
# Delete irrelevent columns to make it more manageable

Fitbit = Fitbit.drop(columns = ['updated', 'im', 'xmlns', 'lang', 'id', 'title', 'link__rel', 'link__type', 'link__href', 'icon', 'author__name', 'author__uri', 'rights', 'entry__id', 'entry__im:contentType__term', 'entry__im:contentType__label', 'entry__im:voteSum', 'entry__im:voteCount', 'entry__author__name', 'entry__author__uri', 'entry__link__rel'])

GarminConnect = GarminConnect.drop(columns = ['updated', 'im', 'xmlns', 'lang', 'id', 'title', 'link__rel', 'link__type', 'link__href', 'icon', 'author__name', 'author__uri', 'rights', 'entry__id', 'entry__im:contentType__term', 'entry__im:contentType__label', 'entry__im:voteSum', 'entry__im:voteCount', 'entry__author__name', 'entry__author__uri', 'entry__link__rel'])

MiFit = MiFit.drop(columns = ['updated', 'im', 'xmlns', 'lang', 'id', 'title', 'link__rel', 'link__type', 'link__href', 'icon', 'author__name', 'author__uri', 'rights', 'entry__id', 'entry__im:contentType__term', 'entry__im:contentType__label', 'entry__im:voteSum', 'entry__im:voteCount', 'entry__author__name', 'entry__author__uri', 'entry__link__rel'])

Misfit = Misfit.drop(columns = ['updated', 'im', 'xmlns', 'lang', 'id', 'title', 'link__rel', 'link__type', 'link__href', 'icon', 'author__name', 'author__uri', 'rights', 'entry__id', 'entry__im:contentType__term', 'entry__im:contentType__label', 'entry__im:voteSum', 'entry__im:voteCount', 'entry__author__name', 'entry__author__uri', 'entry__link__rel'])
#drop rows with empty data since it's associated with rows with HTML, which is redundant with reviews in text format

Fitbit = Fitbit.dropna()

GarminConnect = GarminConnect.dropna()

MiFit = MiFit.dropna()

Misfit = Misfit.dropna()
#drop "entry content type" column since the reviews are now the same type

Fitbit = Fitbit.drop(columns =['entry__content__type'])

GarminConnect = GarminConnect.drop(columns =['entry__content__type'])

MiFit = MiFit.drop(columns =['entry__content__type'])

Misfit = Misfit.drop(columns =['entry__content__type'])
#export preprocessing data in csv format. It should show up in output.

Fitbit.to_csv('Fitbit_ML.csv')

GarminConnect.to_csv('GarminConnect_ML.csv')

MiFit.to_csv('MiFit_ML.csv')

Misfit.to_csv('Misfit_ML.csv')
Fitbit.dtypes
import matplotlib.pyplot as plt
hist = Fitbit.hist(column='entry__im:rating')

plt.title('Fitbit App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')



hist = GarminConnect.hist(column='entry__im:rating')

plt.title('Garmin Connect App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')



hist = MiFit.hist(column='entry__im:rating')

plt.title('MiFit App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')



hist = Misfit.hist(column='entry__im:rating')

plt.title('Misfit App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')


Fitbit_Sentiment['Classification'].value_counts().plot(kind='bar')

plt.title('Fitbit App Review Rating')

plt.xlabel('Type of Rating')

plt.ylabel('Number of Reviews')
GarminConnect_Sentiment['Classification'].value_counts().plot(kind='bar')

plt.title('Garmin Connect App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')
MiFit_Sentiment['Classification'].value_counts().plot(kind='bar')

plt.title('MiFit App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')
Misfit_Sentiment['Classification'].value_counts().plot(kind='bar')

plt.title('Misfit App Review Rating')

plt.xlabel('Rating')

plt.ylabel('Number of Reviews')