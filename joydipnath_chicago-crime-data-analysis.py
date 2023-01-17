# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image # converting images into arrays

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
df = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')

#df.head()
# Convert to pandas datetime format 

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')

df.index = pd.DatetimeIndex(df['Date']) # Index the date column

# df.head()
# Adding few columns

df['time_hour'] = df['Date'].apply(lambda x: x.hour)

df['month'] = df['Date'].apply(lambda x: x.month)

df['year'] = df['Date'].apply(lambda x: x.year)



# removing the column

df = df[df['Date'] != 2017]



df.head()
 # Join together the IUCR code in a single string

crime_types_code = " ".join(crime for crime in df['IUCR'])

#print( crime_types_code)



crime_code_wordcloude = WordCloud().generate(crime_types_code)



plt.figure(figsize = [10,10])

plt.imshow(crime_code_wordcloude, interpolation='bilinear')

plt.axis("off")



plt.show()



# plt.savefig('img/crime_code_wordcloud.png', format='png')

df_wordcloud = df.copy()

df_wordcloud.dropna(axis = 0, subset= ['Block'], inplace=True)



crime_types_location = " ".join(crime for crime in df_wordcloud['Block'])

# mask = np.array(Image,open('us.png'))



crime_location_wordcloud = WordCloud().generate(crime_types_location)



plt.figure(figsize=[30, 30])

plt.imshow(crime_location_wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()

df_yearly_crime = df.copy()

plt.figure(figsize=[20,10])

#In order to reduce processing time, we used a resampling method by month for the number of crimes. 

#The resampling method in pandas is similar to the groupby method for a certain time span.

df.resample('M').size().plot(legend=False) # resampling time series by month

plt.xlabel('No of  years')

plt.ylabel('Months')

plt.title('No of crime per month from 2012-2017')

plt.show()

crime_count_date = df.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=df.index.date, fill_value=0 )

# crime_count_date.head()

crime_count_date.index = pd.DatetimeIndex(crime_count_date.index)

plt = crime_count_date.rolling(365).sum().plot(figsize=[10,20], subplots=True, layout=(-1, 3),  sharex=False, sharey=False  )










