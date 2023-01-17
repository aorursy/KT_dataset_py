# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import datetime as dt

import scipy.stats as stats

import seaborn as sns

from sklearn.linear_model import LinearRegression







df = pd.read_csv('../input/musicfeatures/data.csv', index_col=['filename'])





df.describe()

plt.style.use('seaborn-poster')
#Variable Appendix 



#filename - Filename as given in original marsyas dataset



#tempo - The speed at which a passage of music is played



#beats - Rythmic unit in music



#chroma_stft - Short Time Fourier Transform



#rmse - Root Mean Square Error



#spectral_centroid - Indicates where the "center of mass" of the spectrum is located.



#spectral_bandwidth - It is the Wavelength interval in which a radiated spectral quantity is not less than half its maximum value



#rolloff - Roll-off is the steepness of a transmission function with frequency



#zero_crossing_rate - The rate at which the signal changes from positive to negative or back



#mfcc1 - 20 - Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC.

# In the original dataset, the primary researcher said that, in particular, the first 5 coefficients provide the best genre classification performance



#Primary Categories of Features 



#Timbtral Texture - means and variances of spectral centroid, rolloff, flux, zerocrossings, spectral bandwidth 



#Rhythmic Context - tempo, beats. 



#Pitch Content - data not available in dataset
#I will take a look at descriptive statistics for each genre using Tempo, Beats, and Mfcc1 - 5
#I want to compare the data by label (genre) so I used the groupby function. The describe function gives me access to basic descriptive statistics of the dataframe.

#I will use this function for all variables in this section.



df.groupby('label').describe()['tempo']

#Reggae music has the highest tempo on average, but a fairly high standard deviation so it may be due to outliers shifting the mean upwards

#Disco has the lowest standard deviation by far, tempo seems to be relatively consistent. 

#Genres with higher overall tempos are reggae, classical, and metal

#Genres with lower overall tempos are country, pop, and jazz
df.groupby('label').describe()['beats']

#The differences between genres are far less prominent than tempo 

#Reggae has the highest beat count with a middling standard deviation. 

#Classical music again has the highest standard deviation, indicating that beats may not be a good indication of the genre due to the degree of variety

#Disco again has the lowest standard deviation, indicating a higher level of consistency between songs 

#The genres with the highest beat count are reggae, metal, and disco 

#The genres with the lowest beat count are country, jazz, and pop
df.groupby('label').describe()['mfcc1']

#There are vast differences in texture of Mel-frequency cepstral coefficient 1

#Disco, Hip Hop, Metal, and Pop have low standard deviations

#Classical music also has the highest Mel-frequency cepstral coefficient 1 by far, it also has the highest standard deviation

#Texture seems to be more distinctive 
df.groupby('label').describe()['mfcc2']
df.groupby('label').describe()['mfcc3']
df.groupby('label').describe()['mfcc4']
df.groupby('label').describe()['mfcc4']
df.groupby('label').describe()['mfcc5']
# I will be exploring Tempo, Beats, and Mfcc1 - 5 to see if there are any major outliers and to see if there are any differences in uniformity
#Boxplots give me access to the average distribution of a dataset, but also identifies outliers outside of that distribution.

#I again am looking specifically at genres, so the x axis is "label" and the y axis is "tempo"

#I have also increased the linewideth for better visibility

#I will use this method for each plot in this section.



sns.boxplot( x=df["label"], y=df["tempo"], linewidth=5)



#I was interested in finding outliers and seeing if there were any genres where outliers might be more common and if there were any where it was less common

#This boxplot shows that certain genres, specifically pop and hip hop have more outliers in general

#It also shows that genres such as blues, country, metal, and reggae don't have any outliers in tempo
sns.boxplot( x=df["label"], y=df["beats"], linewidth=5)



#Hip Hop has a lot of outliers, with 10, it is far higher than any other genre. 
sns.boxplot( x=df["label"], y=df["mfcc1"], linewidth=5)
sns.boxplot( x=df["label"], y=df["mfcc2"], linewidth=5)
sns.boxplot( x=df["label"], y=df["mfcc3"], linewidth=5)
sns.boxplot( x=df["label"], y=df["mfcc4"], linewidth=5)
sns.boxplot( x=df["label"], y=df["mfcc5"], linewidth=5)
#For this next section, Barplots seemed like the best fit since they are highly visible, and I don't need to see outliers.

#Having the mean and the average distribution will allow me to compare unique factors



sns.barplot(x=df["label"], y=df["tempo"])



#While tempo doesn't seem to be a particularly strong predictor variable, there may be significance in the differences between genres with statistical analysis
sns.barplot(x=df["label"], y=df["beats"])

#As stated in the first question, there doesn't seem to be too many meaningful differences between genres in beat count 
sns.barplot(x=df["label"], y=df["mfcc1"])

#Mfcc seems to be a much stronger predictor than the variables measuring Rhythmic content. Genres such as Classical, Jazz, and Reggae have distinctively low frequencies.

#Scores lower than -250hz are exclusively seeen in Classical, and in no other genre
sns.barplot(x=df["label"], y=df["mfcc2"])

#This measure also provides diverse scores, though maybe less predictive than Mfcc1

#Classical music is again occupying a unique space where no other genres show up, above about 120 hz
sns.barplot(x=df["label"], y=df["mfcc3"])

#This plot provides two genres that occupy unique spaces. Metal music is exclusively below -25 hz, while Pop music is the only genre consistently with a number over 0 hz.

sns.barplot(x=df["label"], y=df["mfcc4"])

#Metal is particularly distinctive, occupying any frequency over 50hz. Pop is also distinctive, with no values over 20hz
sns.barplot(x=df["label"], y=df["mfcc5"])

#Metal is again distinctive, with no other genre having frequencies below -7hz. Pop is also distinctive, with no other genre having frequencies above 7hz.
#I found a guide on how to have multiple lines occupying the same line graph. This allowed me to to plot all 5 mfcc at the same time

#Here are Mfcc 1 - 5 plotted together. You can see distinctive differences when viewed as whole. 

#For some reason, this entire project, I have not been able to get line graphs to show all my labels. So the next two graphs only have 5 of the 10 genres shown.



print(df["label"].unique())



ax = plt.gca()



df.plot(x='label', y='mfcc1', kind='line', ax=ax)

df.plot(x='label', y='mfcc2', kind='line', color='red', ax=ax)

df.plot(x='label', y='mfcc3', kind='line', color='yellow', ax=ax)

df.plot(x='label', y='mfcc4', kind='line', color='green', ax=ax)

df.plot(x='label', y='mfcc5', kind='line', color='orange', ax=ax)









plt.show()
#When using the same method as above, the differences between genres do exist, but they significantly less prominent. 

ax = plt.gca()



df.plot(x='label', y='tempo', kind='line', ax=ax)

df.plot(x='label', y='beats', kind='line', color='red', ax=ax)