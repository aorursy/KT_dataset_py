# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import seaborn as sns

from sklearn import preprocessing

import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt  

matplotlib.style.use('ggplot')

%matplotlib inline

import math



input_df = pd.read_csv("../input/appendix.csv",sep=',',parse_dates=['Launch Date'])

input_df['year'] = input_df['Launch Date'].dt.year

print(input_df.columns)



# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(input_df['Course Title']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(input_df['Course Subject']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
sns.factorplot('Institution',data=input_df,kind='count')
sns.factorplot('year',data=input_df,hue='Institution',kind='count')
no_of_participents = input_df[['Institution',"Participants (Course Content Accessed)"]].groupby('Institution').sum()

no_of_participents = no_of_participents.reset_index()



print(no_of_participents)



sns.factorplot(x='Institution',y='Participants (Course Content Accessed)',kind='bar',data=no_of_participents)


no_of_participents = input_df[['Institution',"Participants (Course Content Accessed)",'year']].groupby(['Institution','year']).sum()

no_of_participents = no_of_participents.reset_index()



print(no_of_participents)

sns.barplot(x='year',y='Participants (Course Content Accessed)',hue='Institution',data=no_of_participents)
participants_stats = input_df[['Participants (Course Content Accessed)','Audited (> 50% Course Content Accessed)','Certified']]

participants_stats.columns = ['total','50%Accessed','Certified']

no_of_participents = input_df[['Institution',"Participants (Course Content Accessed)","Audited (> 50% Course Content Accessed)","Certified",'year']].groupby(['Institution','year']).sum()

no_of_participents = no_of_participents.reset_index()



print(no_of_participents)





no_of_participents.plot(x='year',kind='bar',alpha=0.5,figsize=(9,5))