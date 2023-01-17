# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
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