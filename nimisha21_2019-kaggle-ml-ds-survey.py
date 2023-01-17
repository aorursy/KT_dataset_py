# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



os.listdir("../input")
#Importing the 2019 Dataset

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df_2019.columns = df_2019.iloc[0]

df_2019=df_2019.drop([0])
df_2019.head(2)
import matplotlib.pyplot as plt

nationality=df_2019['In which country do you currently reside?'].value_counts()

print(nationality)


plt.figure(figsize=(20,60))

ax=df_2019['In which country do you currently reside?'].value_counts()[:59].plot.barh(width=0.6,color=sns.color_palette('Set2',25))

plt.gca().invert_yaxis()

plt.title('Country')

plt.show()
df_2019.iloc[1:, :]['What is your age (# years)?'].value_counts().plot(kind='bar', figsize=(20, 8))

plt.xlabel('Age Groups')

plt.ylabel('Count')

plt.title('Age Groups Distribution')

plt.show()



df_2019.iloc[1:, :]['What is your gender? - Selected Choice'].value_counts().plot(kind='bar', figsize=(20, 8), color=['blue', 'red', 'green', 'cyan'])

plt.xlabel('Sex')

plt.ylabel('Count')

plt.title('Sex s Distribution')

plt.show()
df_2019.iloc[1:, :]['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().plot(kind='bar', figsize=(20, 8))
df_2019.iloc[1:, :]['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?'].value_counts().plot(kind='bar', figsize=(20, 8))
df_2019.iloc[1:, :]['What is your current yearly compensation (approximate $USD)?'].value_counts().plot(kind='bar', figsize=(20, 8))
df_2019.iloc[1:, :]['What is the size of the company where you are employed?'].value_counts().plot(kind='bar', figsize=(20, 8))
df_2019.iloc[1:, :]['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().plot(kind='bar', figsize=(20, 8))


df_2019.iloc[1:, :]['Does your current employer incorporate machine learning methods into their business?'].value_counts().plot(kind='bar', figsize=(20, 8))
df_2019.iloc[1:, :]['For how many years have you used machine learning methods?'].value_counts().plot(kind='bar', figsize=(20, 8)) 

    
question=pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

text=pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
question
text
col=question.columns

for i in range(question.shape[1]):

    

    print(i,question[col[i]])