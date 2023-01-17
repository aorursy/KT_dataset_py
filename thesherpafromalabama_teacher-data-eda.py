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
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/online-teacher-languages-and-rates/data/anonymousteacherdata - Sheet1.csv')
df
# Let's do a little cleaning (note, you can come back to this stage after doing some EDA, cause it's easier to see stuff now)

# Is_Tutor -> Teacher_Type
teacher_dict = {0:"Teacher",1:"Tutor"}
df = df.replace({"Is_Tutor": teacher_dict})
df = df.rename({"Is_Tutor":"Teacher_Type"}, axis = 'columns')

# Rename 1st teahcing lang column
df = df.rename({"1stTeaching Language":"1stTeachingLanguage"}, axis = 'columns')
df
# Let's get how many spoken languages each perosn speaks and append back onto the other df
spoken_languages = df.filter(regex=("TeachingLanguage|AlsoSpeaks"))
spoken_languages
langlist2 = pd.unique(spoken_languages.values.ravel('K')).tolist()
langlist2 = [x for x in langlist2 if str(x) != 'nan']
# How many languages does each teacher speak?

# First iterate over columns
for (columnName, columnData) in spoken_languages.iteritems():
    spoken_languages[columnName] = spoken_languages[columnName].apply(lambda x: True if x in langlist2 else False)
spoken_languages

#Sum up all True's for each row
spoken_languages['Num_Spoken_Languages'] = (spoken_languages == True).sum(axis=1)

# Append back on to original df
df = pd.concat([df,spoken_languages['Num_Spoken_Languages']], axis = 1)
df
# First language value counts (English is the winner, no suprise there)
first_lang = df['1stTeachingLanguage'].value_counts()
first_lang
# Filter < 10
more_than_ten = first_lang[first_lang >10]
# Filter > 10 to be fair and all
less_than_ten = first_lang[first_lang <10]

# Setup subplots object
fig, ax =plt.subplots(1,2, figsize=(15,15))
# Set plots for each position
ax1 = sns.barplot(more_than_ten, more_than_ten.index, ax = ax[0])
ax2 = sns.barplot(less_than_ten, less_than_ten.index, ax = ax[1])
fig.tight_layout()
fig.show()
# Just get the two interesting columns
min_price_df = df[['1stTeachingLanguage','Min_Price']]

# groupby what you want, then run numeric function on everything else
min_price_df = min_price_df.groupby(['1stTeachingLanguage']).mean().sort_values('Min_Price')

# Plot these values
# Setup subplots object
fig, ax =plt.subplots(1, figsize=(15,15))
# Set plots for each position
ax = sns.barplot(x = min_price_df.index, y = "Min_Price", data = min_price_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment = 'right')
fig.tight_layout()
fig.show()
# Filter < 10 and grab top 9
more_than_ten = first_lang[first_lang >10]
more_than_ten.iloc[:9].index.tolist()

top_nine_lang = df[df['1stTeachingLanguage'].isin(more_than_ten.iloc[:9].index.tolist())]
top_nine_lang
# Plot these values
# Setup subplots object
fig, ax =plt.subplots(1, figsize=(15,15))
# Set plots for each position
ax = sns.violinplot(x = '1stTeachingLanguage', y = "Min_Price", data = top_nine_lang, 
                    hue = 'Teacher_Type', inner = 'stick', split=True)
# ax = sns.swarmplot(x = '1stTeaching Language', y = "Min_Price", data = top_nine_lang, color = '0.25')
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment = 'right')
fig.tight_layout()
fig.show()
# Arabic has negative values???
df["Min_Price"].min()
# Let's learn more about teachers that speak multiple languages
df['Num_Spoken_Languages'].value_counts()
# So every teacher claims to speak at least 3 languages huh????