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
BRFSS_data = pd.read_csv("../input/2015.csv") # import 2015 data
columns = ['_STATE', 'PHYSHLTH', 'MENTHLTH', 'MEDCOST', 'BPHIGH4', 'ASTHMA3', 'EDUCA', 'INCOME2', 'EXERANY2', 'EXRACT11']

df = pd.DataFrame(BRFSS_data[columns])

df.describe()



# Some values to assign states from 1 to 50 are missing

# Specifically: 3, 7, 14, 43, 52

# Instead, states are listed from 1 to 56, including DC
df['_STATE'].describe()



df['_STATE'].value_counts().iloc[:3]

# Kansas had the most participants with 23,236

# Nebraska had 17,561 respondents
df['_STATE'].value_counts(ascending=True).iloc[:3]

# Guam had the least amount of responses at 1,669

# Nevada comes in second at 2,926
df['PHYSHLTH'].describe()



phys_health = df['PHYSHLTH'].value_counts()

print("Percentage:", phys_health.iloc[0] / df['PHYSHLTH'].count())

phys_health.iloc[:3]



# Out of 441,455 respondents, 274,143 said they have no days of being physically unhealthly

# That's about 62% in good physical health, which is good news
df['MENTHLTH'].describe()



ment_health = df['MENTHLTH'].value_counts()

print("Percentage:", ment_health.iloc[0] / df['MENTHLTH'].count())

ment_health.iloc[:3]

# Similarly, 68% of respondents say they are in a healthy mental state

# Specifically, 301,076 respondents
# MEDCOST queries if a participant's income limited them from seeing a doctor



df['MEDCOST'].describe()



medcost = df['MEDCOST'].value_counts()

medcost

# 2 = Money was not an issue

# For 396,748 respondents, their ability to see a doctor was not limited by money.



# df['MEDCOST'].count() # 441455



# Unfortunately, 43,514 were not able to schedule with a doctor within the last year due to money

# That's almost 10% of the respondents. Although it is a small percentage, it's not something nice to see

# medcost.iloc[1] / df['MEDCOST'].count() # 0.099
# BPHIGH4 collects data on who has been told by a doctor that they have high blood pressure

df['BPHIGH4'].describe()



bldprssr = df['BPHIGH4'].value_counts()

bldprssr

# bldprssr.iloc[0] # 254,318 = No

# 254,318 respondents have never been TOLD they have high blood pressure

# As a follow up, they were asked if they were tested for high blood pressure [BLOODCHO]

# BRFSS_data['BLOODCHO'].value_counts().iloc[0] # 382,302 = Yes

# 382,302 followed up and stated they were tested



# In addition, 3,271 were experiencing high blood pressure due to pregnancy
# ASTHMA3

# Participants were asked if they have ever been TOLD they had asthma



df['ASTHMA3'].describe()



asthma = df['ASTHMA3'].value_counts()

asthma

# 380,554 said they have never been told they had asthma
# EDUCA focuses on the highest completed level of education

df['EDUCA'].describe()



edu = df['EDUCA'].value_counts()

print(edu)

edu.plot.bar()

# Most responses have shown people have at least graduated high school (4-6)

# Unfortunately, 34,259 have never succeeded past high school (1-3)
# INCOME2 addresses the participants' household incomes

income = df['INCOME2']

print(income.describe())

# income.mode() # 8 = $75k+



# income.value_counts().plot.bar()

income.value_counts().sort_index().plot(kind='bar')

# income.value_counts()



# If we consider $50k to be financially stable for a household, this would mean 173,442 fulfill this.

# Unfortunately, the other 264,713 are sustaining an income below $50k

# This means 60% are possibly in a financial situation

# 264713 / income.count() # 0.604
# EXERANY2 finds out about the participants' physical involvement in activities/exercise

# Complete list of activities in [EXRACT11] on page 38 - 39



exer = df['EXERANY2']

# exer.describe()



exer.count() # 406,012

exer.value_counts() # 1 = Yes | 2 = No

# exer.value_counts().iloc[0] / 406012 # 0.7291

# About 73% participate in physical activities outside of work

# This should keep them in a healthy physical state, indicative of a previous statistic [PHYSHLTH]
# EXRACT11 lists the specific activities people prefer

activities = df['EXRACT11']

activities.describe()

activities.count() # 295,778



activities.mode() # 64 = Walking

# Walking seems to be a frequent response

# Perhaps because of its simplicity?



print("[Top 3]")

print(activities.value_counts().head(3)) # 37 = Running

# Because of the similar way they work, running is second most frequent



activities.value_counts().iloc[0:2].sum() / activities.count() # 0.612

# Running and walking seem to be preferred by 61% of respondents



print("\n[Bottom 3]")

print(activities.value_counts().tail(3))

# Interestingly, stream fishing (55), sledding/tobogganing (45), inline skating (27) are the least popular