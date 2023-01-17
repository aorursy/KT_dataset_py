# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# import and loading

df = pd.read_csv("../input/StudentsPerformance.csv")
# data cleaning

df.rename(columns={"race/ethnicity": "ethnicity", 

"parental level of education":"parent_education",

"test preparation course":"preparation",

"math score":"m_score",

"reading score": "r_score",

"writing score": "w_score"}, inplace = True)



# feature engineering on the data to visualize and solve the dataset more accurately

df['total_score'] = (df['m_score'] + df['r_score'] + df['w_score'])/3

df.head()
df.isnull().sum()
# start data exploration

df.corr()
df.describe()
df.groupby(by="ethnicity").size().plot.barh()

plt.show()
df.groupby(by="gender").size().plot.bar()

plt.show()
df.groupby("ethnicity").gender.value_counts().plot.bar()

# non mi piace, vorrei farlo come quello sotto ma mi sto confondendo!

plt.show()
f_filter = df["gender"] == "female"

females = df[f_filter].groupby(["ethnicity"]).size()

females = females.reset_index(name="female")



m_filter = df["gender"] == "male"

males = df[m_filter].groupby(["ethnicity"]).size()

males = males.reset_index(name="male")



genders_df = males.join(females,rsuffix='_drop').drop(columns="ethnicity_drop").set_index("ethnicity")

# grafico che voglio ma fa schifo ottenuto così

genders_df.plot.bar()

plt.xticks(rotation=0)

plt.show()
# visualizing the differnt parental education levels



df['parent_education'].value_counts()

df['parent_education'].value_counts().plot.bar()

plt.title('Comparison of Parental Education')

plt.xlabel('Titolo di studio')

plt.ylabel('Numero')

plt.xticks(rotation=45)



plt.show()
# Come varia la distribuzione dei voti al variare del livello di educazione dei genitori?"

sns.set(rc={'figure.figsize':(20,7)})

fig, axs = plt.subplots(ncols=3)



sns.barplot(x = "parent_education", y = "w_score",  data = df, ax=axs[0])

sns.barplot(x = "parent_education", y = "r_score",  data = df, ax=axs[1])

sns.barplot(x = "parent_education", y = "m_score",  data = df, ax=axs[2])

for ax in axs:

    ax.tick_params(labelrotation=45)

    ax.tick_params(labelsize=12)

plt.show()

# Come varia tra i gruppi la distribuzione dei voti?

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.barplot(x = "ethnicity", y = "w_score",  data = df, ax=axs[0])

sns.barplot(x = "ethnicity", y = "r_score",  data = df, ax=axs[1])

sns.barplot(x = "ethnicity", y = "m_score",  data = df, ax=axs[2])

for ax in axs:

    ax.tick_params(labelrotation=45,labelsize=12)

plt.show()
# Come varia tra i generi la distribuzione dei voti?

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.barplot(x = "gender", y = "w_score",  data = df, ax=axs[0])

sns.barplot(x = "gender", y = "r_score",  data = df, ax=axs[1])

sns.barplot(x = "gender", y = "m_score",  data = df, ax=axs[2])



plt.show()
# Come influisce il tipo di pasto sulla distribuzione dei voti?

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.violinplot(x = "lunch", y = "w_score",  data = df, ax=axs[0])

sns.violinplot(x = "lunch", y = "r_score",  data = df, ax=axs[1])

sns.violinplot(x = "lunch", y = "m_score",  data = df, ax=axs[2])



plt.show()
# Come influisce il tipo di pasto sulla distribuzione dei voti?

sns.boxplot(x = "ethnicity", y = "total_score",  data = df, hue="lunch")



plt.show()


df.groupby("lunch")["ethnicity"].value_counts()

# serve calcolare i rapporti tra standard e free per ogni gruppo, per vedere se i pasti gratis prevalgono in percentuale in qualche gruppo
df.groupby("lunch")["parent_education"].value_counts()
# Data to plot

labels = 'group A', 'group B', 'group C', 'group D','group E'

sizes = df.groupby('ethnicity')['r_score'].mean().values

explode = (0.1, 0, 0, 0,0)  # explode 1st slice

 

# Plot

#plt.pie(sizes, explode=explode, labels=labels, colors=sns.color_palette("Set3"),

#autopct='%1.1f%%', shadow=True, startangle=140)

#plt.title('Reading Score for Every Ethnicity Mean')

#plt.axis('equal')

#plt.show()



# fare per le altre due materie
# setting a passing mark for the students to pass on the three subjects individually

passmarks = 40



# creating a new column pass_math, this column will tell us whether the students are pass or fail

df['pass_math'] = np.where(df['m_score']< passmarks, 'Fail', 'Pass')

df['pass_reading'] = np.where(df['r_score']< passmarks, 'Fail', 'Pass')

df['pass_writing'] = np.where(df['w_score']< passmarks, 'Fail', 'Pass')



# checking which student is fail overall



df['status'] = df.apply(lambda x : 'Fail' if x['pass_math'] == 'Fail' or 

                           x['pass_reading'] == 'Fail' or x['pass_writing'] == 'Fail'

                           else 'pass', axis = 1)

# Assigning grades to the grades according to the following criteria :

# A: 90-100

# B: 80-89

# C: 70-79

# D: 60-69

# F: 0-59



def getgrade(total_score, status):

  if status == 'Fail':

    return 'F'

  if(total_score >= 90):

    return 'A'

  if(total_score >= 80):

    return 'B'

  if(total_score >= 70):

    return 'C'

  if(total_score >= 60):

    return 'D'

  else :

    return 'E'



df['grades'] = df.apply(lambda x: getgrade(x['total_score'], x['status']), axis = 1 )



axs = df['grades'].value_counts().plot.bar()

axs.tick_params(rotation = 0)

axs.set_xlabel('Voti',size=16)

plt.show()
plt.figure(figsize=(9,6))

sns.countplot(x = "preparation", hue="gender", data = df)



plt.show()
fig, axs = plt.subplots(ncols=3)

colors = [ "amber", "windows blue","dusty purple","faded green", ]

color = sns.xkcd_palette(colors)



sns.barplot(x = "preparation", y = "w_score", hue="gender", data = df, ax=axs[0],palette=color)

sns.barplot(x = "preparation", y = "r_score",  hue="gender", data = df, ax=axs[1],palette=color[2:])

sns.barplot(x = "preparation", y = "m_score",  hue="gender", data = df, ax=axs[2])

plt.show()
sns.catplot(y="parent_education", hue="ethnicity", col="preparation", data=df, kind="count")

plt.show()