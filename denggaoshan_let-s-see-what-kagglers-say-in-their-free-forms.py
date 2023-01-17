# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Read each of the file

freeForm = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1")

# Data Quality Check

import missingno as msno

msno.matrix(df=freeForm, figsize=(20, 14), color=(0.42, 0.1, 0.05))
freeForm.count().sort_values(ascending=False).head(15)
gender_free = freeForm['GenderFreeForm'].apply(lambda x: str(x).lower()).copy()

gender_free.apply(lambda x : 'nan' in str(x)).value_counts()
labels = ['nan','not empty']

values = gender_free.apply(lambda x : 'nan' in str(x)).value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'How many people fill GenderFreeForm options for gender？',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot1')

#Drop nan and some useless info such as male, female , human

gender_free_copy = gender_free.copy()



gender_free_copy[gender_free_copy.apply(lambda x: x == 'male')] = 'nan'

gender_free_copy[gender_free_copy.apply(lambda x: x == 'female')] = 'nan'

gender_free_copy[gender_free_copy.apply(lambda x: 'human' in x)] = 'nan'



gender_free_copy = gender_free_copy[gender_free_copy != 'nan']
# Think he/she is a helicopter

gender_free_copy[gender_free_copy.apply(lambda x: 'helicopter' in x or 'apache' in x)] = 'Attack Helicopter'

gender_free_copy[gender_free_copy == 'Attack Helicopter'].count()
# just complain

complains = ["it doesn't matter",

             "i'm a dragon you racist fuck",

            "stop being sexist",

            "there are only two genders",

            "please stop this nonsense",

            "wtf???", "wtf",

            "there are only two genders",

            "there are only 2 genders... stop helping sexually fucked up minorities lobbies",

            "why the hell do we need so many genders?",

             "there are only two genders",

             "this question is crap. i'm male.",

             "there are two genders: male/female. i am sorry, but as i scientist i cannot change reality to make people feel good. i refuse to participate in surveys that deny science.",

            "gender is a social construct and we should be working towards abolishing it",

            "lol seriously",

            "i prefer not to say",

            "there are only two genders ",

            "irrelevant",

            ]



gender_free_copy[gender_free_copy.apply(lambda x: x in complains)] = 'Complaint'

gender_free_copy[gender_free_copy == 'Complaint'].count()
# It looks like a serious answer 

serious = ["trans female",

           "bisexual transgender non-conformist",

            ]



gender_free_copy[gender_free_copy.apply(lambda x: x in serious)] = 'Serious'

gender_free_copy.value_counts().index



# The following are some of the remaining strange answers
gender_free_copy[(gender_free_copy != 'Complaint') 

                 & (gender_free_copy != 'Attack Helicopter')

                & (gender_free_copy != 'Serious')] = 'other'



labels = gender_free_copy.value_counts().index

values = gender_free_copy.value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'What are the main contents of the gender fill？',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot1')

kaggle_motivation = freeForm['KaggleMotivationFreeForm'].apply(lambda x: str(x).lower()).copy()



labels = ['nan','not empty']

values = kaggle_motivation.apply(lambda x : 'nan' in str(x)).value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'How many people filled out their motives for Kaggle？',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot2')

kaggle_motivation_copy = kaggle_motivation.copy()

kaggle_motivation_copy = kaggle_motivation_copy[kaggle_motivation_copy != 'nan']

# How many people are for curiosity

key_words = ["curious","curiosity",

            "interesting","interest","enjoy",

             "fun","motivation","enthusiast",

             "hobby","passion"

             "love",

             "like",

            ]

for word in key_words:

    kaggle_motivation_copy[kaggle_motivation_copy.apply(lambda x: word in x)] = 'Curious'

kaggle_motivation_copy[kaggle_motivation_copy == 'Curious'].count()
# How many people are teacher requirements, courses

key_words = ["university",

             "college",

             "professor",

             "class",

             "school",

            ]

for word in key_words:

    kaggle_motivation_copy[kaggle_motivation_copy.apply(lambda x: word in x)] = 'Class'

kaggle_motivation_copy[kaggle_motivation_copy == 'Class'].count()
# How many people are for studying / courses

key_words = ["study","education"

             "studied",

             "studies",

             "learn",

             "know",

             "course",

            ]



for word in key_words:

    kaggle_motivation_copy[kaggle_motivation_copy.apply(lambda x: word in x)] = 'Study'



kaggle_motivation_copy[kaggle_motivation_copy == 'Study'].count()
# How many people are working/research or looking for work

key_words = ["work",

             "job",

             "career",

             "research",

             "industry",

            ]



for word in key_words:

    kaggle_motivation_copy[kaggle_motivation_copy.apply(lambda x: word in x)] = 'Work'

kaggle_motivation_copy[kaggle_motivation_copy == 'Work'].count()
# How many people are for competition, datasets, bonuses

key_words = ["competition","compete","challeng",

             "dataset","exploration",

             "money",

             "data","trend",

             "challenge","experience",

             "life",

            ]



for word in key_words:

    kaggle_motivation_copy[kaggle_motivation_copy.apply(lambda x: word in x)] = 'Competition'

kaggle_motivation_copy[kaggle_motivation_copy == 'Competition'].count()
kaggle_motivation_copy[(kaggle_motivation_copy != 'Competition') 

                & (kaggle_motivation_copy != 'Work')

                & (kaggle_motivation_copy != 'Curious')

                & (kaggle_motivation_copy != 'Class')

                & (kaggle_motivation_copy != 'Study')

                      ] = 'Other'



labels = kaggle_motivation_copy.value_counts().index

values = kaggle_motivation_copy.value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'The motivation to take Kaggle',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')

from wordcloud import WordCloud



# Read the whole text.

text = ""

for tmp in kaggle_motivation:

    text = text + tmp



text.replace('ml','machine learning')

text.replace('machine','machine learning')

text.replace('interested','interesting')



# lower max_font_size

wordcloud = WordCloud(max_words=80, max_font_size=40).generate(text)



plt.figure(figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
personal_challenge = freeForm['PersonalProjectsChallengeFreeForm'].apply(lambda x: str(x).lower()).copy()



labels = ['nan','not empty']

values = personal_challenge.apply(lambda x : 'nan' in str(x)).value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'How many people filled their PersonalProjectsChallengeFreeForm for Kaggle？',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')
personal_challenge_copy = personal_challenge.copy()

personal_challenge_copy = personal_challenge_copy[personal_challenge_copy != 'nan']
# How many people are working/research or looking for work

key_words = ["clean",

             "dirty",

             "quality",

             "missing",

             "mung",

             "preprocessing",

             "wrangl"

            ]



for word in key_words:

    personal_challenge_copy[personal_challenge_copy.apply(lambda x: word in x)] = 'Dirty Data'

personal_challenge_copy[personal_challenge_copy == 'Dirty Data'].count()



personal_challenge_copy[personal_challenge_copy=='Dirty Data'].count()
# How many people are working/research or looking for work

key_words = ["size",

             "small",

             "large",

             "big",

             "enough",

            ]



for word in key_words:

    personal_challenge_copy[personal_challenge_copy.apply(lambda x: word in x)] = 'Data Set Size'

personal_challenge_copy[personal_challenge_copy == 'Data Set Size'].count()
# How many people are working/research or looking for work

key_words = ["find",

             "authenti",

             "availab",

             "collect",

            ]



for word in key_words:

    personal_challenge_copy[personal_challenge_copy.apply(lambda x: word in x)] = 'Fiding Data'

personal_challenge_copy[personal_challenge_copy == 'Fiding Data'].count()
# How many people are working/research or looking for work

key_words = ["document",

            ]



for word in key_words:

    personal_challenge_copy[personal_challenge_copy.apply(lambda x: word in x)] = 'Lack Documentation'

personal_challenge_copy[personal_challenge_copy == 'Lack Documentation'].count()
# How many people are working/research or looking for work

key_words = ["none",

             "na",

             "-",

             "nothing",

            ]



for word in key_words:

    personal_challenge_copy[personal_challenge_copy.apply(lambda x: word in x)] = 'None'

personal_challenge_copy[personal_challenge_copy == 'None'].count()
# How many people are working/research or looking for work

key_words = ["data",

            ]



for word in key_words:

    personal_challenge_copy[personal_challenge_copy.apply(lambda x: word in x)] = 'Other About Data'

personal_challenge_copy[personal_challenge_copy == 'Other About Data'].count()
personal_challenge_copy[(personal_challenge_copy != 'Fiding Data') 

                & (personal_challenge_copy != 'Lack Documentation')

                & (personal_challenge_copy != 'Data Set Size')

                & (personal_challenge_copy != 'Dirty Data')

                & (personal_challenge_copy != 'None')      

                & (personal_challenge_copy != 'Other About Data')

                      ] = 'Other'



labels = personal_challenge_copy.value_counts().index

values = personal_challenge_copy.value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'What is your biggest challenge?',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')

from wordcloud import WordCloud, STOPWORDS



# Read the whole text.

text = ""

for tmp in personal_challenge:

    text = text + tmp



text.replace('data set','dataset')

text.replace('often','')



stopwords = set(STOPWORDS)

stop_words = ['one','want','often','usual','always','sometimes','much',

             'problem','make','finding','uaually','biggest','one','make','may','way',

             'also','use','lot','want','hard','often','take','sometime']



stopwords.intersection(stop_words)

    

# lower max_font_size

wordcloud = WordCloud(max_words=80,max_font_size=40,stopwords=stopwords).generate(text)



plt.figure(figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
interest_problem = freeForm['InterestingProblemFreeForm'].apply(lambda x: str(x).lower()).copy()



labels = ['nan','not empty']

values = interest_problem.apply(lambda x : 'nan' in str(x)).value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'How many people filled the InterestingProblemFreeForm？',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')
interest_problem_copy = interest_problem.copy()

interest_problem_copy = interest_problem_copy[interest_problem_copy != 'nan']

# How many people are for curiosity

tmp = interest_problem_copy.value_counts()[interest_problem_copy.value_counts() > 10]





labels = tmp.index

values = tmp



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'What is the most interesting problem for Kaggler?',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')

from wordcloud import WordCloud, STOPWORDS



# Read the whole text.

text = ""

for tmp in interest_problem_copy:

    text = text + tmp



text.replace('data set','dataset')

text.replace('often','')

text.replace('will','')

text.replace('ml','machine learning')



stopwords = set(STOPWORDS)

stop_words = ['one','want','often','usual','always','sometimes','much',

             'problem','make','finding','uaually','biggest','one','make','may','way',

             'also','use','lot','want','hard','often','take','sometime']



stopwords.intersection(stop_words)

    

# lower max_font_size

wordcloud = WordCloud(max_words=80,max_font_size=40,stopwords=stopwords).generate(text)



plt.figure(figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
impactful_algorithm = freeForm['ImpactfulAlgorithmFreeForm'].apply(lambda x: str(x).lower()).copy()



labels = ['nan','not empty']

values = interest_problem.apply(lambda x : 'nan' in str(x)).value_counts()



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'How many people filled the ImpactfulAlgorithmFreeForm？',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')
impactful_algorithm_copy = impactful_algorithm.copy()

impactful_algorithm_copy = impactful_algorithm_copy[impactful_algorithm_copy != 'nan']

impactful_algorithm_copy.value_counts()

# How many people are working/research or looking for work

key_words = ["random forest",

            ]



for word in key_words:

    impactful_algorithm_copy[impactful_algorithm_copy.apply(lambda x: word in x)] = 'Random Forest'

impactful_algorithm_copy[impactful_algorithm_copy == 'Random Forest'].count()
# How many people are working/research or looking for work

key_words = ["neural network",

             "rnn",

             "cnn",

            ]



for word in key_words:

    impactful_algorithm_copy[impactful_algorithm_copy.apply(lambda x: word in x)] = 'Neural Network'

impactful_algorithm_copy[impactful_algorithm_copy == 'Neural Network'].count()
# How many people are working/research or looking for work

key_words = ["regression",

            ]



for word in key_words:

    impactful_algorithm_copy[impactful_algorithm_copy.apply(lambda x: word in x)] = 'Regression'

impactful_algorithm_copy[impactful_algorithm_copy == 'Regression'].count()
impactful_algorithm_copy = impactful_algorithm_copy[impactful_algorithm_copy != 'nan']

# How many people are for curiosity

tmp = impactful_algorithm_copy.value_counts()[impactful_algorithm_copy.value_counts() > 20]





labels = tmp.index

values = tmp



fig = go.Pie(labels=labels, values=values)



layout= go.Layout(

    title= 'What is the most impactful algorithm for Kaggler?',

)



fig = go.Figure(data=[fig], layout=layout)

py.iplot(fig,filename='plot3')

from wordcloud import WordCloud, STOPWORDS



# Read the whole text.

text = ""

for tmp in impactful_algorithm_copy:

    text = text + tmp



text.replace('data set','dataset')

text.replace('often','')

text.replace('will','')

text.replace('ml','machine learning')



stopwords = set(STOPWORDS)

stop_words = ['one','want','often','usual','always','sometimes','much',

             'problem','make','finding','uaually','biggest','one','make','may','way',

             'also','use','lot','want','hard','often','take','sometime']



stopwords.intersection(stop_words)

    

# lower max_font_size

wordcloud = WordCloud(max_words=80,max_font_size=40,stopwords=stopwords).generate(text)



plt.figure(figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()