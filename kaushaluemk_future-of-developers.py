import pandas as pd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")
data
#top 5 Country with most number of upcoming developers
data['Country'].value_counts().head(5)
#Which Gender will be dominating in this field
data['Gender'].value_counts().head(2).plot(kind='pie')
# Top 15 age group using stack overflow
data['Age'].value_counts().head(15).plot(kind='bar')
data['OpenSourcer'].value_counts()
data_india=data[data['Country']=='India']
data_usa=data[data['Country']=='United States']
data['OpenSourcer'].value_counts().plot(kind='pie')
data_india['OpenSourcer'].value_counts()
# ONLY 1562 developers from India , contribute to open source on a regular basis out 0f 9061
data['UndergradMajor'].value_counts().head(5).plot(kind='pie')
# students with a computer science background in their undergrad forms the majority 
data_india['EdLevel'].value_counts()
data_usa['EdLevel'].value_counts()
#It clearly shows that people in USA in their bachelors degree are more towards development and contributing towards open source