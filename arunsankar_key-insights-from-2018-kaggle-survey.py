from IPython.display import Image
Image(filename="../input/kaggle/kaggle-logo-transparent-300.png")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import folium

from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict

pd.set_option('display.max_colwidth', -1)

import os
print(os.listdir("../input/kaggle-survey-2018"))
schema = pd.read_csv('../input/kaggle-survey-2018/SurveySchema.csv', low_memory=False)
freeform = pd.read_csv('../input/kaggle-survey-2018/freeFormResponses.csv', low_memory=False)
multiple = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory=False)

print('Schema data: \nRows: {}\nCols: {}'.format(schema.shape[0],schema.shape[1]))
print(schema.columns)

print('\Free form responses data: \nRows: {}\nCols: {}'.format(freeform.shape[0],freeform.shape[1]))
print(freeform.columns)

print('\nMultiple choice responses data: \nRows: {}\nCols: {}'.format(multiple.shape[0],multiple.shape[1]))
print(multiple.columns)

free = freeform[1:]
freeform_df = free.copy()

responses = multiple[1:]
responses_df = responses.copy()
responses_df['Time from Start to Finish (seconds)'] = responses_df['Time from Start to Finish (seconds)'].astype('float')
responses_df['Time from Start to Finish (minutes)'] = responses_df['Time from Start to Finish (seconds)'] / 60

temp1 = responses_df['Q1'].value_counts().reset_index()
temp3 = responses_df['Q1'].value_counts(normalize=True) * 100
temp3 = temp3.reset_index()
temp2 = responses_df.groupby(['Q1'])['Time from Start to Finish (seconds)'].mean().reset_index()
temp = pd.merge(temp1, temp2, how='inner', left_on='index', right_on='Q1')
temp = temp[['index','Q1_x','Time from Start to Finish (seconds)']]
temp = pd.merge(temp, temp3, how='inner', on='index')
temp.columns = ['Gender','Number of respondents','Time to finish the survey in seconds','% Number of respondents']

f, ax = plt.subplots(figsize=(10, 2))
sns.barplot(x="% Number of respondents", y="Gender", data=temp, label="% Number of respondents", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width() + 3,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="green")
ax.set_ylabel('Gender', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] % of respondents across gender', size=12, color="green")
plt.show()
def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, col, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df[col].dropna():
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df
temp = generate_ngrams(freeform_df,'Q1_OTHER_TEXT',1,10)

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="wordcount", y="word", data=temp, label="wordcount", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width() + .2,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Count of word', size=10, color="green")
ax.set_ylabel('Word', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] Count of words in self described answers for gender', size=12, color="green")
plt.show()
freeform_df[freeform_df['Q1_OTHER_TEXT'].str.len()>50]['Q1_OTHER_TEXT']
fig, ax = plt.subplots(figsize=(15,3))
sns.boxplot(x="Time from Start to Finish (minutes)", y="Q1", data=responses_df[responses_df['Time from Start to Finish (minutes)']<responses_df['Time from Start to Finish (minutes)'].quantile(.8)], ax=ax, palette=sns.color_palette("RdYlGn", 4))
ax.set_xlabel('Time from Start to Finish (minutes)', size=10, color="#0D47A1")
ax.set_ylabel('Gender', size=10, color="#0D47A1")
ax.set_title('[Box Plot] Time taken by each gender to fill the survey', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
temp = responses_df['Q2'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="Q2", y="index", data=temp, label="index", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width()+1,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="green")
ax.set_ylabel('Age group', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] % of respondents across age groups', size=12, color="green")
plt.show()
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(x="Time from Start to Finish (minutes)", y="Q2", data=responses_df[responses_df['Time from Start to Finish (minutes)']<responses_df['Time from Start to Finish (minutes)'].quantile(.8)], ax=ax, palette=sns.color_palette("RdYlGn", 12), order=np.sort(responses_df['Q2'].unique()))
ax.set_xlabel('Time from Start to Finish (minutes)', size=10, color="#0D47A1")
ax.set_ylabel('Age group', size=10, color="#0D47A1")
ax.set_title('[Box Plot] Time taken by each age group to fill the survey', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
temp = responses_df['Q3'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Q3", y="index", data=temp.head(20), label="index", color="aqua")

for p in ax.patches:
    ax.text(p.get_width()+.6,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="darkcyan")
ax.set_ylabel('Country', size=10, color="darkcyan")
ax.set_title('[Horizontal Bar Plot] % of respondents across countries', size=12, color="darkcyan")
plt.show()
temp['index'] = temp['index'].replace("United States of America", "United States")
temp['index'] = temp['index'].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
temp['index'] = temp['index'].replace("Viet Nam", "Vietnam")
temp['index'] = temp['index'].replace("Republic of Korea", "North Korea")
temp['index'] = temp['index'].replace("Iran, Islamic Republic of...", "Iran")

country_geo = os.path.join("../input/worldcountries1/", 'world-countries.json')

m = folium.Map(location=[20, 0], zoom_start=1.5)
m = folium.Map(location=[48.85, 2.35], tiles="Mapbox Bright", zoom_start=1.5)

m.choropleth(
    geo_data=country_geo,
    name='Choropleth',
    data=temp,
    columns=['index', 'Q3'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='% of respondents'
)
folium.LayerControl().add_to(m)
m
temp = responses_df['Q4'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10,3))
sns.barplot(x="Q4", y="index", data=temp, label="index", color="silver")

for p in ax.patches:
    ax.text(p.get_width()+1.5,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="black")
ax.set_ylabel('Education', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] % of respondents by education', size=12, color="black")
plt.show()
temp = responses_df['Q5'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10,6))
sns.barplot(x="Q5", y="index", data=temp, label="index", color="peachpuff")

for p in ax.patches:
    ax.text(p.get_width()+1.5,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="black")
ax.set_ylabel('Undergraduate major', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] % of respondents by undergraduate major', size=12, color="black")
plt.show()
temp = responses_df['Q6'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10,8))
sns.barplot(x="Q6", y="index", data=temp, label="index", color="mediumspringgreen")

for p in ax.patches:
    ax.text(p.get_width()+.75,
            p.get_y() + (p.get_height()/2) + .05,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="black")
ax.set_ylabel('Job', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] % of respondents by job', size=12, color="black")
plt.show()
temp = generate_ngrams(freeform_df,'Q6_OTHER_TEXT',1,10)

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="wordcount", y="word", data=temp, label="wordcount", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width() + 4,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Count of word', size=10, color="green")
ax.set_ylabel('Word', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] Count of words in self described answers for role', size=12, color="green")
plt.show()
temp = responses_df['Q7'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10,8))
sns.barplot(x="Q7", y="index", data=temp, label="index", color="mediumspringgreen")

for p in ax.patches:
    ax.text(p.get_width()+.75,
            p.get_y() + (p.get_height()/2) + .05,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="black")
ax.set_ylabel('Industry', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] % of respondents by industry', size=12, color="black")
plt.show()
temp = generate_ngrams(freeform_df,'Q7_OTHER_TEXT',1,10)

f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="wordcount", y="word", data=temp, label="wordcount", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width() + 2,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Count of word', size=10, color="green")
ax.set_ylabel('Word', size=10, color="green")
ax.set_title('[Horizontal Bar Plot] Count of words in self described answers for industry', size=12, color="green")
plt.show()
temp = responses_df['Q8'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10,4))
sns.barplot(x="Q8", y="index", data=temp, label="index", color="mediumspringgreen")

for p in ax.patches:
    ax.text(p.get_width()+.75,
            p.get_y() + (p.get_height()/2) + .05,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="black")
ax.set_ylabel('Years of experience in current role', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] % of respondents by current experience', size=12, color="black")
plt.show()
temp = responses_df['Q9'].value_counts(normalize=True) * 100
temp = temp.reset_index()

f, ax = plt.subplots(figsize=(10,8))
sns.barplot(x="Q9", y="index", data=temp, label="index", color="mediumspringgreen")

for p in ax.patches:
    ax.text(p.get_width()+.75,
            p.get_y() + (p.get_height()/2) + .05,
            '{:1.1f}%'.format(p.get_width()),
            ha="center")

ax.set_xlabel('% of respondents', size=10, color="black")
ax.set_ylabel('Yearly compensation in USD', size=10, color="black")
ax.set_title('[Horizontal Bar Plot] % of respondents by current compensation', size=12, color="black")
plt.show()