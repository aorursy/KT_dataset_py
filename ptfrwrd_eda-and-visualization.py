import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load dataset

data = pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")

data.head()
data.info()
data.drop("Unnamed: 0", axis=1, inplace=True)
data.replace(to_replace = "-1", value = np.nan, inplace = True)

data.replace(to_replace = -1.0, value = np.nan, inplace = True)

data.replace(to_replace = -1, value = np.nan, inplace = True)

data.head()
data.isna().sum()
def clean_data(name):

    data[name] = data[name].str.replace(r'\r\n|\r|\n\d*[0-9](|.\d*[0-9]|)*$', '')

    data[name] = data[name].str.replace(r'\r\n|\r|\n', ' ')

    data[name] = data[name].str.replace(r"[$K*]", ' ')
clean_data("Company Name")

clean_data("Job Description")

clean_data("Salary Estimate")

data.head()
divided_data = data["Salary Estimate"].str.split("-", expand=True)

data['Minimum Salary'] = pd.to_numeric(divided_data[0].str.extract('(\d+)', expand=False)) * 1000

data["Maximum Salary"] = pd.to_numeric(divided_data[1].str.extract('(\d+)', expand=False)) * 1000

data.head()
data["Job Title"].value_counts()[:20]
data.describe()
top_companies = data["Company Name"].value_counts().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14,9))

rect1 = sns.barplot(x = top_companies.index, y = top_companies.values, palette="deep").set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title("Top 20 Company with Highest number of Jobs", fontweight="bold")
ratings_data = data['Rating'].value_counts()

fig, ax = plt.subplots(figsize=(14,9))

rect1 = sns.barplot(x = ratings_data.index, y = ratings_data.values, palette="deep").set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title("Ratings distribution", fontweight="bold")
top_industries = data["Industry"].value_counts().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14,9))

rect1 = sns.barplot(x = top_industries.index, y = top_industries.values, palette="deep").set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title("Top 20 Industries with Highest number of Jobs", fontweight="bold")
data_corr = {

    'Rating': data.Rating,

    'Location': data.Location,

    'Industry': data.Industry,

    'Founded': data.Founded,

    'Min Salary': data['Minimum Salary'],

    'Max Salary': data['Maximum Salary']

}



data_corr = pd.DataFrame.from_dict(data_corr)

data_corr['Location'] = data_corr['Location'].astype('category').cat.codes

data_corr['Industry'] = data_corr['Industry'].astype('category').cat.codes



corr = data_corr.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

corr
from wordcloud import WordCloud, ImageColorGenerator

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from PIL import Image

from nltk.corpus import stopwords



words_cloud = data["Job Description"].str.split("(").str[0].value_counts().keys()

wc1 = WordCloud(stopwords=stopwords.words("english"),scale=5,max_words=1000,colormap="rainbow",background_color="white").generate(" ".join(words_cloud))

plt.figure(figsize=(20,14))

plt.imshow(wc1,interpolation="bilinear")

plt.axis("off")

plt.title("Key Words in Job Descriptions",color='black',fontsize=20)

plt.show()
gr = sns.catplot(x = 'Minimum Salary', y = 'Industry', kind = "box", data = data, order = data.Industry.value_counts().iloc[:20].index)

gr.fig.set_size_inches(30, 10)
gr = sns.catplot(x = 'Maximum Salary', y = 'Industry', kind = "box", data = data, order = data.Industry.value_counts().iloc[:20].index)

gr.fig.set_size_inches(30, 10)
gr = sns.catplot(x = 'Rating', y = 'Industry', kind = "box", data = data, order = data.Industry.value_counts().iloc[:20].index)

gr.fig.set_size_inches(30, 10)