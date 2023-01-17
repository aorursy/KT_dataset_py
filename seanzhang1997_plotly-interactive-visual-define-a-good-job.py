import pandas as pd

import numpy as numpy

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")
df.head(3)
#df["Size"].unique()                                # 9 sizes with Unknow and -1.

#len(df["Type of ownership"].unique())              # 16 categories

#len(df["Industry"].unique())                       # 89 industry categories

#len(df["Sector"].unique())                         # 25 sectors

#len(df["Location"].unique())                       # 253 locations

#df.shape[0]                                        # 2253 jobs

#len(df["Salary Estimate"].unique())                # 90 unique values for salary estimate
df.drop(df[df["Salary Estimate"]=='-1'].index, axis=0, inplace=True)

len(df["Salary Estimate"].unique())
# Since all the salary data are estimated by glass doors and they are all in the same range

# We can use average salary as indicator.

# We have to use regular expression here:

import re

def get_avg_salary(salary):

    salary_list = re.findall(r"\$(.+?)K",salary)

    salary_list = [int(i) for i in salary_list]

    return sum(salary_list)/2
df["Avg Salary"] = df["Salary Estimate"].apply(lambda x : get_avg_salary(x))
df["Avg Salary"].median() # 69. Pretty close to what I researched online.

df["Avg Salary"].mean() # 72. A bit high according to indeed and monster
# Good Job Indicator = Salary * 0.8 + Rating * 0.2

drop_cols = ["Unnamed: 0","Salary Estimate", "Headquarters", "Size","Founded","Revenue","Competitors", "Easy Apply"]

df.drop(drop_cols, axis=1, inplace=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[["Rating Scaled","Avg Salary Scaled"]] = pd.DataFrame(scaler.fit_transform(df[["Rating","Avg Salary"]]))
df["Good Job Indicator"] = df["Rating Scaled"]*20 + df["Avg Salary Scaled"]*80

df.sort_values(by="Good Job Indicator", ascending=False, inplace=True)

df.head() # Best 5 jobs. This result make sense to me.

# For jobs with lowest salary section, their indicator is 0, which does not make sense.
df.drop(df[df["Good Job Indicator"].isnull()].index, axis=0, inplace=True)

import plotly.graph_objects as go

fig = go.Figure(data=[go.Histogram(x=df["Good Job Indicator"])])

fig.show()

# Lets define jobs scoring more than 60 as great jobs and check their company names, location, job description and industry.
good_job_df = df[df["Good Job Indicator"]>=60]

good_job_df.shape #284 out of 2253 jobs.

city_rank = good_job_df.groupby(by="Location").count().reset_index().sort_values("Job Title", ascending=False)[0:20][["Location", "Job Title"]]

city_rank.rename(columns={"Job Title":"Num of Jobs"},inplace=True)
import plotly.express as px

fig = px.bar(city_rank, x='Location', y='Num of Jobs')

fig.show()
industry_rank = good_job_df.groupby(by="Industry").count().reset_index().sort_values("Job Title", ascending=False)[0:20][["Industry", "Job Title"]]

industry_rank.rename(columns={"Job Title":"Num of Jobs"},inplace=True)

# fig = px.bar(industry_rank, x='Industry', y='Num of Jobs',width=1000, height=500)

# fig.show()

fig = px.pie(industry_rank, values='Num of Jobs', names='Industry', title='Good Jobs Industry Distribution',width=1200, height=600)

fig.show()
from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords

job_description_all = " ".join(str(x) for x in df["Job Description"]).lower()

job_description_good = " ".join(str(x) for x in good_job_df["Job Description"]).lower()

#Getting rid of stopwords

STOPWORDS = stopwords.words('english')

new_stopwords = ["data","analysis","analytics","analyst","ability","work","opportunity","knowledge","experience",\

    "customer", "team","develop","provide","report","system","including","support","use","service","company","reporting","understand","requirement"]

STOPWORDS.extend(new_stopwords)

STOPWORDS = set(STOPWORDS)



job_description_all = [word for word in job_description_all.split() if word not in STOPWORDS]

job_description_good = [word for word in job_description_good.split() if word not in STOPWORDS]

job_description_all = " ".join(word for word in job_description_all)

job_description_good = " ".join(word for word in job_description_good)





wordcloud1 = WordCloud(stopwords=STOPWORDS, background_color="white",width=2000, height=800,max_font_size=100).generate(job_description_all)

wordcloud2 = WordCloud(stopwords=STOPWORDS, background_color="white",width=2000, height=800,max_font_size=100).generate(job_description_good)
plt.figure(figsize=(20,15), facecolor='k')

plt.imshow(wordcloud1)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
plt.figure(figsize=(20,15), facecolor='k')

plt.imshow(wordcloud2)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()