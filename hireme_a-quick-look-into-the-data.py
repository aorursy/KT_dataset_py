#standart stuff

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#load data to pandas frames

appls = pd.read_csv("../input/Wuzzuf_Applications_Sample.csv")

JobPosts = pd.read_csv("../input/Wuzzuf_Job_Posts_Sample.csv")
appls.info()
JobPosts.info()
appls.head(10)
#Convert it to datetime

appls['app_date'] = pd.to_datetime(appls['app_date'])

appls.index = appls['app_date']



# now we don't need app_data column

del appls["app_date"]

appls["count"] = 1
#Check what happen in May 2014?



appls["2014-05-03"]
#initital set-up for the seaborn for jupyther notebook

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})





ax = appls.resample('M').sum().plot()

ax.set_xlabel("")

ax.set_ylabel("Number of Job Applications")

#ax.set_title("Frequancy of ")

plt.show()
new_df = JobPosts



new_df['post_date'] = pd.to_datetime(new_df['post_date'])

new_df.index = new_df['post_date']



#now we don't need post_data column

del new_df["post_date"] 
ax = new_df["views"].resample('M').sum().plot()

ax.set_xlabel("")

ax.set_title("Number of Views per Month")

plt.show()
ax = new_df["num_vacancies"].resample('M').sum().plot()

ax.set_xlabel("")

#ax.set_ylabel("Number of Job Applications")

ax.set_title("Number of Vacancies per Month")

plt.show()
ax = new_df[["salary_maximum","salary_minimum"]].resample('M').mean().plot()

ax.set_xlabel("")

#ax.set_ylabel("Number of Job Applications")

ax.set_title("Summe of Min and Max Salaries per Month")

plt.show()


f, ax = plt.subplots(figsize=(12, 40))

#check the colours please!

sns.countplot(y="job_industry1", data=JobPosts,  palette="Set3", 

              order=JobPosts.job_industry1.value_counts().iloc[:].index )



plt.show()
#value counts for job_title column

JobPosts["job_title"].value_counts()
from wordcloud import WordCloud, STOPWORDS

words = ' '.join(JobPosts['job_title'])





# Generate a word cloud image

wordcloud = WordCloud().generate(words)





# lower max_font_size

wordcloud = WordCloud(max_font_size=40,background_color="white").generate(words)



plt.figure()



plt.imshow(wordcloud, interpolation="bilinear")



plt.axis("off")



plt.show()