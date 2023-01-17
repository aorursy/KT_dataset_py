import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# gjt = google job tittle



gjt=pd.read_csv("/kaggle/input/google-job-skills/job_skills.csv")
gjt.head(10)

#it gives first 10 rows in data
gjt.columns

# shows that we have which columns
# describing the data set

gjt.describe()
gjt.dtypes

#it gives int,float,object etc.
# cheking the null values in the dataset



gjt.isnull().any()
#I'll check if there is any NaN

gjt.isnull().sum()
#calculates the number of rows and columns

print(gjt.shape)
gjt.Title.value_counts().head(20)
sns.catplot(y = "Category", kind = "count",

            palette = "colorblind", edgecolor = ".6",

            data = gjt)

plt.show()

#this graph gives categorical numbers that they position
gjt.Title.value_counts().head(20).plot.bar()

#it is giving about numbers of the tittle
sns.set(style="darkgrid")

sns.countplot(gjt['Company'])

plt.title('')



print(gjt['Company'].value_counts())
#it gives most 10 place in world. 

plt.title('Top 10 Location')

top_location=gjt['Location'].value_counts().sort_values(ascending=False).head(10)

top_location.plot(kind='bar')
# checking most popular top 20 types of job Titles 



plt.rcParams['figure.figsize'] = (19, 8)



color = plt.cm.PuRd(np.linspace(0, 1, 20))

gjt['Title'].value_counts().sort_values(ascending = False).head(20).plot.bar(color = color)

plt.title("Most Popular 20 Job Titles of Google", fontsize = 20)

plt.xlabel('Names of Job Titles', fontsize = 15)

plt.ylabel('count', fontsize = 15)

plt.show()