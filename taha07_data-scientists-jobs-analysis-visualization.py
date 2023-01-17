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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
df = pd.read_csv("/kaggle/input/data-scientist-jobs/DataScientist.csv")

df.head()
df.info()
df.describe()
df.drop(["Unnamed: 0","index"],axis=1,inplace=True)

df.head()
hours_per_week = 40

weeks_per_year = 52



for i in range(df.shape[0]):

    salary_estimate = df.loc[i,"Salary Estimate"]

    salary_estimate = salary_estimate.replace("$", "")

    

    if "Per Hour" in salary_estimate:

        lower, upper = salary_estimate.split("-")

        upper, _ = upper.split("Per")

        upper= upper.strip()

        lower = int(lower) *hours_per_week*weeks_per_year*(1/1000)

        upper = int(upper) *hours_per_week*weeks_per_year*(1/1000)

        

    else:

        lower, upper = salary_estimate.split("-")

        lower = lower.replace("K", "")

        upper, _= upper.split("(")

        upper=upper.replace("K", "")

        upper = upper.strip()

    

        

    lower = int(lower)

    upper = int(upper)

    df.loc[i,"salary_estimate_lower_bound"] = lower

    df.loc[i,"salary_estimate_upper_bound"] = upper

  
for i in range(df.shape[0]):

    name = df.loc[i,"Company Name"]

    if "\n" in name:

        name,_ = name.split("\n")

    df.loc[i,"Company Name"] = name
df["Size"].value_counts()
for i in range(df.shape[0]):

    size = df.loc[i,"Size"]

    if "to" in  size:

        lower,upper = size.split("to")

        lower = lower.strip() 

        _, upper, _ = upper.split(" ")

        upper = upper.strip()

        lower = int(lower)

        upper = int(upper)

    elif "+" in size:

        lower,_ = size.split("+")

        lower = int(lower)

        upper = np.inf

    else:

        lower = np.nan

        upper = np.nan

    df.loc[i,"Minimum Size"] = lower

    df.loc[i,"Maximum Size"] = upper

    
df.head()
df.drop(["Salary Estimate","Size"],axis=1,inplace=True)

df.head()
df["Minimum Size"].fillna(0,inplace=True)

df["Maximum Size"].fillna(0,inplace=True)

df.isnull().sum()
plt.rcParams["figure.figsize"] = (12,9)

plt.style.use("classic")

color = plt.cm.PuRd(np.linspace(0,1,20))

df["Company Name"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)

plt.title("Top 20 Company with Highest number of Jobs in Data Science",fontsize=20)

plt.xlabel("Company Name",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

color = plt.cm.BuPu(np.linspace(0,1,20))

df["Job Title"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)

plt.title("Top 20 Data Science Job",fontsize=20)

plt.xlabel("Job Title",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

color = plt.cm.BuPu(np.linspace(0,1,20))

df["Location"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)

plt.title("Top 20 locations for Data Science Job",fontsize=20)

plt.xlabel("Locations",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

color = plt.cm.BuPu(np.linspace(0,1,20))

df["Headquarters"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)

plt.title("Top 20 Head Quarters of Data Science Job Holder Company",fontsize=20)

plt.xlabel("Head Quarters",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
color = plt.cm.PuRd(np.linspace(0,1,20))

df["Headquarters"].value_counts().sort_values(ascending=False).head(20).plot.pie(y="Headquarters",colors=color,autopct="%0.1f%%")

plt.title("Head Quarters according to Locations")

plt.axis("off")

plt.show()

plt.rcParams['figure.figsize'] = (12,9)

color = plt.cm.PuRd(np.linspace(0,1,20))

df["Founded"].value_counts().sort_values(ascending=False)[1:21].plot.bar(color=color)

plt.title("Number of Company, Founded in a Year",fontsize=20)

plt.xlabel("Foundation Year",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

color = plt.cm.hsv(np.linspace(0,1,20))

df["Type of ownership"].value_counts().sort_values(ascending=False).plot.bar(color=color)

plt.title("Types of Ownership",fontsize=20)

plt.xlabel("Ownership",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

color = plt.cm.hsv(np.linspace(0,1,20))

df["Sector"].value_counts().sort_values(ascending=False).plot.bar(color=color)

plt.title("Different types of Sectors in DataScience",fontsize=20)

plt.xlabel("Sectors",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
df[['Job Title','salary_estimate_upper_bound']].nlargest(10,"salary_estimate_upper_bound")
plt.rcParams['figure.figsize'] = (12,9)

df[['Job Title','salary_estimate_upper_bound']].nlargest(10,"salary_estimate_upper_bound").plot.bar(x="Job Title",y="salary_estimate_upper_bound",color='cyan')

plt.title("Top 10 Jobs according to Salary",fontsize=20)

plt.xlabel("Job Title",fontsize=15)

plt.ylabel("Salary in 'k' ",fontsize=15)

plt.show()
df["Maximum Size"].replace(np.inf,0,inplace=True)

df[['Company Name','Maximum Size']].nlargest(10,'Maximum Size')
plt.rcParams['figure.figsize'] = (12,9)

df[['Company Name','Maximum Size']].nlargest(30,'Maximum Size').plot.bar(x="Company Name",y="Maximum Size",color='cyan')

plt.title("Company's with Highest Number of Employees",fontsize=20)

plt.xlabel("Company Name",fontsize=15)

plt.ylabel("No. of Employees",fontsize=15)

plt.show()
from wordcloud import WordCloud

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow',

                      height =2000,

                      width = 2000

                     ).generate(str(df["Job Title"]))

plt.rcParams['figure.figsize'] = (12,12)

plt.axis("off")

plt.imshow(wordcloud)

plt.title("Most available Job Title")

plt.show()
from wordcloud import WordCloud

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'lightgreen',

                      height =2000,

                      width = 2000

                     ).generate(str(df["Company Name"]))

plt.rcParams['figure.figsize'] = (12,12)

plt.axis("off")

plt.imshow(wordcloud)

plt.title("Most available Company")

plt.show()
from wordcloud import WordCloud

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'lightpink',

                      height =2000,

                      width = 2000

                     ).generate(str(df["Headquarters"]))

plt.rcParams['figure.figsize'] = (12,12)

plt.axis("off")

plt.imshow(wordcloud)

plt.show()
hq = pd.read_csv("../input/country/Country.csv")

hq.head()
fig = px.choropleth(hq,   

    locationmode='country names',

    locations='Country',

    color="Country",

    featureidkey="Location",

    hover_name = "Country",

    labels=hq["No of HeadQuarters"],

    color_continuous_scale=px.colors.sequential.Plasma

)

fig.show()