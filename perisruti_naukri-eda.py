import pandas as pd

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)
naukri_df=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv',parse_dates=[1])
naukri_df.head()
naukri_df.info()
naukri_df.shape
naukri_df.nunique()
#column-wise null count

naukri_df.isnull().sum()
round((100*naukri_df.isnull().sum()/naukri_df.shape[0]),2)
plt.figure(figsize=(15,4))

colour=['#000099','#ffff00']

sns.heatmap(naukri_df[naukri_df.columns].isnull(),cmap=sns.color_palette(colour))
cols=["Job Title",  "Job Salary","Job Experience Required","Key Skills","Role Category","Location",                   

      "Functional Area","Industry", "Role"]
naukri_df['Job Title'].value_counts()[0:20]
naukri_df["Job Title"].value_counts(normalize=True)[0:20].plot.barh()

plt.show()
sns.barplot(x=naukri_df['Job Title'].value_counts()[0:20],y=naukri_df['Job Title'].value_counts()[0:20].index)
df=naukri_df[naukri_df["Job Title"]=="Job Description"]

df.head()
naukri_df["Job Salary"].value_counts()[0:20]
naukri_df["Job Salary"]= naukri_df["Job Salary"].replace([' Not disclosed ','Not Disclosed by Recruiter', ' Not Disclosed by Recruiter', ' Not Disclosed by Recruiter '], "Not_disclosed")
naukri_df["Job Salary"].value_counts()[0:20]
sns.barplot(x=naukri_df['Job Salary'].value_counts()[0:20],y=naukri_df['Job Salary'].value_counts()[0:20].index)
naukri_df["Job Experience Required"].value_counts()[0:20]
sns.barplot(x=naukri_df['Job Experience Required'].value_counts()[0:20],y=naukri_df['Job Experience Required'].value_counts()[0:20].index)
naukri_df["Job Experience Required"].value_counts(normalize=True)[0:20].plot.barh()

plt.show()
naukri_df["Location"].nunique()
naukri_df["Location"].value_counts()[0:20]
naukri_df["Location"]= naukri_df["Location"].replace(' Bengaluru', "Bengaluru")
naukri_df["Location"]= naukri_df["Location"].replace(' Hyderabad', "Hyderabad")
naukri_df["Location"]= naukri_df["Location"].replace(' Pune', "Pune")
naukri_df["Location"].value_counts()[0:20]
sns.barplot(x=naukri_df['Location'].value_counts()[0:20],y=naukri_df['Location'].value_counts()[0:20].index,palette="ch:2.5,-.2,dark=.3")
naukri_df["Role"].value_counts()[0:20]
sns.barplot(x=naukri_df['Role'].value_counts()[0:20],y=naukri_df['Role'].value_counts()[0:20].index)
naukri_df["Role"].value_counts(normalize=True)[0:20].plot.barh()

plt.show()
df=naukri_df[naukri_df["Role"]=="Other"]

df.head()
city=["Bengaluru","Mumbai","Pune","Hyderabad"]
plt.figure(figsize = (20, 30))

for i in enumerate(city):

    plt.subplot(2, 2, i[0]+1)

    sns.barplot(x=naukri_df[naukri_df['Location'].isin([i[1]])]['Job Title'].value_counts().nlargest(n=20).index,

                        y=naukri_df[naukri_df['Location'].isin([i[1]])]['Job Title'].value_counts().nlargest(n=20).values)

    plt.title(i[1])

    plt.xlabel("Location")

    plt.ylabel("Job Titles")

    plt.tight_layout(pad=3.0)

    plt.xticks(rotation = 90)
plt.figure(figsize = (20, 30))

for i in enumerate(city):

    plt.subplot(2, 2, i[0]+1)

    sns.barplot(x=naukri_df[naukri_df['Location'].isin([i[1]])]['Role'].value_counts().nlargest(n=20).index,

                        y=naukri_df[naukri_df['Location'].isin([i[1]])]['Role'].value_counts().nlargest(n=20).values)

    plt.title(i[1])

    plt.xlabel("Location")

    plt.ylabel("Role")

    plt.tight_layout(pad=3.0)

    plt.xticks(rotation = 90)
plt.figure(figsize = (20, 20))

for i in enumerate(city):

    plt.subplot(2, 2, i[0]+1)

    sns.barplot(x=naukri_df[naukri_df['Location'].isin([i[1]])]['Job Experience Required'].value_counts().nlargest(n=5).index,

                        y=naukri_df[naukri_df['Location'].isin([i[1]])]['Job Experience Required'].value_counts().nlargest(n=5).values)

    plt.title(i[1])

    plt.xlabel("Location")

    plt.ylabel("Job Experience Required")

    plt.tight_layout(pad=3.0)

    plt.xticks(rotation = 90)
plt.figure(figsize=(12,8))

naukri_df[naukri_df["Role"]=='Software Developer']['Functional Area'].value_counts().plot.barh()
functional_words = naukri_df[naukri_df["Role"]=='Software Developer']['Functional Area'].value_counts()

functional_words
plt.figure(figsize=(12,8))

naukri_df[naukri_df["Role"]=='Associate/Senior Associate -(NonTechnical)']['Functional Area'].value_counts().plot.barh()
naukri_df[naukri_df["Role"]=='Associate/Senior Associate -(NonTechnical)']['Functional Area'].value_counts()
plt.figure(figsize=(12,8))

naukri_df[naukri_df["Role"]=='HR Executive']['Functional Area'].value_counts().plot.barh()
functional_words = naukri_df[naukri_df["Role"]=='HR Executive']['Functional Area'].value_counts()

functional_words
plt.figure(figsize=(12,8))

naukri_df[naukri_df["Role"]=='Testing Engineer']['Functional Area'].value_counts().plot.barh()
functional_words = naukri_df[naukri_df["Role"]=='Testing Engineer']['Functional Area'].value_counts()

functional_words
plt.figure(figsize=(12,8))

naukri_df[naukri_df["Role"]=='Business Analyst']['Functional Area'].value_counts().plot.barh()
functional_words=naukri_df[naukri_df["Role"]=='Business Analyst']['Functional Area'].value_counts()

functional_words
df=naukri_df[naukri_df['Role']=='Software Developer']
required_skills = df['Key Skills'].to_list()

wordcloud = WordCloud(width = 700, height = 700, 

                background_color ='white', 

                min_font_size = 10).generate(str(required_skills)) 

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
df1= df[df['Location'].isin(['Bengaluru'])]
required_skills = df1['Key Skills'].to_list()

wordcloud = WordCloud(width = 700, height = 700, 

                background_color ='white', 

                min_font_size = 10).generate(str(required_skills)) 

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
df=naukri_df[naukri_df['Role']=='Testing Engineer']
required_skills = df['Key Skills'].to_list()

wordcloud = WordCloud(width = 700, height = 700, 

                background_color ='white', 

                min_font_size = 10).generate(str(required_skills)) 

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
df=naukri_df[naukri_df['Role']=='Associate/Senior Associate -(NonTechnical)']
required_skills = df['Key Skills'].to_list()

wordcloud = WordCloud(width = 700, height = 700, 

                background_color ='white', 

                min_font_size = 10).generate(str(required_skills)) 

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 