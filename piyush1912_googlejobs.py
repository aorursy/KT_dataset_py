
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt# plotting the graphs
from wordcloud import WordCloud

from collections import defaultdict
plt.style.use('ggplot')
import seaborn as sns
import re

%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




# read the data 
df_dataset = pd.read_csv('../input/google-jobs/job_skills.csv')
# print the top 5 row from the dataframe
df_dataset.head()
# most popular language list 
programing_language_list = ['python', 'java', 'c++', 'php', 'javascript', 'objective-c', 'ruby', 'perl','c','c#', 'sql','kotlin']
# get our Preferred Qualifications and convert into a list
pref_qualifications = df_dataset['Preferred Qualifications'].tolist()
# let's join our list to a single string and lower case the letter
pref_qualifications_string = "".join(str(v) for v in pref_qualifications).lower()
# find out which language occurs in most in minimum Qualifications string
wordcount = dict((x,0) for x in programing_language_list)
for w in re.findall( r"[\w'+#-]+|[.!?;’]", pref_qualifications_string):
    if w in wordcount:
        wordcount[w] += 1
# sort the dictionary
programming_language_popularity = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
# make a new dataframe using programming_language_popularity for easy use cases
df_popular_programming_lang = pd.DataFrame(programming_language_popularity,columns=['Language','Popularity'])
# Capitalize each programming language first letter
df_popular_programming_lang['Language'] = df_popular_programming_lang.Language.str.capitalize()
df_popular_programming_lang = df_popular_programming_lang[::-1]
# plot
df_popular_programming_lang.plot.barh(x='Language',y='Popularity',figsize=(10, 8), legend=False,stacked=True)





# add a suptitle
plt.suptitle("Programming Languages popularity at Google Jobs", fontsize=18)
plt.xlabel("")
plt.ylabel("Language",fontsize=18)
# change xticks fontsize to 14
plt.yticks(fontsize=18)
# finally show the plot
plt.show()
pref_qualifications_string = " ".join(str(v) for v in pref_qualifications)
degree_list = ["BA", "BS", "Bachelor's","Masters","MS", "PhD"]
wordcount = dict((x,0) for x in degree_list)
for w in re.findall(r"[\w']+|[.,!?;’]", pref_qualifications_string):
    if w in wordcount:
        wordcount[w] += 1
degree_popularity = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
df_degree_popular = pd.DataFrame(degree_popularity,columns=['Degree','Popularity'])
df_degree_popular = df_degree_popular[::-1] 
# plot
wordcloud = WordCloud(width=480, height=480, margin=0).generate(pref_qualifications_string)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

df_degree_popular.plot.barh(x='Degree',y='Popularity',figsize=(10, 8), legend=False,stacked=True)




# add a suptitle
plt.suptitle("Degree popularity at Google Jobs", fontsize=18)
plt.xlabel("")
# change xticks fontsize to 14
plt.yticks(fontsize=18)
# finally show the plot
plt.show()
#Years of experience count
years_exp = defaultdict(lambda: 0)

for w in re.findall(r'([0-9]+) year', pref_qualifications_string):
     years_exp[w] += 1
        

years_exp = sorted(years_exp.items(), key=lambda kv: kv[1], reverse=True)
df_years_exp = pd.DataFrame(years_exp,columns=['Years of experience','Popularity'])
df_years_exp = df_years_exp[::-1] 
# plot
df_years_exp.plot.barh(x='Years of experience',y='Popularity',figsize=(10, 8), legend=False,stacked=True)
# add a suptitle
plt.title("Years of experiences needed for Google Jobs", fontsize=18)
# set xlabel to ""
plt.xlabel("Popularity", fontsize=14)
plt.ylabel("Number of Years",fontsize=18)
# change xticks fontsize to 14
plt.yticks(fontsize=18)
# finally show the plot
plt.show()
df_dataset['Experience'] = df_dataset['Preferred Qualifications'].str.extract(r'([0-9]+) year')
dff = df_dataset[['Experience','Category']]
dff = dff.dropna()
plt.figure(figsize=(10,15))
plt.title('Experiences needed in different job category', fontsize=24)
sns.countplot(y='Category', hue='Experience', data=dff, hue_order=dff.Experience.value_counts().iloc[:3].index)
plt.yticks(fontsize=18)
plt.show()
threshold = 10
location_value_counts = df_dataset.Location.value_counts()
to_remove = location_value_counts[location_value_counts <= threshold].index
df_dataset['Location'].replace(to_remove, np.nan, inplace=True)
location_value_counts = df_dataset.Location.value_counts()
location_value_counts = location_value_counts[::-1]
location_value_counts.plot.barh(figsize=(20, 20))
# add a suptitle
plt.title("Google Jobs Location Popularity", fontsize=24)
# set xlabel to ""
plt.xlabel("Popularity", fontsize=20)
plt.ylabel("Location",fontsize=20)
# change xticks fontsize to 14
plt.yticks(fontsize=24)
# finally show the plot
plt.show()
category_value_counts = df_dataset.Category.value_counts()
category_value_counts = category_value_counts[::-1]
category_value_counts.plot.barh(figsize=(20, 20))
# add a suptitle
plt.title("What is the most popular job category at Google?", fontsize=24)
# set xlabel to ""
plt.xlabel("Popularity", fontsize=20)
plt.ylabel("Job Category",fontsize=20)
plt.yticks(fontsize=24)
# finally show the plot
plt.show()
plt.figure(figsize=(20,25))
plt.title('Google job categories popularity in different locations', fontsize=24)

sns.countplot(y='Location', hue='Category', data=df_dataset, hue_order=dff.Category.value_counts().iloc[:3].index)

plt.yticks(fontsize=18)
plt.show()
df_Data = df_dataset.loc[df_dataset.Title.str.contains('Data').fillna(False)]
df_Data.head(5)
from PIL import Image

g = np.array(Image.open('../input/picture/google//google-logo_318-50213.jpg'))
R_DA = ' '.join(df_Data['Responsibilities'].tolist())
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=g,background_color="white").generate(R_DA)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilites',size=24)
plt.show()
MP_DA = ' '.join(df_Data['Minimum Qualifications'].tolist())
MP_DA =' '.join(df_Data['Preferred Qualifications'].tolist())
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=g,background_color="white").generate(MP_DA)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Qualifications',size=24)
plt.show()
df_Data1 = df_dataset.loc[df_dataset.Title.str.contains('Sales').fillna(False)]
df_Data1.head(5)
DE_Q =''.join(df_Data1['Responsibilities'].tolist())
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=g,background_color="white").generate(DE_Q)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilities',size=24)
plt.show()
DE_R =''.join(df_Data1['Minimum Qualifications'].tolist())
DE_R = ''.join(df_Data1['Preferred Qualifications'].tolist())
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=g,background_color="white").generate(DE_R)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Qualifications',size=24)
plt.show()
df_Data2 = df_dataset.loc[df_dataset.Title.str.contains('Cloud').fillna(False)]
df_Data2.head(5)
SE_L = ''.join(df_Data2['Responsibilities'].tolist())
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=g,background_color="white").generate(SE_L)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responisibilities',size=24)
plt.show()
SE_QA =''.join(df_Data2['Minimum Qualifications'].tolist())
SE_QA =''.join(df_Data2['Preferred Qualifications'].tolist())
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=g,background_color="white").generate(SE_QA)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Qualificatons',size=24)
plt.show()