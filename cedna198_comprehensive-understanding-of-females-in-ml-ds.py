# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
schema = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')

questions = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')

mcq = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

responses = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')

responses.head()
responses.Q13_OTHER_TEXT.head(50)
questions.head()
schema.head()
schema.groupby('Q13')['Q31'].count().head(500)
schema['Q4'].describe()
mcq.head()
## Let us first get to understand the shape of the data before choosing which one to work with



responses_len, schema_len, mcq_len, questions_len = len(responses.index), len(schema.index),len(mcq.index), len(questions.index)

print(f'responses size: {responses_len}, schema size: {schema_len} , mcq size: {mcq_len}, questions size: {questions_len}')
## Count the missing values 

miss_val_mcq = mcq.isnull().sum(axis=0) / mcq_len

miss_val_mcq = miss_val_mcq[miss_val_mcq> 0] * 100

miss_val_mcq
## Number of mcq columns are

len(list(mcq.columns))
## Lets see the words of first question



plt.figure(figsize=(20, 5))



text = schema.Q4[0]



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
## Lets see the words of first question



plt.figure(figsize=(20, 5))



text = mcq.Q4[200]



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
mcq.Q2.head(50)
##How many males and females specified their gender? 



text = " ".join(Q2 for Q2 in mcq.Q2)

print ("There were {} males and females who specified their gender in the survey conducted.".format(len(text)))
## Check the scoring for questions

all_mcq_columns = list(mcq.columns)

Q2 = all_mcq_columns[:11]

Q4 = all_mcq_columns[11:32]

Q3  = all_mcq_columns[32:41]
mcq[Q4]
mcq.Q5.head(50)

mcq.Q4.head(50)
mcq.groupby('Q2')['Q4'].count().head(50)

mcq['Q4'].describe()
schema['Q3'].describe()
schema['Q5'].describe()
schema['Q1'].describe()
schema['Q2'].describe()
pt1 = mcq[['Q1', 'Q2']]

pt1 = pt1.rename(columns={"Q1": "Age", "Q2": "Gender"})

pt1.drop(0, axis=0, inplace=True)



# plotting to create pie chart 

plt.figure(figsize=(18,12))

plt.subplot(221)

pt1["Gender"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",8),startangle = 60,labels=["Male","Female","Prefer not to say","Prefer to self-describe"],

wedgeprops={"linewidth":4,"edgecolor":"k"},explode=[.1,.1,.2,.3],shadow =True)

plt.title("Gender Distribution in (%)")



plt.show()
plt.figure(figsize=(16,10))

sorted_age=['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']

sns.countplot(x='Age', hue='Gender',order=sorted_age, data=pt1 )

plt.title('Gender distribution based on Their Ages',fontsize=24)

plt.ylabel('Number of Survey Takers', fontsize = 18.0) # Y label

plt.xlabel('Age of the Survey Takers', fontsize = 18) # X label

plt.show()
plt.figure(figsize=(12,6))

pt2 = mcq[['Q2','Q4', 'Q5']]

pt2 = pt2.rename(columns={"Q2":"Gender","Q4": "Education", "Q5": "title"})

pt2.drop(0, axis=0, inplace=True)



# Replacing the ambigious education names with easy to use names

pt2['Education'].replace(

                                                   {'Master’s degree':'MS',

                                                    'Bachelor’s degree':'Grad',

                                                    "Doctoral degree":'Doct',

                                                    "Some college/university study without earning a bachelor’s degree":'Under Grad',

                                                    "Professional degree":"Professional",

                                                   "I prefer not to answer":"Prefer NA",

                                                   "No formal education past high school":"No edu"},inplace=True)



sns.countplot(x='Education', data=pt2 )

plt.title('Education wise distribution',fontsize=18)

plt.ylabel('Number of People', fontsize = 16.0) # Y label

plt.xlabel('Education', fontsize = 16) # X label

plt.show()



plt.figure(figsize=(12,6))



sns.countplot(y='title', data=pt2 )

plt.title("Job Profile Distribution", fontsize=18)



plt.ylabel('Job Profile', fontsize = 16.0) # Y label

plt.xlabel('Number of People', fontsize = 16) # X label

plt.show()
pt2 = pt2.rename(columns={"Q5":"title","Q10": "Education"})

pt2.drop(0, axis=0, inplace=True)

plt.figure(figsize=(16,8))



sns.countplot(x='title',hue='Education', data=pt2 )

plt.title("Job Profile vs Education Distribution", fontsize=18)



plt.ylabel('Number of People', fontsize = 16.0) # Y label

plt.xlabel('Job Profile', fontsize = 16) # X label

plt.xticks(rotation=45)

plt.show()
plt.style.use('fivethirtyeight')

sns.barplot(mcq['Q15'].value_counts()[0:7].values,mcq['Q15'].value_counts()[0:7].index,palette=('bright'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Coding Experience in relation to their salary levels')

plt.show()
plt.figure(figsize=(14,8))

ptML = mcq[['Q15','Q10']]

ptML = ptML.rename(columns={"Q15":"code_experience","Q10": "Income"})

ptML.drop(0, axis=0, inplace=True)

sorted_exp=['I have never written code','< 1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years']

sns.countplot(x='code_experience',hue='Income',order=sorted_exp, data=ptML )

plt.title("Coding Experience wise Income Distribution", fontsize=18)



plt.ylabel('Number of People', fontsize = 16.0) # Y label

plt.xlabel('Coding Experience in years', fontsize = 16) # X label

plt.xticks(rotation=45)

plt.show()
media = ["Twitter", "HackerNews", "Reddit", "Kaggle", "Forums", "YouTube", "Podcasts", "Blogs", "Journals", "Slack"]

media_count = [sum(~mcq.Q12_Part_1.isna()), sum(~mcq.Q12_Part_2.isna()), sum(~mcq.Q12_Part_3.isna()), sum(~mcq.Q12_Part_4.isna()), sum(~mcq.Q12_Part_5.isna()),

               sum(~mcq.Q12_Part_6.isna()), sum(~mcq.Q12_Part_7.isna()), sum(~mcq.Q12_Part_8.isna()), sum(~mcq.Q12_Part_9.isna()), sum(~mcq.Q12_Part_10.isna())]



pt_media = pd.DataFrame({"media": media, "media_percentage": np.array(media_count) * 100 / mcq.shape[0]})

pt_media.sort_values("media_percentage", ascending=False, inplace=True)

plt.style.use('fivethirtyeight')

sns.barplot(pt_media['media_percentage'],pt_media['media'],palette=('coolwarm'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Media Sources Responsible for Learning Datascience')

plt.show()
labels_frame = ["Scikit-Learn","Tensorflow ","Keras","Random Forest","Xgboost","Pytorch","Caret","LightGBM", "Spark MLLib", "Fast.ai"]

frame_count = [sum(~mcq.Q28_Part_1.isna()), sum(~mcq.Q28_Part_2.isna()), sum(~mcq.Q28_Part_3.isna()), sum(~mcq.Q28_Part_4.isna()), sum(~mcq.Q28_Part_5.isna()),

             sum(~mcq.Q28_Part_6.isna()), sum(~mcq.Q28_Part_7.isna()), sum(~mcq.Q28_Part_8.isna()), sum(~mcq.Q28_Part_9.isna()), sum(~mcq.Q28_Part_10.isna())]



pt_frame = pd.DataFrame({"ML Frameworks": labels_frame, "Percentage Used": np.array(frame_count) * 100 / mcq.shape[0]})

pt_frame.sort_values("Percentage Used", ascending=False, inplace=True)



plt.style.use('classic')

sns.barplot(pt_frame['ML Frameworks'],pt_frame['Percentage Used'])

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.title('ML Frameworks Popularity')

plt.show()