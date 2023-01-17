import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

schema = pd.read_csv('../input/SurveySchema.csv')
df = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)

african_countries_survey = ['Morocco','Tunisia','Kenya','Egypt','South Africa','Nigeria']

df_afr = df[df['Q3'].isin(african_countries_survey )]
df_restworld = df[~df['Q3'].isin(african_countries_survey )]
df['Zone']=["Africa" if x in african_countries_survey else "RoW" for x in df['Q3']]

class Tweet(object):
    def __init__(self, embed_str=None):
        self.embed_str = embed_str

    def _repr_html_(self):
        return self.embed_str
df_afr.Q1.value_counts()/df_afr.Q1.value_counts().cumsum()[-1]

tmp = df_afr.Q1.value_counts()
labels = (np.array(tmp.index))
sizes = (np.array((tmp / tmp.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Gender of African Kagglers respondents'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="Gender")
Age = (df.groupby(['Zone'])['Q2']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Q2'))

plt.figure(figsize=(10,7))
p = sns.barplot(x="Q2", y="percentage", hue="Zone", data=Age[:-1])
_ = plt.setp(p.get_xticklabels(), rotation=45)  # Rotate labels

education = (df.groupby(['Zone'])['Q4']
             .value_counts(normalize=True)
             .rename('Ratio')
             .mul(100)
             .reset_index()
             .sort_values('Ratio', ascending=False))
education.Q4.replace(to_replace={"Some college/university study without earning a bachelor’s degree":"College without degree"}, 
                     inplace=True)
education.rename(columns={"Q4": "Highest education attained / planned"}, inplace=True)


plt.figure(figsize=(10,7))

p = sns.barplot(x="Highest education attained / planned", y="Ratio", hue="Zone", data=education[:10])
_ = plt.setp(p.get_xticklabels(), rotation=45)


#story about new master
tmp = df_afr[df_afr.Q1.isin(['Male','Female'])]

education_gender = (tmp.groupby(['Q1'])['Q4']
             .value_counts(normalize=True)
             .rename('Ratio')
             .mul(100)
             .reset_index()
             .sort_values('Ratio', ascending=False))
education_gender.Q4.replace(to_replace={"Some college/university study without earning a bachelor’s degree":"College without degree"}, 
                     inplace=True)
education_gender.rename(columns={"Q4": "Highest education attained / planned", 
                                "Q1" : "Gender"}, inplace=True)

plt.figure(figsize=(10,7))

p = sns.barplot(x="Highest education attained / planned", y="Ratio", hue="Gender", data=education_gender[:8])
_ = plt.setp(p.get_xticklabels(), rotation=45)

#  + anecdote cissé sur le ratio hommes / femmes 
tmp ={
    'Self-taught' : (df_afr['Q35_Part_1'].astype(float)>0).sum(),
    'Online courses' : (df_afr['Q35_Part_2'].astype(float)>0).sum(),
    'Work' : (df_afr['Q35_Part_3'].astype(float)>0).sum(),
    'University' : (df_afr['Q35_Part_4'].astype(float)>0).sum(),
    'Kaggle' : (df_afr['Q35_Part_5'].astype(float)>0).sum(),
    'Other' : (df_afr['Q35_Part_6'].astype(float)>0).sum()
}


tmp = round(100*pd.DataFrame.from_dict(tmp, orient='index')/len(df_afr),2)
tmp.reset_index(inplace=True)
tmp.rename(columns={"index": "Mean of learning ML", 0 : "percentage"}, inplace=True)
tmp.sort_values(by='percentage', ascending=False, inplace=True)

plt.figure(figsize=(11,7))
p = sns.barplot(x="Mean of learning ML", y="percentage", data=tmp)
_ = plt.setp(p.get_xticklabels(), rotation=45)
tmp = {
    'Coursera' : (df_afr['Q36_Part_2'].count()),
    'Udemy' : (df_afr['Q36_Part_9'].count()),
    'DataCamp' : (df_afr['Q36_Part_4'].count()),
    'Kaggle' : (df_afr['Q36_Part_6'].count()),
    'Udacity' : (df_afr['Q36_Part_1'].count()),
    'edX' : (df_afr['Q36_Part_3'].count()),
    'Online University Courses' : (df_afr['Q36_Part_11'].count()),
    'Fast.AI' : (df_afr['Q36_Part_7'].count()),
}


tmp = pd.DataFrame.from_dict(tmp, orient='index')
tmp.reset_index(inplace=True)
tmp.rename(columns={"index": "Online platform", 0 : "Number of respondents"}, inplace=True)
tmp.sort_values(by='Number of respondents', ascending=False, inplace=True)

plt.figure(figsize=(11,7))
p = sns.barplot(x="Online platform", y="Number of respondents", data=tmp)
_ = plt.setp(p.get_xticklabels(), rotation=45)
education_gender = (df.groupby(['Zone'])['Q39_Part_1']
             .value_counts(normalize=True)
             .rename('Ratio')
             .mul(100)
             .reset_index()
             .sort_values('Ratio', ascending=False))

education_gender.rename(columns={"Q39_Part_1": "Opinion : MOOCs VS Uni"}, inplace=True)

plt.figure(figsize=(11,7))

p = sns.barplot(x="Opinion : MOOCs VS Uni", y="Ratio", hue="Zone", data=education_gender[:-1])
_ = plt.setp(p.get_xticklabels(), rotation=45)
s = ("""<blockquote class="twitter-tweet" data-lang="fr"><p lang="en" dir="ltr">Wonderful to hear these testimonials that show the impact on African deep learning of <a href="https://twitter.com/DeepIndaba?ref_src=twsrc%5Etfw">@DeepIndaba</a> , <a href="https://twitter.com/black_in_ai?ref_src=twsrc%5Etfw">@black_in_ai</a> , and <a href="https://twitter.com/fastdotai?ref_src=twsrc%5Etfw">@fastdotai</a> . Just 2 years ago our African students told us there was no community in most countries at all! <a href="https://t.co/eCmHnMxYgD">pic.twitter.com/eCmHnMxYgD</a></p>&mdash; Jeremy Howard (@jeremyphoward) <a href="https://twitter.com/jeremyphoward/status/1054015751940521985?ref_src=twsrc%5Etfw">21 octobre 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)
s = """<blockquote class="twitter-tweet" data-conversation="none" data-lang="fr"><p lang="en" dir="ltr">No other Master&#39;s program in the entire world has such an amazing line up of lecturers.</p>&mdash; Yann LeCun (@ylecun) <a href="https://twitter.com/ylecun/status/1025213542339948544?ref_src=twsrc%5Etfw">3 août 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
"""
Tweet(s)
use_ml = (df.groupby(['Zone'])['Q25']
             .value_counts(normalize=True)
             .rename('Ratio')
             .mul(100)
             .reset_index()
             .sort_values('Ratio', ascending=False))


use_ml.Q25.replace(to_replace={"I have never studied machine learning but plan to learn in the future":"Plan to in future"}, 
                     inplace=True)
use_ml.rename(columns={"Q25": "Years of experience in ML"}, inplace=True)



plt.figure(figsize=(11,7))
p = sns.barplot(x="Years of experience in ML", y="Ratio", hue="Zone", data=use_ml[:14])
_ = plt.setp(p.get_xticklabels(), rotation=45)

# df_afr.Q25.value_counts()
ml_work = (df.groupby(['Zone'])['Q10']
             .value_counts(normalize=True)
             .rename('Ratio')
             .mul(100)
             .reset_index()
             .sort_values('Ratio', ascending=False))


ml_work.Q10.replace(to_replace={"We are exploring ML methods (and may one day put a model into production)":"Exploration Phase",
                               "No (we do not use ML methods)" : "No use of ML",
                               "We recently started using ML methods (i.e., models in production for less than 2 years)":"ML in prod for < 2 years",
                               "We have well established ML methods (i.e., models in production for more than 2 years)":"ML in prod fore > 2 years",
                               "We use ML methods for generating insights (but do not put working models into production)":"ML for insights but not prod"
                               }, 
                     inplace=True)
ml_work.rename(columns={"Q10": "Years of ML at workplace"}, inplace=True)

plt.figure(figsize=(11,7))
p = sns.barplot(x="Years of ML at workplace", y="Ratio", hue="Zone", data=ml_work[:-2])
_ = plt.setp(p.get_xticklabels(), rotation=45)

industry = (df.groupby(['Zone'])['Q7']
             .value_counts(normalize=True)
             .rename('Ratio')
             .mul(100)
             .reset_index()
             .sort_values('Ratio', ascending=False))
industry.rename(columns={"Q7": "Industry"}, inplace=True)

industry = industry[industry.Zone=="Africa"]
plt.figure(figsize=(11,7))

p = sns.barplot(y="Industry", x="Ratio", data=industry[:12])