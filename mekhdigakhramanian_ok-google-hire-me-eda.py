
plt.style.use('ggplot')

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization

import seaborn as sns
import plotly.express as px

import pydicom # for DICOM images
from skimage.transform import resize

# SKLearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import os
import random
import re
import math
import time
from IPython.display import display_html
import missingno as msno 
import gc
import cv2
import matplotlib.image as mpimg


# Setting color palette.
black_red = [
    '#1A1A1D', '#4E4E50', '#C5C6C7', '#6F2232', '#950740', '#C3073F'
]

# Setting plot styling.
plt.style.use('fivethirtyeight')


import warnings

warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
df = pd.read_csv('../input/google-job-skills/job_skills.csv')
df.head()
df.tail()
# Change the columns name
df = df.rename(columns={'Minimum Qualifications': 'Minimum_Qualifications', 'Preferred Qualifications': 'Preferred_Qualifications'})
pd.isnull(df).sum()
df = df.dropna(how='any',axis='rows')
df.Company.value_counts()
#labels
lab = df["Company"].value_counts().keys().tolist()

#values
val = df["Company"].value_counts().values.tolist()

trace1 = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ "#6F2232" ,"#4E4E50"],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Gender Distribution",
                        plot_bgcolor  = "#1A1A1D",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )



data = [trace1]
fig  = go.Figure(data = data,layout = layout)
py.iplot(fig)
# So I drop Youtube
df = df[df.Company != 'YouTube']
df.Title.value_counts()[:10]
df.Location.value_counts()[:10]
df['Country'] = df['Location'].apply(lambda x : x.split(',')[-1])
df.Country.value_counts()[:15]
# Creating a customized chart and giving in figsize etc.

fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid:

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

sns.countplot(df.Country,
              alpha=0.9,
              color='#1A1A1D',
              label='Country',
              order=df['Country'].value_counts().index)
sns.countplot(df.Country,
              alpha=0.7,
              color='#C3073F',
              label='Country',
              order=df['Country'].value_counts().index)

plt.xticks(rotation=75)
plt.show()
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english')) 

df['Responsibilities'] = df.Responsibilities.apply(lambda x: word_tokenize(x))
df['Responsibilities'] = df.Responsibilities.apply(lambda x: [w for w in x if w not in stop_words])
df['Responsibilities'] = df.Responsibilities.apply(lambda x: ' '.join(x))

df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: word_tokenize(x))
df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])
df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: ' '.join(x))

df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: word_tokenize(x))
df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])
df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: ' '.join(x))
# The way to extract year refer to https://www.kaggle.com/niyamatalmass/what-you-need-to-get-a-job-at-google.
# Thanks Niyamat Ullah for such brilliant way. Go check his kernel. It's great!
import re
df['Minimum_years_experience'] = df['Minimum_Qualifications'].apply(lambda x : re.findall(r'([0-9]+) year',x))
# Fill empty list with [0]
df['Minimum_years_experience'] = df['Minimum_years_experience'].apply(lambda y : [0] if len(y)==0 else y)
#Then extract maximum in the list to have the work experience requirement
df['Minimum_years_experience'] = df['Minimum_years_experience'].apply(lambda z : max(z))
df['Minimum_years_experience'] = df.Minimum_years_experience.astype(int)
df.head(3)
df.Minimum_years_experience.describe()
df.Category.value_counts()[:10]
#labels
lab = df["Category"].value_counts().keys().tolist()

#values
val = df["Category"].value_counts().values.tolist()

trace1 = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Category",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace1]
fig  = go.Figure(data = data,layout = layout)
py.iplot(fig)
pd.set_option('display.max_colwidth', -1)
df.head(1)
Degree = ['BA','BS','Bachelor','MBA','Master','PhD']

Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)
degree_requirement = sorted(Degrees.items(), key=lambda x: x[1], reverse=True)
degree = pd.DataFrame(degree_requirement,columns=['Degree','Count'])
degree['Count'] = degree.Count.astype('int')
degree
fig = px.sunburst(data_frame=degree,
                  path=['Degree','Count'],
                  color='Degree',
                  color_discrete_sequence=black_red,
                  maxdepth=-1,
                  title='Degrees Distribution')

fig.update_traces(textinfo='label+percent parent')
fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
fig.show()
df.Minimum_years_experience.plot(kind='box', patch_artist=True)
plt.title('Minimum work experience')
plt.ylabel('Years')
import seaborn as sns
sns.countplot('Minimum_years_experience',data=df)
plt.suptitle('Minimum work experience')
Programming_Languages = ['Python', 'Java ','C#', 'PHP', 'Javascript', 'Ruby', 'Perl', 'SQL','Go ']

Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)
languages_requirement = sorted(Languages.items(), key=lambda x: x[1], reverse=True)
language = pd.DataFrame(languages_requirement,columns=['Language','Count'])
language['Count'] = language.Count.astype('int')
language
fig = px.sunburst(data_frame=language,
                  path=['Language','Count'],
                  color='Language',
                  color_discrete_sequence=black_red,
                  maxdepth=-1,
                  title='Languages Distribution')

fig.update_traces(textinfo='label+text+current path+percent entry+percent parent')
fig.update_layout(margin=dict(t=20, l=0, r=0, b=0))
fig.show()
language.plot.barh(x='Language',y='Count',legend=False)
plt.suptitle('Languages Distribution',fontsize=14)
plt.xlabel('Count')
def MadeWordCloud(title,text):
    df_subset = df.loc[df.Title.str.contains(title).fillna(False)]
    long_text = ' '.join(df_subset[text].tolist())
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    wordcloud = WordCloud(mask=G,background_color="white").generate(long_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title(text,size=24)
    plt.show()
# Refer to https://python-graph-gallery.com/262-worcloud-with-specific-shape/
# https://amueller.github.io/word_cloud/auto_examples/masked.html

df_Analyst = df.loc[df.Title.str.contains('Analyst').fillna(False)]
df_Analyst.head(1)
df_Analyst.Country.value_counts()
Res_AN = ' '.join(df_Analyst['Responsibilities'].tolist())
# from wordcloud import WordCloud, ImageColorGenerator
# from PIL import Image
# G = np.array(Image.open('../input/googlelogo/img_2241.png'))
# # I spent a while to realize that the image must be black-shaped to be a mask
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=G,background_color="white").generate(Res_AN)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilites',size=24)
plt.show()
MadeWordCloud('Analyst','Minimum_Qualifications')
MadeWordCloud('Analyst','Preferred_Qualifications')
DataSkill = [' R','Python','SQL','SAS']

DataSkills = dict((x,0) for x in DataSkill)
for i in DataSkill:
    x = df_Analyst['Minimum_Qualifications'].str.contains(i).sum()
    if i in DataSkill:
        DataSkills[i] = x
        
print(DataSkills)
Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Analyst['Minimum_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)
Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Analyst['Preferred_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)
sns.countplot('Minimum_years_experience',data=df_Analyst)
plt.suptitle('Minimum work experience')
df_Developer = df.loc[df.Title.str.contains('Developer').fillna(False)]
df_Developer.Country.value_counts()
MadeWordCloud('Developer','Responsibilities')
MadeWordCloud('Developer','Minimum_Qualifications')
MadeWordCloud('Developer','Preferred_Qualifications')
DataSkill = ['Java ','Javascript','Go ','Python','Kotlin','SQL']

DataSkills = dict((x,0) for x in DataSkill)
for i in DataSkill:
    x = df_Developer['Minimum_Qualifications'].str.contains(i).sum()
    if i in DataSkill:
        DataSkills[i] = x
        
print(DataSkills)
Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Developer['Minimum_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)
Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Developer['Preferred_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)
sns.countplot('Minimum_years_experience',data=df_Developer)
plt.suptitle('Minimum work experience')
df_MBA = df.loc[df.Title.str.contains('MBA').fillna(False)]
df_MBA.head(1)
df_MBA.Category.value_counts()
df_MBA.Country.value_counts()
MadeWordCloud('MBA','Responsibilities')
MadeWordCloud('MBA','Minimum_Qualifications')
MadeWordCloud('MBA','Preferred_Qualifications')
Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_MBA['Minimum_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)
Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_MBA['Preferred_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)
sns.countplot('Minimum_years_experience',data=df_MBA)
plt.suptitle('Minimum work experience')
df_Sales = df.loc[df.Title.str.contains('Sales').fillna(False)]
df_Sales.Category.value_counts()
df_Sales.Country.value_counts()[:5]
MadeWordCloud('Sales','Responsibilities')
MadeWordCloud('Sales','Minimum_Qualifications')
MadeWordCloud('Sales','Preferred_Qualifications')
Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_Sales['Minimum_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)
Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_Sales['Preferred_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)
sns.countplot('Minimum_years_experience',data=df_Sales)
plt.suptitle('Minimum work experience')
Microsoft_Office = ['Excel','Powerpoint','Word','Microsoft']

MO = dict((x,0) for x in Microsoft_Office)
for i in MO:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Microsoft_Office:
        MO[i] = x
        
print(MO)
DV_Tools = ['Tableau','Power BI','Qlik','Data Studio','Google Analytics','GA']

DV = dict((x,0) for x in DV_Tools)
for i in DV:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in DV_Tools:
        DV[i] = x
        
print(DV)
SA_Tools = ['SPSS','R ','Matlab','Excel','Spreadsheet','SAS']

SA = dict((x,0) for x in SA_Tools)
for i in SA:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in SA_Tools:
        SA[i] = x
        
print(SA)
df_US = df.loc[df.Country == ' United States']
df_US_Type = df_US.Category.value_counts()
df_US_Type = df_US_Type.rename_axis('Type').reset_index(name='counts')
import squarify
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
cmap = matplotlib.cm.Blues
norm = matplotlib.colors.Normalize(vmin=min(df_US_Type.counts), vmax=max(df_US_Type.counts))
colors = [cmap(norm(value)) for value in df_US_Type.counts]
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(24, 6)
squarify.plot(sizes=df_US_Type['counts'], label=df_US_Type['Type'], alpha=.8, color=colors)
plt.title('Type of positions',fontsize=20,fontweight="bold")
plt.axis('off')
plt.show()
PM_positions = ['Product Manager','Project Manager','Program Manager']

PM = dict((x,0) for x in PM_positions)
for i in PM:
    x = df['Title'].str.contains(i).sum()
    if i in PM_positions:
        PM[i] = x
        
print(PM)
Project_Management_words = ['Jira','scrum','agile']

Project_Management = dict((x,0) for x in Project_Management_words)
for i in Project_Management:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Project_Management_words:
        Project_Management[i] = x
        
print(Project_Management)
Project_Management = dict((x,0) for x in Project_Management_words)
for i in Project_Management:
    x = df['Preferred_Qualifications'].str.contains(i).sum()
    if i in Project_Management_words:
        Project_Management[i] = x
        
print(Project_Management)
df_groupby_country_category = df.groupby(['Country','Category'])['Category'].count()
df_groupby_country_category.loc[' United States']
category_country = df.pivot_table(index=['Country','Category'],values='Minimum_years_experience',aggfunc='median')
category_country.loc[' United States']
category_country.loc['Singapore']
category_country.loc[' Taiwan']
category_country.loc[' India']

