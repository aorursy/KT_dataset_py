# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
viz=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
df.columns=df.iloc[0]
df=df.drop([0])
stu_ds=df[df['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(df['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()[:2].index)]
stu_ds['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().plot.bar(width=0.8)
plt.gcf().set_size_inches(8,6)
plt.title('No of Respondents')
plt.xticks(rotation=0)
plt.show()
ds=stu_ds[stu_ds['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=='Data Scientist']
stu=stu_ds[stu_ds['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=='Student']
ds['Which best describes your undergraduate major? - Selected Choice'].value_counts().plot.barh(width=0.95,color=sns.color_palette('RdYlGn',30))
plt.gcf().set_size_inches(8,8)
plt.gca().invert_yaxis()
plt.title('UnderGrad Major')
plt.show()
ds['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().plot.barh(width=0.95,color=sns.color_palette('RdYlGn',10))
plt.gcf().set_size_inches(8,8)
plt.gca().invert_yaxis()
plt.title('Highest Level of Education')
plt.show()
ds['Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:'].value_counts().plot.barh(width=0.95,color=sns.color_palette('viridis',5))
plt.gcf().set_size_inches(6,8)
plt.title('Personal Projects vs Academics')
plt.gca().invert_yaxis()
plt.show()
ds['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts().plot.barh(width=0.95,color=sns.color_palette('RdYlGn',10))
plt.gcf().set_size_inches(10,12)
plt.gca().invert_yaxis()
plt.title('First Language')
plt.show()
plt.show()
l1=[col for col in ds if col.startswith("What programming languages do you use on a regular basis? (Select all that apply)")]
col1=[]
col2=[]
l2=ds[l1[:-2]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
lang=pd.DataFrame({'Lang':col1,'Count':col2})
lang.set_index('Lang').plot.barh(width=0.9)
plt.gcf().set_size_inches(8,10)
plt.title('Languages Used')
plt.show()
lang=[col for col in ds if col.startswith('What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice -')]
lang=lang[:-2]
df_lang=ds[lang]
df_lang.columns=[i[102:] for i in lang]
c = df_lang.stack().groupby(level=0).apply(tuple).value_counts()
out = [i + (j,) for i, j in c.items()]
out=[word for word in out if len(word)==3]
lang_net=pd.DataFrame(out)
lang_net.columns=['Lang 1','Lang 2','Count']
g = nx.from_pandas_edgelist(lang_net,source='Lang 1',target='Lang 2')
cmap = plt.cm.RdYlGn
colors = [n for n in range(len(g.nodes()))]
k = 0.35
pos=nx.spring_layout(g, k=k)
nx.draw_networkx(g,node_size=lang_net['Count'].values*20, cmap = cmap, node_color=colors, edge_color='grey', font_size=20, width=lang_net['Count'].values*0.05)
plt.title('Languages Network')
plt.gcf().set_size_inches(22,20)
import itertools
import math
train=[col for col in ds if col.startswith('What percentage of your current machine learning/data science training falls under each category? ')]
train=train[:-2]
plt.figure(figsize=(20,20))
length=len(train)
for i,j in itertools.zip_longest(train,range(length)):
    plt.subplot(math.ceil((length/2)),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    ds[i].astype('float').hist(bins=10,edgecolor='black',color='khaki')
    plt.axvline(ds[i].astype('float').mean(),linestyle='dashed',color='r')
    plt.title(i[130:],size=20)
    plt.xlabel('% Time')
plt.show()
l1=[col for col in ds if col.startswith('Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice -')]
col1=[]
col2=[]
l2=ds[l1[:-4]]
l2
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
activity=pd.DataFrame({'Source':col1,'Count':col2})
activity.set_index('Source').plot.barh(width=0.95)
plt.gcf().set_size_inches(8,12)
plt.gca().invert_yaxis()
plt.title('Blogs/Articles',size=20)
plt.show()
ds['On which online platform have you spent the most amount of time? - Selected Choice'].value_counts().plot.barh(width=0.9,color=sns.color_palette('viridis',10))
plt.gcf().set_size_inches(8,8)
plt.gca().invert_yaxis()
plt.title("Top MOOC's")
plt.show()
ds['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts().plot.barh(width=0.95,color=sns.color_palette('viridis',20))
plt.gcf().set_size_inches(10,12)
plt.gca().invert_yaxis()
plt.title('Current Industry')
plt.show()
l1=[col for col in ds if col.startswith("Select any activities that make up an important part of your role at work: (Select all that apply) ")]
col1=[]
col2=[]
l2=ds[l1[:-2]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
activity=pd.DataFrame({'Activity':col1,'Count':col2})
activity.set_index('Activity').plot.barh(width=0.95)
plt.gcf().set_size_inches(10,12)
plt.gca().invert_yaxis()
plt.title('Daily Activities',size=20)
plt.show()
time_spent=[col for col in ds if col.startswith("During a typical data science project at work or school, approximately what proportion of your time is devoted to the following? ")]
time_spent=time_spent[:-1]
plt.figure(figsize=(20,20))
length=len(time_spent)
for i,j in itertools.zip_longest(time_spent,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    ds[i].astype('float').hist(bins=10,edgecolor='black',color='tomato')
    plt.axvline(ds[i].astype('float').mean(),linestyle='dashed',color='black')
    plt.title(i[161:],size=20)
    plt.xlabel('% Time')
plt.show()
import matplotlib.gridspec as gridspec
scientist=viz[viz['DataScienceIdentitySelect']=='Yes']
fig = plt.figure(figsize=(15,18))
gridspec.GridSpec(2,2)

plt.subplot2grid((2,2), (0,0), colspan=1,rowspan=2)
ds['Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].value_counts().plot.barh(width=0.95,color=sns.color_palette('inferno',10))
plt.gca().invert_yaxis()
plt.title('Top Visualization Libraries')
plt.subplot2grid((2,2), (0,1))
sns.countplot(scientist['JobSkillImportanceVisualizations'])
plt.title('Is Visualization Skill Necessary?')
plt.xlabel('')

plt.subplot2grid((2,2), (1,1), colspan=1,rowspan=2)
scientist['WorkDataVisualizations'].value_counts().plot.pie(autopct='%2.0f%%',colors=sns.color_palette('Paired',10))
plt.title('Use Of Visualisations in Projects')
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
fig = plt.figure(figsize=(15,16))
gridspec.GridSpec(2,2)

plt.subplot2grid((2,2), (0,0), colspan=1,rowspan=2)
ds['Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?'].value_counts().plot.barh(width=0.9,color=sns.color_palette('viridis',5))
plt.title('Can you explain working of ML Models?')
plt.gca().invert_yaxis()
plt.subplot2grid((2,2), (0,1))
sns.countplot(scientist['JobSkillImportanceStats'])
plt.title('Stats Necessary?')
plt.xlabel('')
plt.subplot2grid((2,2), (1,1), colspan=1,rowspan=2)
ds['How do you perceive the importance of the following topics? - Being able to explain ML model outputs and/or predictions'].value_counts().plot.pie(autopct='%2.0f%%',colors=sns.color_palette('RdYlGn',4))
plt.title('Importance of Explaining ML Models')
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()
l1=[col for col in ds if col.startswith("What machine learning frameworks have you used in the past 5 years? ")]
col1=[]
col2=[]
l2=ds[l1[:-2]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
lib=pd.DataFrame({'Library':col1,'Count':col2})
lib.set_index('Library').plot.barh(width=0.9)
plt.gcf().set_size_inches(8,10)
plt.title('Libraries')
plt.xlabel('')
plt.show()
ds['How do you perceive the importance of the following topics? - Fairness and bias in ML algorithms:'].value_counts().plot.pie(autopct='%2.0f%%',shadow=True,startangle=90,colors=sns.color_palette('Paired',6))
plt.title('Importance of Fairness and Bias in ML Models')
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.gcf().set_size_inches(8,8)
plt.gca().add_artist(my_circle)
plt.ylabel('')
ds['Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?'].value_counts().plot.barh(width=0.95,color=sns.color_palette('inferno_r',15))
plt.gca().invert_yaxis()
plt.gcf().set_size_inches(8,8)
plt.title('Time spent exploring Unfair Bias')
l1=[col for col in ds if col.startswith("What do you find most difficult about ensuring that your algorithms are fair and unbiased?")]
col1=[]
col2=[]
l2=ds[l1[:]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
bias=pd.DataFrame({'reason':col1,'Count':col2})
bias.set_index('reason').plot.barh(width=0.95)
plt.gcf().set_size_inches(10,12)
plt.gca().invert_yaxis()
plt.title('Difficulty in ensuring Fairness-Bias Tradeoff',size=20)
plt.show()
l1=[col for col in ds if col.startswith("What methods do you prefer for explaining and/or interpreting decisions that are made by ML models? (Select all that apply) - ")]
col1=[]
col2=[]
l2=ds[l1[:-2]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
measure=pd.DataFrame({'Measure':col1,'Count':col2}).sort_values(by='Count')
measure.set_index('Measure').plot.barh(width=0.95,color='orange')
plt.gcf().set_size_inches(8,12)
plt.title('Measure',size=20)
plt.show()
l1=[col for col in ds if col.startswith("In what circumstances would you explore model insights and interpret your model's predictions? (Select all that apply) - ")]
col1=[]
col2=[]
l2=ds[l1[:]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
measure=pd.DataFrame({'Measure':col1,'Count':col2}).sort_values(by='Count')
measure.set_index('Measure').plot.barh(width=0.95)
plt.gcf().set_size_inches(8,8)
plt.title('Cases when exploring ML models',size=20)
plt.show()
ds['How do you perceive the importance of the following topics? - Reproducibility in data science'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,startangle=180,colors=sns.color_palette('viridis',4))
plt.title('Importance of Reproducibility in Data Science')
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.gcf().set_size_inches(7,7)
plt.gca().add_artist(my_circle)
plt.ylabel('')
l1=[col for col in ds if col.startswith("What tools and methods do you use to make your work easy to reproduce? (Select all that apply) - Selected Choice - ")]
col1=[]
col2=[]
l2=ds[l1[:]]
for i in l2.columns:
    col1.append(ds[i].value_counts().index.values[0])
    col2.append(ds[i].value_counts().values[0])
measure=pd.DataFrame({'Measure':col1,'Count':col2}).sort_values(by='Count')
measure.set_index('Measure').plot.barh(width=0.95)
plt.gcf().set_size_inches(8,8)
plt.title('Methods Used for reproducibility',size=20)
plt.show()
ds['What is your current yearly compensation (approximate $USD)?'].value_counts()[1:].plot.bar(width=0.9)
plt.gcf().set_size_inches(15,6)
plt.show()
model=ds[['What is your age (# years)?','What is the highest level of formal education that you have attained or plan to attain within the next 2 years?','In which country do you currently reside?','In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice','What is your current yearly compensation (approximate $USD)?','How long have you been writing code to analyze data?','For how many years have you used machine learning methods (at work or in school)?',]]
model.loc[np.logical_not(model['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].isin(["Master’s degree","Doctoral degree","Bachelor’s degree"])),'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?']='Others'
model.loc[(model['What is your age (# years)?'].isin(["18-21","22-24"])),'What is your age (# years)?']='<25'
model.loc[(model['What is your age (# years)?'].isin(["25-29","40-44","30-34","35-39"])),'What is your age (# years)?']='25-45'
model.loc[(model['What is your age (# years)?'].isin(["45-49","50-54","55-59","60-69","70-79",'80+'])),'What is your age (# years)?']='45+'
model.loc[model['In which country do you currently reside?'].isin(['France','Germany','Spain','Netherlands','Italy','Poland','Belgium','Portugal','Switzerland','United Kingdom of Great Britain and Northern Ireland']),'In which country do you currently reside?']='Europe'
model.loc[np.logical_not(model['In which country do you currently reside?'].isin(["United States of America","India","Europe"])),'In which country do you currently reside?']='Others'
model.loc[model['How long have you been writing code to analyze data?'].isin(['< 1 year','I have never written code but I want to learn','I have never written code and I do not want to learn']),'How long have you been writing code to analyze data?']='1'
model.loc[model['How long have you been writing code to analyze data?'].isin(['1-2 years','3-5 years','5-10 years']),'How long have you been writing code to analyze data?']='1-10'
model.loc[model['How long have you been writing code to analyze data?'].isin(['30-40 years','20-30 years','10-20 years','40+ years']),'How long have you been writing code to analyze data?']='10+'
model.loc[model['For how many years have you used machine learning methods (at work or in school)?'].isin(['I have never studied machine learning but plan to learn in the future','I have never studied machine learning and I do not plan to','< 1 year']),'For how many years have you used machine learning methods (at work or in school)?']='1'
model.loc[model['For how many years have you used machine learning methods (at work or in school)?'].isin(['1-2 years','2-3 years','3-4 years','5-10 years','4-5 years']),'For how many years have you used machine learning methods (at work or in school)?']='1-10'
model.loc[np.logical_not(model['For how many years have you used machine learning methods (at work or in school)?'].isin(['<1','1-10'])),'For how many years have you used machine learning methods (at work or in school)?']='10+'
model.loc[model['What is your current yearly compensation (approximate $USD)?'].isin(['0-10,000','10-20,000','20-30,000','40-50,000','30-40,000']),'What is your current yearly compensation (approximate $USD)?']='Low'
model.loc[model['What is your current yearly compensation (approximate $USD)?'].isin(['125-150,000','100-125,000','90-100,000','60-70,000','80-90,000','70-80,000','50-60,000']),'What is your current yearly compensation (approximate $USD)?']='Average'
model.loc[model['What is your current yearly compensation (approximate $USD)?'].isin(['150-200,000','200-250,000','250-300,000','300-400,000','500,000+','400-500,000']),'What is your current yearly compensation (approximate $USD)?']='High'
model.loc[np.logical_not(model['What is your current yearly compensation (approximate $USD)?'].isin(['Low','Average','High'])),'What is your current yearly compensation (approximate $USD)?']='Others'

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
le=LabelEncoder()
for i in model.columns:
    le=le.fit(model[i].astype(str))
    model[i]=le.transform(model[i].astype(str))

from sklearn.model_selection import cross_val_predict
X=model[model.columns.difference(['What is your current yearly compensation (approximate $USD)?'])]
Y=model['What is your current yearly compensation (approximate $USD)?']
model1=RandomForestClassifier(n_estimators=100,random_state=10)
model1.fit(X,Y)
pd.Series(model1.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.95)
plt.gcf().set_size_inches(8,8)
plt.title('Factors Influencing Salary')
