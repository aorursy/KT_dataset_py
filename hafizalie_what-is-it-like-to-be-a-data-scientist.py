

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#IMPORTING THE DATASET

data= pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

data=data.iloc[1:,:]

data.head()

plt.figure(figsize=(15,8))    

visl= sns.countplot(data.Q1.sort_values(ascending=True))

            

for p in visl.patches:

    visl.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('AGE',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('AGE DISTRIBUTION',fontsize=18)
plt.figure(figsize=(15,8))    

genvis = sns.countplot(x= data.Q2)



for p in genvis.patches:

    genvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('GENDER',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('GENDER DISTRIBUTION',fontsize=18)
x1=data.Q3.value_counts().index[:10]

y1=data.Q3.value_counts().iloc[:10]

plt.figure(figsize=(12,8))

countvis= sns.barplot(x=x1,y=y1)



for p in countvis.patches:

    countvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('COUNTRY',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('TOP 10 COUNTRIES',fontsize=18)

plt.xticks(rotation=75)

plt.show()

plt.figure(figsize=(8,10))

plt.pie(y1,labels=x1,explode=[0.1,0,0,0,0,0,0,0,0,0],shadow=True)

plt.legend(x1,bbox_to_anchor=(2,1),loc='upper right')

plt.draw()

top3=data[(data['Q3']=='India') | (data['Q3']=='United States of America') | (data['Q3']=='Brazil')]



A1=sns.catplot(y='Q1',kind='count',height=8, hue='Q3', data=top3,order=top3['Q1'].value_counts().index)

plt.xlabel('COUNT',fontsize=15)

plt.ylabel('AGE GROUP',fontsize=15)

plt.title('AGE vs COUNTRY', fontsize=18, weight='bold' )



plt.show()
plt.figure(figsize=(12,8))

edvis = sns.countplot(data.Q4)

for p in edvis.patches:

    edvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('EDUCATION',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('EDUCATION OF RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation=75)

plt.show()
A2=sns.catplot(y='Q4',kind='count',height=8, hue='Q3', data=top3,order=top3['Q4'].value_counts().index)



plt.xlabel('COUNT',fontsize=15)

plt.ylabel('EDUCATION',fontsize=15)

plt.title('COUNTRY VS EDUCATION', fontsize=18, weight='bold' )





plt.show()

plt.figure(figsize=(12,8))

jobvis = sns.countplot(data.Q5.dropna())

for p in jobvis.patches:

    jobvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('JOB',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('JOB PROFILE',fontsize=18, weight= 'bold')

plt.xticks(rotation=90)

plt.show()

A3=sns.catplot(y='Q5',kind='count',height=8, hue='Q3', data=top3,order=top3['Q5'].value_counts().index)



plt.xlabel('COUNT',fontsize=15)

plt.ylabel('JOB',fontsize=15)

plt.title('COUNTRY vs JOB', fontsize=18, weight='bold')



plt.show()

plt.figure(figsize=(10,8))

jobvis = sns.countplot(data.Q6.dropna().sort_values(ascending = True))



for p in jobvis.patches:

    jobvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('COMPANY SIZE',fontsize= 15)

plt.ylabel('COUNT',fontsize = 15)

plt.xticks(rotation=90)

plt.title('COMPANY SIZE',fontsize = 18,weight='bold')

plt.show()
A6=sns.catplot(y='Q6',kind='count',height=8, hue='Q3', data=top3,order=top3['Q6'].value_counts().index)

plt.xlabel('COUNTRIES',fontsize=15)

plt.ylabel('No.OF EMPLOYEES',fontsize=15)

plt.title('COMPANY SIZE vs COUNTRY', fontsize=18, weight='bold' )



plt.show()
plt.figure(figsize=(14,8))

salvis = sns.countplot(data.Q10.dropna().sort_values(ascending = True))



for p in salvis.patches:

    salvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel('SALARY',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('SALARY OF RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation=90)

plt.show()



plt.figure(figsize=(12,6))

A10=sns.catplot(y='Q10',kind='count',height=8, hue='Q3', data=top3,order=top3['Q10'].value_counts().index)



plt.title('SALARY vs COUNTRY', fontsize=15, weight='bold' )

plt.subplots_adjust(top=0.85)



plt.show()
plt.figure(figsize=(12,8))

mlvis = sns.countplot(data.Q11.dropna().sort_values(ascending = True))



for p in mlvis.patches:

    mlvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel('MONEY SPENT',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('MONEY SPENT ON ML',fontsize=18,weight='bold')

plt.xticks(rotation=90)

plt.show()

plt.figure(figsize=(12,6))

A10=sns.catplot(y='Q11',kind='count',height=8, hue='Q3', data=top3,order=top3['Q11'].value_counts().index)



plt.title('Company Size', fontsize=15, weight='bold' )

plt.subplots_adjust(top=0.85)



plt.show()

ds_media={}



ds_media['Twitter']=data.Q12_Part_1.value_counts().sum()

ds_media['Hacker News']=data.Q12_Part_2.value_counts().sum()

ds_media['Reddit']=data.Q12_Part_3.value_counts().sum()

ds_media['Kaggle']=data.Q12_Part_4.value_counts().sum()

ds_media['Course Forums']=data.Q12_Part_5.value_counts().sum()

ds_media['YouTube']=data.Q12_Part_6.value_counts().sum()

ds_media['Podcasts']=data.Q12_Part_7.value_counts().sum()

ds_media['Blogs']=data.Q12_Part_8.value_counts().sum()

ds_media['Journal Publications']=data.Q12_Part_9.value_counts().sum()

ds_media['Slack']=data.Q12_Part_10.value_counts().sum()

ds_media['Other']=data.Q12_Part_11.value_counts().sum()





ds_media = pd.DataFrame.from_dict(ds_media,orient='index',columns=['count'])

ds_media.reset_index(inplace=True)





plt.figure(figsize = (10,7))

ds_media_vis = sns.barplot(x= 'index',y='count', data = ds_media)

plt.xlabel('MEDIA SOURCES',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('FAVOURITE MEDIA SOURCES',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ds_media_vis.patches:

    ds_media_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
ds_plat={}



ds_plat['Udacity']=data.Q13_Part_1.value_counts().sum()

ds_plat['Coursera']=data.Q13_Part_2.value_counts().sum()

ds_plat['edX']=data.Q13_Part_3.value_counts().sum()

ds_plat['DataCamp']=data.Q13_Part_4.value_counts().sum()

ds_plat['DataQuest']=data.Q13_Part_5.value_counts().sum()

ds_plat['Kaggle Courses']=data.Q13_Part_6.value_counts().sum()

ds_plat['fast.ai']=data.Q13_Part_7.value_counts().sum()

ds_plat['Udemy']=data.Q13_Part_8.value_counts().sum()

ds_plat['LinkedIn Learning']=data.Q13_Part_9.value_counts().sum()

ds_plat['University Courses']=data.Q13_Part_10.value_counts().sum()

ds_plat['None']=data.Q13_Part_11.value_counts().sum()

ds_plat['Other']=data.Q13_Part_12.value_counts().sum()



ds_plat = pd.DataFrame.from_dict(ds_plat,orient='index',columns=['count'])

ds_plat.reset_index(inplace=True)



plt.figure(figsize = (10,7))

ds_plat = sns.barplot(x= 'index',y='count', data = ds_plat,palette='rocket')

plt.xlabel('PLATFORM',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('DS PLATFORM USED',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ds_plat.patches:

    ds_plat.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

plt.figure(figsize=(12,8))

antoolvis = sns.countplot(data.Q14.dropna().sort_values(ascending = True))



for p in antoolvis.patches:

    antoolvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('ANALYSIS TOOL',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('DS ANALYSIS TOOL USED',fontsize=18,weight='bold')

plt.xticks(rotation=75)



plt.show()

plt.figure(figsize=(12,8))

codexvis = sns.countplot(data.Q15.dropna().sort_values(ascending = True),palette='terrain')



for p in codexvis.patches:

    codexvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel('EXPERIENCE',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('CODING EXPERIENCE OF RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation=90)

plt.show()

ds_IDE={}



ds_IDE['Jupyter']=data.Q16_Part_1.value_counts().sum()

ds_IDE['RStudio']=data.Q16_Part_2.value_counts().sum()

ds_IDE['PyCharm']=data.Q16_Part_3.value_counts().sum()

ds_IDE['Atom']=data.Q16_Part_4.value_counts().sum()

ds_IDE['MATLAB']=data.Q16_Part_5.value_counts().sum()

ds_IDE['Visual Studio']=data.Q16_Part_6.value_counts().sum()

ds_IDE['Spyder']=data.Q16_Part_7.value_counts().sum()

ds_IDE['Vim/Emacs']=data.Q16_Part_8.value_counts().sum()

ds_IDE['Notepad++']=data.Q16_Part_9.value_counts().sum()

ds_IDE['Sublime Text']=data.Q16_Part_10.value_counts().sum()

ds_IDE['None']=data.Q16_Part_11.value_counts().sum()

ds_IDE['Other']=data.Q16_Part_12.value_counts().sum()



ds_IDE = pd.DataFrame.from_dict(ds_IDE,orient='index',columns=['count'])

ds_IDE.reset_index(inplace=True)





plt.figure(figsize = (10,8))

ds_IDE = sns.barplot(x= 'index',y='count', data = ds_IDE,palette='ocean')

plt.xlabel('IDEs',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('DS IDEs USED',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ds_IDE.patches:

    ds_IDE.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

ds_note={}

ds_note['Kaggle Notebooks']=data.Q17_Part_1.value_counts().sum()

ds_note['Google Colab']=data.Q17_Part_2.value_counts().sum()

ds_note['MS Azure Notebook']=data.Q17_Part_3.value_counts().sum()

ds_note['Google Cloud Notebooks']=data.Q17_Part_4.value_counts().sum()

ds_note['Paperspace/Gradient']=data.Q17_Part_5.value_counts().sum()

ds_note['FloydHub']=data.Q17_Part_6.value_counts().sum()

ds_note['Binder/JupyterHub']=data.Q17_Part_7.value_counts().sum()

ds_note['IBM Watson Studio']=data.Q17_Part_8.value_counts().sum()

ds_note['Code Ocean']=data.Q17_Part_9.value_counts().sum()

ds_note['AWS Notebook']=data.Q17_Part_10.value_counts().sum()

ds_note['None']=data.Q17_Part_11.value_counts().sum()

ds_note['Other']=data.Q17_Part_12.value_counts().sum()







ds_note = pd.DataFrame.from_dict(ds_note,orient='index',columns=['count'])

ds_note.reset_index(inplace=True)





plt.figure(figsize = (10,7))

ds_note = sns.barplot(x= 'index',y='count', data = ds_note,palette='inferno')

plt.xlabel('NOTEBOOKS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('DS NOTEBOOKS USED BY RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ds_note.patches:

    ds_note.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

ds_lan={}

ds_lan['Python']=data.Q18_Part_1.value_counts().sum()

ds_lan['R']=data.Q18_Part_2.value_counts().sum()

ds_lan['SQL']=data.Q18_Part_3.value_counts().sum()

ds_lan['C']=data.Q18_Part_4.value_counts().sum()

ds_lan['C++']=data.Q18_Part_5.value_counts().sum()

ds_lan['Java']=data.Q18_Part_6.value_counts().sum()

ds_lan['JavaScript']=data.Q18_Part_7.value_counts().sum()

ds_lan['TypeScript']=data.Q18_Part_8.value_counts().sum()

ds_lan['Bash']=data.Q18_Part_9.value_counts().sum()

ds_lan['MATLAB']=data.Q18_Part_10.value_counts().sum()

ds_lan['None']=data.Q18_Part_11.value_counts().sum()

ds_lan['Other']=data.Q18_Part_12.value_counts().sum()







ds_lan = pd.DataFrame.from_dict(ds_lan,orient='index',columns=['count'])

ds_lan.reset_index(inplace=True)





plt.figure(figsize = (10,8))

ds_lan = sns.barplot(x= 'index',y='count', data = ds_lan,palette='hot')

plt.xlabel('LANGUAGE',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('PROGRAMMING LANGUAGE USED BY RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ds_lan.patches:

    ds_lan.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

plt.figure(figsize=(12,8))

lanvis = sns.countplot(data.Q19.dropna().sort_values(ascending = True))



for p in lanvis.patches:

    lanvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel('LANGUAGE',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('LANGUAGE PREFERENCE OF RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation=90)

plt.show()

ds_tool={}

ds_tool['Ggplot/ggplot2']=data.Q20_Part_1.value_counts().sum()

ds_tool['Matplotlib']=data.Q20_Part_2.value_counts().sum()

ds_tool['Altair']=data.Q20_Part_3.value_counts().sum()

ds_tool['Shiny']=data.Q20_Part_4.value_counts().sum()

ds_tool['D3.js']=data.Q20_Part_5.value_counts().sum()

ds_tool['Plotly/Plotly Express']=data.Q20_Part_6.value_counts().sum()

ds_tool['Bokeh']=data.Q20_Part_7.value_counts().sum()

ds_tool['Seaborn']=data.Q20_Part_8.value_counts().sum()

ds_tool['Geoplotlib']=data.Q20_Part_9.value_counts().sum()

ds_tool['Leaflet/Folium']=data.Q20_Part_10.value_counts().sum()

ds_tool['None']=data.Q20_Part_11.value_counts().sum()

ds_tool['Other']=data.Q20_Part_12.value_counts().sum()







ds_tool = pd.DataFrame.from_dict(ds_tool,orient='index',columns=['count'])

ds_tool.reset_index(inplace=True)





plt.figure(figsize = (10,7))

ds_tool = sns.barplot(x= 'index',y='count', data = ds_tool,palette='icefire')

plt.xlabel('VISUALIZATION TOOL',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('VISUALIZATION TOOLS USED BY RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation = 80)

for p in ds_tool.patches:

    ds_tool.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

ds_hard={}



ds_hard['CPUs']=data.Q21_Part_1.value_counts().sum()

ds_hard['GPUs']=data.Q21_Part_2.value_counts().sum()

ds_hard['TPUs']=data.Q21_Part_3.value_counts().sum()

ds_hard['None/ I do not know']=data.Q21_Part_4.value_counts().sum()

ds_hard['Other']=data.Q21_Part_5.value_counts().sum()



ds_hard = pd.DataFrame.from_dict(ds_hard,orient='index',columns=['count'])

ds_hard.reset_index(inplace=True)





plt.figure(figsize = (10,8))

ds_hard = sns.barplot(x= 'index',y='count', data = ds_hard,palette='tab20')

plt.xlabel('HARDWARE',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('HARDWARE USED BY RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ds_hard.patches:

    ds_hard.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

plt.figure(figsize=(12,8))

tpuvis = sns.countplot(data.Q22.dropna().sort_values(ascending = True))



for p in tpuvis.patches:

    tpuvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel('USE OF TPU',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('USAGE OF TPU BY RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation=90)

plt.show()

plt.figure(figsize=(12,8))

mlexpvis = sns.countplot(data.Q23.dropna().sort_values(ascending = True),palette='PuBuGn_r')



for p in mlexpvis.patches:

    mlexpvis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel('EXPERIENCE',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('ML EXPERIENCE OF RESPONDENTS',fontsize=18,weight='bold')

plt.xticks(rotation=90)

plt.show()

ml_algo = {}

ml_algo['Linear or logistic Regression']=data.Q24_Part_1.value_counts().sum()

ml_algo['Decision Trees or Random Forests']=data.Q24_Part_2.value_counts().sum()

ml_algo['Gradient Boosting Machines']=data.Q24_Part_3.value_counts().sum()

ml_algo['Bayesian Approaches']=data.Q24_Part_4.value_counts().sum()

ml_algo['Evolutionary Approaches']=data.Q24_Part_5.value_counts().sum()

ml_algo['Dense Neural Networks']=data.Q24_Part_6.value_counts().sum()

ml_algo['Convolutional Neural Networks']=data.Q24_Part_7.value_counts().sum()

ml_algo['Generative Adversarial Networks']=data.Q24_Part_8.value_counts().sum()

ml_algo['Recurrent Neural Networks']=data.Q24_Part_9.value_counts().sum()

ml_algo[' Transformer Networks']=data.Q24_Part_10.value_counts().sum()

ml_algo['None']=data.Q24_Part_11.value_counts().sum()

ml_algo['Other']=data.Q24_Part_12.value_counts().sum()



ml_algo = pd.DataFrame.from_dict(ml_algo,orient='index',columns=['count'])

ml_algo.reset_index(inplace=True)



plt.figure(figsize = (12,9))

ml_algo_vis = sns.barplot(x= 'index',y='count', data = ml_algo,palette='mako_r')

plt.xlabel('ML ALGORITHMS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('ML ALGORITHMS USING REGULARLY',fontsize=18,weight='bold')



plt.xticks(rotation = 75)

for p in ml_algo_vis.patches:

    ml_algo_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

ml_tool = {}

ml_tool['Automated data augmentation']=data.Q25_Part_1.value_counts().sum()

ml_tool['Automated feature engineering/selection']=data.Q25_Part_2.value_counts().sum()

ml_tool['Automated model selection']=data.Q25_Part_3.value_counts().sum()

ml_tool['Automated model architecture searches']=data.Q25_Part_4.value_counts().sum()

ml_tool['Automated hyperparameter tuning']=data.Q25_Part_5.value_counts().sum()

ml_tool['Automation of full ML pipelines']=data.Q25_Part_6.value_counts().sum()

ml_tool['None']=data.Q25_Part_7.value_counts().sum()

ml_tool['Other']=data.Q25_Part_8.value_counts().sum()





ml_tool = pd.DataFrame.from_dict(ml_tool,orient='index',columns=['count'])

ml_tool.reset_index(inplace=True)



plt.figure(figsize = (10,7))

ml_tool_vis = sns.barplot(x= 'index',y='count', data = ml_tool,palette='Spectral_r')

plt.xlabel('ML TOOLS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('ML TOOLS USING REGULARLY',fontsize=18,weight='bold')

plt.xticks(rotation = 75)

for p in ml_tool_vis.patches:

    ml_tool_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
cv_method = {}

cv_method['General purpose image/video tools']=data.Q26_Part_1.value_counts().sum()

cv_method['Image segmentation methods']=data.Q26_Part_2.value_counts().sum()

cv_method['Object detection methods']=data.Q26_Part_3.value_counts().sum()

cv_method['Image classification and other general purpose networks']=data.Q26_Part_4.value_counts().sum()

cv_method['Generative Networks']=data.Q26_Part_5.value_counts().sum()

cv_method['None']=data.Q26_Part_6.value_counts().sum()

cv_method['Other']=data.Q26_Part_7.value_counts().sum()

cv_method = pd.DataFrame.from_dict(cv_method,orient='index',columns=['count'])

cv_method.reset_index(inplace=True)



plt.figure(figsize = (10,7))

cv_method_vis = sns.barplot(x= 'index',y='count', data = cv_method,palette='bone_r')

plt.xlabel('CV METHODS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('CV METHODS USING REGULARLY',fontsize=18,weight='bold')

plt.xticks(rotation = 75)

for p in cv_method_vis.patches:



    cv_method_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 



               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()

nlp_method = {}



nlp_method['Word embeddings/vectors']=data.Q27_Part_1.value_counts().sum()



nlp_method['Encoder-decorder models']=data.Q27_Part_2.value_counts().sum()



nlp_method['Contextualized embeddings']=data.Q27_Part_3.value_counts().sum()



nlp_method['Transformer language models']=data.Q27_Part_4.value_counts().sum()



nlp_method['None']=data.Q27_Part_5.value_counts().sum()



nlp_method['Other']=data.Q27_Part_6.value_counts().sum()







nlp_method = pd.DataFrame.from_dict(nlp_method,orient='index',columns=['count'])



nlp_method.reset_index(inplace=True)











plt.figure(figsize = (10,7))



nlp_method_vis = sns.barplot(x= 'index',y='count', data = nlp_method,palette='PRGn_r')

plt.xlabel('NLP METHODS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('NLP METHODS USING REGULARLY',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in nlp_method_vis.patches:

    nlp_method_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 



               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()



ml_frame = {}



ml_frame['Scikit-learn']=data.Q28_Part_1.value_counts().sum()



ml_frame['TensorFlow']=data.Q28_Part_2.value_counts().sum()



ml_frame['Keras']=data.Q28_Part_3.value_counts().sum()



ml_frame['RandomForest']=data.Q28_Part_4.value_counts().sum()



ml_frame['Xgboost']=data.Q28_Part_5.value_counts().sum()



ml_frame['Pytorch']=data.Q28_Part_6.value_counts().sum()



ml_frame['Caret']=data.Q28_Part_7.value_counts().sum()



ml_frame['LightGBM']=data.Q28_Part_8.value_counts().sum()



ml_frame['Spark MLib']=data.Q28_Part_9.value_counts().sum()



ml_frame['Fast.ai']=data.Q28_Part_10.value_counts().sum()



ml_frame['None']=data.Q28_Part_11.value_counts().sum()



ml_frame['Other']=data.Q28_Part_12.value_counts().sum()







ml_frame = pd.DataFrame.from_dict(ml_frame,orient='index',columns=['count'])



ml_frame.reset_index(inplace=True)











plt.figure(figsize = (10,7))



ml_frame_vis = sns.barplot(x= 'index',y='count', data = ml_frame,palette='gist_earth_r')

plt.xlabel('ML FRAMEWORKS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('ML FRAMEWORKS USING REGULARLY',fontsize=18,weight='bold')



plt.xticks(rotation = 90)



for p in ml_frame_vis.patches:



    ml_frame_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 



               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.show()





cloud_plat = {}



cloud_plat['Google Cloud Platform (GCP)']=data.Q29_Part_1.value_counts().sum()



cloud_plat['Amazon Web Services (AWS)']=data.Q29_Part_2.value_counts().sum()



cloud_plat['Microsoft Azure']=data.Q29_Part_3.value_counts().sum()



cloud_plat['IBM Cloud']=data.Q29_Part_4.value_counts().sum()



cloud_plat['Alibaba Cloud']=data.Q29_Part_5.value_counts().sum()



cloud_plat['Salesforce Cloud']=data.Q29_Part_6.value_counts().sum()



cloud_plat['Oracle Cloud']=data.Q29_Part_7.value_counts().sum()



cloud_plat['SAP Cloud']=data.Q29_Part_8.value_counts().sum()



cloud_plat['VMware Cloud']=data.Q29_Part_9.value_counts().sum()



cloud_plat['Red Hat Cloud']=data.Q29_Part_10.value_counts().sum()



cloud_plat['None']=data.Q29_Part_11.value_counts().sum()



cloud_plat['Other']=data.Q29_Part_12.value_counts().sum()











cloud_plat = pd.DataFrame.from_dict(cloud_plat,orient='index',columns=['count'])



cloud_plat.reset_index(inplace=True)











plt.figure(figsize = (10,7))



cloud_plat_vis = sns.barplot(x= 'index',y='count', data = cloud_plat,palette='BuPu_r')

plt.xlabel('CLOUD PLATFORMS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('CLOUD COMPUTING PLATFORMS',fontsize=18,weight='bold')



plt.xticks(rotation = 90)



for p in cloud_plat_vis.patches:



    cloud_plat_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 



               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.show()



cloud_prod = {}



cloud_prod['AWS Elastic Compute Cloud (EC2)']=data.Q30_Part_1.value_counts().sum()

cloud_prod['Google Compute Engine (GCE)']=data.Q30_Part_2.value_counts().sum()

cloud_prod['AWS Lambda']=data.Q30_Part_3.value_counts().sum()

cloud_prod['Azure Virtual Machines']=data.Q30_Part_4.value_counts().sum()

cloud_prod['Google App Engine']=data.Q30_Part_5.value_counts().sum()

cloud_prod['Google Cloud Functions']=data.Q30_Part_6.value_counts().sum()

cloud_prod['AWS Elastic Beanstalk']=data.Q30_Part_7.value_counts().sum()

cloud_prod['Google Kubernetes Engine']=data.Q30_Part_8.value_counts().sum()

cloud_prod['AWS Batch']=data.Q30_Part_9.value_counts().sum()

cloud_prod['Azure Container Service']=data.Q30_Part_10.value_counts().sum()

cloud_prod['None']=data.Q30_Part_11.value_counts().sum()

cloud_prod['Other']=data.Q30_Part_12.value_counts().sum()







cloud_prod = pd.DataFrame.from_dict(cloud_prod,orient='index',columns=['count'])



cloud_prod.reset_index(inplace=True)











plt.figure(figsize = (10,7))



cloud_prod_vis = sns.barplot(x= 'index',y='count', data = cloud_prod,palette='Oranges_r')

plt.xlabel('CLOUD COMPUTING PRODUCTS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('CLOUD COMPUTING PRODUCTS USING REGULARLY ',fontsize=18,weight='bold')



plt.xticks(rotation = 90)



for p in cloud_prod_vis.patches:



    cloud_prod_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 



               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.show()



big_prod = {}

big_prod['Google BigQuery']=data.Q31_Part_1.value_counts().sum()

big_prod['AWS Redshift']=data.Q31_Part_2.value_counts().sum()

big_prod['Databricks']=data.Q31_Part_3.value_counts().sum()

big_prod['AWS Elastic MapReduce']=data.Q31_Part_4.value_counts().sum()

big_prod['Teradata']=data.Q31_Part_5.value_counts().sum()

big_prod['Microsoft Analysis Services']=data.Q31_Part_6.value_counts().sum()



big_prod['Google Cloud Dataflow']=data.Q31_Part_7.value_counts().sum()



big_prod['AWS Athena']=data.Q31_Part_8.value_counts().sum()



big_prod['AWS Kinesis']=data.Q31_Part_9.value_counts().sum()



big_prod['Google Cloud Pub/Sub']=data.Q31_Part_10.value_counts().sum()



big_prod['None']=data.Q31_Part_11.value_counts().sum()



big_prod['Other']=data.Q31_Part_12.value_counts().sum()







big_prod = pd.DataFrame.from_dict(big_prod,orient='index',columns=['count'])



big_prod.reset_index(inplace=True)











plt.figure(figsize = (10,7))



big_prod_vis = sns.barplot(x= 'index',y='count', data = big_prod,palette='rainbow_r')

plt.xlabel('BIG DATA PRODUCTS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('BIG DATA PRODUCTS USING REGULARLY ',fontsize=18,weight='bold')



plt.xticks(rotation = 90)



for p in big_prod_vis.patches:



    big_prod_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 



               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.show()

ml_prod = {}

ml_prod['SAS']=data.Q32_Part_1.value_counts().sum()

ml_prod['Cloudera']=data.Q32_Part_2.value_counts().sum()

ml_prod['Azure Machine Learning Studio']=data.Q32_Part_3.value_counts().sum()

ml_prod['Google Cloud Machine Learning Engine']=data.Q32_Part_4.value_counts().sum()

ml_prod['Google Cloud Visio']=data.Q32_Part_5.value_counts().sum()

ml_prod['Google Cloud Speech-to-Text']=data.Q32_Part_6.value_counts().sum()

ml_prod['Google Cloud Natural Language']=data.Q32_Part_7.value_counts().sum()

ml_prod['RapidMiner']=data.Q32_Part_8.value_counts().sum()

ml_prod['Google Cloud Translation']=data.Q32_Part_9.value_counts().sum()

ml_prod['Amazon SageMaker']=data.Q32_Part_10.value_counts().sum()

ml_prod['None']=data.Q32_Part_11.value_counts().sum()

ml_prod['Other']=data.Q32_Part_12.value_counts().sum()



ml_prod = pd.DataFrame.from_dict(ml_prod,orient='index',columns=['count'])

ml_prod.reset_index(inplace=True)



plt.figure(figsize = (10,7))

ml_prod_vis = sns.barplot(x= 'index',y='count', data = ml_prod,palette='Set1_r')

plt.xlabel('ML PRODUCTS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('ML  PRODUCTS USING REGULARLY ',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ml_prod_vis.patches:

    ml_prod_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
ml_par = {}

ml_par['Google AutoML']=data.Q33_Part_1.value_counts().sum()

ml_par['H20 Driverless AI']=data.Q33_Part_2.value_counts().sum()

ml_par['Databricks AutoML']=data.Q33_Part_3.value_counts().sum()

ml_par['DataRobot AutoML']=data.Q33_Part_4.value_counts().sum()

ml_par['Tpot']=data.Q33_Part_5.value_counts().sum()

ml_par['Auto-Keras']=data.Q33_Part_6.value_counts().sum()

ml_par['Auto-Sklearn']=data.Q33_Part_7.value_counts().sum()

ml_par['Auto_ml']=data.Q33_Part_8.value_counts().sum()

ml_par['Xcessiv']=data.Q33_Part_9.value_counts().sum()

ml_par['MLbox']=data.Q33_Part_10.value_counts().sum()

ml_par['None']=data.Q33_Part_11.value_counts().sum()

ml_par['Other']=data.Q33_Part_12.value_counts().sum()



ml_par = pd.DataFrame.from_dict(ml_par,orient='index',columns=['count'])

ml_par.reset_index(inplace=True)



plt.figure(figsize = (10,7))

ml_par_vis = sns.barplot(x= 'index',y='count', data = ml_par,palette='CMRmap_r')

plt.xlabel('AUTOMATED ML TOOLS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('AUTOMATED ML TOOLS USING REGULARLY ',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in ml_par_vis.patches:

    ml_par_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
db_prod = {}

db_prod['MySQL']=data.Q34_Part_1.value_counts().sum()

db_prod['PostgresSQL']=data.Q34_Part_2.value_counts().sum()

db_prod['SQLite']=data.Q34_Part_3.value_counts().sum()

db_prod['Microsoft SQL Server']=data.Q34_Part_4.value_counts().sum()

db_prod['Oracle Database']=data.Q34_Part_5.value_counts().sum()

db_prod['Microsoft Access']=data.Q34_Part_6.value_counts().sum()

db_prod['AWS Relational Database Service']=data.Q34_Part_7.value_counts().sum()

db_prod['AWS DynamoDB']=data.Q34_Part_8.value_counts().sum()

db_prod['Azure SQL Database']=data.Q34_Part_9.value_counts().sum()

db_prod['Google Cloud SQL']=data.Q34_Part_10.value_counts().sum()

db_prod['None']=data.Q34_Part_11.value_counts().sum()

db_prod['Other']=data.Q34_Part_12.value_counts().sum()



db_prod = pd.DataFrame.from_dict(db_prod,orient='index',columns=['count'])

db_prod.reset_index(inplace=True)



plt.figure(figsize = (10,7))

db_prod_vis = sns.barplot(x= 'index',y='count', data = db_prod)

plt.xlabel('DATABASE PRODUCTS',fontsize=15)

plt.ylabel('COUNT',fontsize=15)

plt.title('DATABASE PRODUCTS USING REGULARLY ',fontsize=18,weight='bold')

plt.xticks(rotation = 90)

for p in db_prod_vis.patches:

    db_prod_vis.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
