# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import pylab
color = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/multipleChoiceResponses.csv", encoding = "ISO-8859-1")
# Any results you write to the current directory are saved as output.

dft = df.transpose()
dfs = df[df.Q7 != 'I am a student']
tot_count  = 23859
df.head()
questions = df.iloc[0]
df = df[1:]

#questions
df = df.replace({'Masterâs degree':"Master's",'Bachelorâs degree': "Bachelor's", 'Doctoral degree':'Doctoral','Some college/university study without earning a bachelorâs degree':"Some other degree",'Professional degree':'Professional','No formal education past high school':'No Education','I prefer not to answer':'Unknown'})
df = df.replace({'United States of America':'USA','United Kingdom of Great Britain and Northern Ireland': 'UK & Ireland', 'Iran, Islamic Republic of...':'Iran','Hong Kong (S.A.R.)':'Hong Kong','I do not wish to disclose my location':'Mysterious Location'})
df = df.replace({'Computer science (software engineering, etc.)':"Computer Sci",'Engineering (non-computer focused)': "Engineering", 'Mathematics or statistics':'Math & Stats','A business discipline (accounting, economics, finance, etc.)':"Buisness",'Physics or astronomy':'Phy & Astronomy','Information technology, networking, or system administration':'IT & Networking','Medical or life sciences (biology, chemistry, medicine, etc.)':'Medical','Social sciences (anthropology, psychology, sociology, etc.)':"Social Sci",'Humanities (history, literature, philosophy, etc.)':'Humanities','Environmental science or geology':'Environmental Sci','I never declared a major':"No Major",'Fine arts or performing arts':'Fine Arts'})

dfs = dfs.replace({'Masterâs degree':"Master's",'Bachelorâs degree': "Bachelor's", 'Doctoral degree':'Doctoral','Some college/university study without earning a bachelorâs degree':"Some other degree",'Professional degree':'Professional','No formal education past high school':'No Education','I prefer not to answer':'Unknown'})
dfs = dfs.replace({'United States of America':'USA','United Kingdom of Great Britain and Northern Ireland': 'UK & Ireland', 'Iran, Islamic Republic of...':'Iran','Hong Kong (S.A.R.)':'Hong Kong','I do not wish to disclose my location':'Mysterious Location'})
dfs = dfs.replace({'Computer science (software engineering, etc.)':"Computer Sci",'Engineering (non-computer focused)': "Engineering", 'Mathematics or statistics':'Math & Stats','A business discipline (accounting, economics, finance, etc.)':"Buisness",'Physics or astronomy':'Phy & Astronomy','Information technology, networking, or system administration':'IT & Networking','Medical or life sciences (biology, chemistry, medicine, etc.)':'Medical','Social sciences (anthropology, psychology, sociology, etc.)':"Social Sci",'Humanities (history, literature, philosophy, etc.)':'Humanities','Environmental science or geology':'Environmental Sci','I never declared a major':"No Major",'Fine arts or performing arts':'Fine Arts'})

dft = dft.replace({'Masterâs degree':"Master's",'Bachelorâs degree': "Bachelor's", 'Doctoral degree':'Doctoral','Some college/university study without earning a bachelorâs degree':"Some other degree",'Professional degree':'Professional','No formal education past high school':'No Education','I prefer not to answer':'Unknown'})
dft = dft.replace({'United States of America':'USA','United Kingdom of Great Britain and Northern Ireland': 'UK & Ireland', 'Iran, Islamic Republic of...':'Iran','Hong Kong (S.A.R.)':'Hong Kong','I do not wish to disclose my location':'Mysterious Location'})
dft = dft.replace({'Computer science (software engineering, etc.)':"Computer Sci",'Engineering (non-computer focused)': "Engineering", 'Mathematics or statistics':'Math & Stats','A business discipline (accounting, economics, finance, etc.)':"Buisness",'Physics or astronomy':'Phy & Astronomy','Information technology, networking, or system administration':'IT & Networking','Medical or life sciences (biology, chemistry, medicine, etc.)':'Medical','Social sciences (anthropology, psychology, sociology, etc.)':"Social Sci",'Humanities (history, literature, philosophy, etc.)':'Humanities','Environmental science or geology':'Environmental Sci','I never declared a major':"No Major",'Fine arts or performing arts':'Fine Arts'})

df['Q1'].value_counts()/tot_count*100
labels = 'Male', 'Prefer not to say', 'Female', 'Prefer to self-describe'
sizes = [81.436774,1.425039, 16.807075, 0.331112]
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue']
explode = (0, 0, 0.1, 0)  # explode 1st slice
font = {'weight':'normal','size'   : 10} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df11 = df[['Q3','Q1']][df[['Q3','Q1']].Q1 != 'Prefer not to say']
df11 = df11[df11.Q1 != 'Prefer to self-describe']
dfGC = df11.groupby(["Q3", "Q1"]).size().reset_index()
dfGC.columns = ['Countries', 'Female', 'Male']
for i in range(0,len(dfGC),2):
    dfGC['Female'][i] = dfGC['Male'][i]
    dfGC['Male'][i] = dfGC['Male'][i+1]

dfGC = dfGC[dfGC.index % 2 != 1]
dfGC['Total'] = dfGC['Female'] + dfGC['Male'] 
dfGCn = dfGC.sort_values(['Total'], ascending = False)[:20]
dfGCn
dfGCn = dfGCn.melt('Countries', var_name='Gender',  value_name='Count')[:40]
ax = sns.barplot(x="Countries", y="Count", hue='Gender', data=dfGCn)
plt.xlabel('Age', fontsize=25)
plt.ylabel('Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=1.5)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q2'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q2'].value_counts().index, y=df['Q2'].value_counts()/tot_count*100)
sns.set_style("white")
plt.xlabel('Age', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=0)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df21 = df[['Q2','Q1']]
dfGAP = df21.groupby(["Q2", "Q1"]).size().reset_index()
dfGAP.columns = ['Age', 'Female', 'Male']
dfGAP
for i in range(0,len(dfGAP),4):
    dfGAP['Female'][i] = dfGAP['Male'][i]
    dfGAP['Male'][i] = dfGAP['Male'][i+1]  
    
dfGAP = dfGAP[dfGAP.index % 4 == 0]
dfGAP['Total'] = dfGAP['Female']+ dfGAP['Male']
dfGAP['Female'] = dfGAP['Female']/dfGAP['Total']*100
dfGAP['Male'] = dfGAP['Male']/dfGAP['Total']*100
dfGAP
sns.set(font_scale=1.5)
ax = dfGAP[['Age', 'Female', 'Male']].set_index('Age').plot(kind='bar', stacked=True)
plt.xlabel('Age', fontsize= 25)
plt.ylabel('Users', fontsize=25)
plt.xticks(rotation=0)
sns.set(rc={'figure.figsize':(20,12)})
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q3'].value_counts()/tot_count*100
x = df['Q3'].value_counts().index
y = df['Q3'].value_counts()/tot_count*100
df3 = pd.DataFrame(list(zip(x,y)),columns=['Countries', '% Users'])
df3 = df3.drop(12)
sns.barplot(x = df3['Countries'], y = df3['% Users'])
plt.xlabel('Countries', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=1.8)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q4'].value_counts()/tot_count*100
x = df['Q4'].value_counts().index
y = df['Q4'].value_counts()/tot_count*100
df4 = pd.DataFrame(list(zip(x,y)),columns=['Countries', '% Users'])
df4
labels = "Master's","No Education","Bachelor's",'Unknown',"Doctoral","Some degree but not Bachelor's",'Professional'
sizes = [45.496458, 0.972379, 29.686911, 1.445995,14.070162, 4.052978, 2.510583]
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue','red','blue','lightyellow']
explode = (0, 0, 0.1, 0.1,0.1,0.1,0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=False, startangle=140)
 
plt.axis('equal')
plt.show()
df41 = df[['Q4','Q1']]
dfGE = df41.groupby(["Q4", "Q1"]).size().reset_index()
dfGE.columns = ['Education', 'Female', 'Male']
for i in range(0,len(dfGE),4):
    dfGE['Female'][i] = dfGE['Male'][i]
    dfGE['Male'][i] = dfGE['Male'][i+1]    
dfGE = dfGE[dfGE.index % 4 == 0]
dfGE
dfGE = dfGE.melt('Education', var_name='Gender',  value_name='Count')
ax = sns.barplot(x="Education", y="Count", hue='Gender', data=dfGE)
plt.xlabel('Age', fontsize=25)
plt.ylabel('Users', fontsize=25)
plt.xticks(rotation=0)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=1.5)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df41 = df[['Q4','Q1']]
dfGEP = df41.groupby(["Q4", "Q1"]).size().reset_index()
dfGEP.columns = ['Education', 'Female', 'Male']
for i in range(0,len(dfGEP),4):
    dfGEP['Female'][i] = dfGEP['Male'][i]
    dfGEP['Male'][i] = dfGEP['Male'][i+1]  
    
dfGEP = dfGEP[dfGEP.index % 4 == 0]
dfGEP['Total'] = dfGEP['Female']+ dfGEP['Male']
dfGEP['Female'] = dfGEP['Female']/dfGEP['Total']*100
dfGEP['Male'] = dfGEP['Male']/dfGEP['Total']*100
dfGEP
sns.set(font_scale=1.5)
ax = dfGEP[['Education', 'Female', 'Male']].set_index('Education').plot(kind='bar', stacked=True)
plt.xlabel('Age', fontsize= 25)
plt.ylabel('Users', fontsize=25)
plt.xticks(rotation=0)
sns.set(rc={'figure.figsize':(20,12)})
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q5'].value_counts()/tot_count*100
x = df['Q5'].value_counts().index
y = df['Q5'].value_counts()/tot_count*100
df5 = pd.DataFrame(list(zip(x,y)),columns=['Discipline', '% Users'])
df5 = df5.replace({'Computer science (software engineering, etc.)':"Computer Sci",'Engineering (non-computer focused)': "Engineering", 'Mathematics or statistics':'Math & Stats','A business discipline (accounting, economics, finance, etc.)':"Buisness",'Physics or astronomy':'Phy & Astronomy','Information technology, networking, or system administration':'IT & Networking','Medical or life sciences (biology, chemistry, medicine, etc.)':'Medical','Social sciences (anthropology, psychology, sociology, etc.)':"Social Sci",'Humanities (history, literature, philosophy, etc.)':'Humanities','Environmental science or geology':'Environmental Sci','I never declared a major':"No Major",'Fine arts or performing arts':'Fine Arts'})
df5
sns.barplot(x = df5['Discipline'], y = df5['% Users'])
plt.xlabel('Discipline', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q6'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q6'].value_counts().index, y=df['Q6'].value_counts()/tot_count*100)
plt.xlabel('Designation', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q7'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q7'].value_counts().index, y=df['Q7'].value_counts()/tot_count*100)
plt.xlabel('Industry', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q8'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q8'].value_counts().index, y=df['Q8'].value_counts()/tot_count*100)
plt.xlabel('Experience', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=0)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q9'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q9'].value_counts().index[1:], y=df['Q9'].value_counts()[1:]/tot_count*100)
plt.xlabel('Yearly Compensation (in $)', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q10'].value_counts()/tot_count*100
x = df['Q10'].value_counts().index
y = df['Q10'].value_counts()/tot_count*100
df10 = pd.DataFrame(list(zip(x,y)),columns=['ML Use', '% Users'])
df10 = df10.replace({'We are exploring ML methods (and may one day put a model into production)':"Exploring ML",'No (we do not use ML methods)': "Don't Use", 'We recently started using ML methods (i.e., models in production for less than 2 years)':'Recently started using','I do not know':"Don't know",'We have well established ML methods (i.e., models in production for more than 2 years)':'Use well established models','We use ML methods for generating insights (but do not put working models into production)':'Generate Insights'})
df10
sns.barplot(x = df10['ML Use'], y = df10['% Users'])
plt.xlabel('ML Use', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
q11 = []
for i in range(14,21):
    q11.append(dft.iloc[i].count()/tot_count*100)
    
Ind = ['Analyse and Influence','Build, Run models which make product better','Build, Run data used for storing,Analyzing,etc','Build Prototypes','Research','NOTA','Others']
df11 = pd.DataFrame(list(zip(Ind,q11)),columns=['Work Role', '% Users'])
df11 = df11.sort_values('% Users',ascending=False)
df11
ax = sns.barplot(x = df11['Work Role'], y = df11['% Users'])
plt.xlabel('Work Role', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q13 = []
for i in range(29,44):
    q13.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Jupyter/IPython','RStudio','PyCharm','Visual Studio Code','nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text','Vim','IntelliJ','Spyder','None','Other' ]
df13 = pd.DataFrame(list(zip(Ind,q13)),columns=['IDE', '% Users'])
df13 = df13.sort_values('% Users',ascending=False)
df13
ax = sns.barplot(x = df13['IDE'], y = df13['% Users'])
plt.xlabel('IDE', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q14 = []
for i in range(45,56):
    q14.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Kaggle Kernel','Google Colab','Azure Notebook','Domino Datalab','Google Cloud Datalab','Paperspace','Floydhub','Crestle','JupyterHub/Binder','None','Other']
df14 = pd.DataFrame(list(zip(Ind,q14)),columns=['Notebooks', '% Users'])
df14 = df14.sort_values('% Users',ascending=False)
df14
sns.barplot(x = df14['Notebooks'], y = df14['% Users'])
plt.xlabel('Notebooks', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q15 = []
for i in range(57,64):
    q15.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Google Cloud Platform','Amazon Web Services','Microsoft Azure','IBM Cloud','Alibaba Cloud','Not used any Cloud Provider','Other']
df15 = pd.DataFrame(list(zip(Ind,q15)),columns=['Cloud Provider', '% Users'])
df15 = df15.sort_values('% Users',ascending=False)
df15
sns.barplot(x = df15['Cloud Provider'], y = df15['% Users'])
plt.xlabel('Cloud Provider', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q16 = []
for i in range(65,83):
    q16.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Python','R','SQL','Bash','Java','Javascript/Typescript','Visual Basic/VBA','C/C++','MATLAB','Scala','Julia','Go','C#/.NET','PHP','Ruby','SAS/STATA','None','Other']
df16 = pd.DataFrame(list(zip(Ind,q16)),columns=['Language', '% Users'])
df16 = df16.sort_values('% Users',ascending=False)
df16
sns.barplot(x = df16['Language'], y = df16['% Users'])
plt.xlabel('Language', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
sns.barplot(x = df16['Language'], y = df16['% Users'])
plt.xlabel('Language', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
count = df['Q17'].count()
df['Q17'].value_counts()/count*100
ax = sns.barplot(x=df['Q17'].value_counts().index, y=df['Q17'].value_counts()/count*100)
plt.xlabel('Favorite Language', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
count = df['Q18'].count()
df['Q18'].value_counts()/count*100
ax = sns.barplot(x=df['Q18'].value_counts().index, y=df['Q18'].value_counts()/count*100)
plt.xlabel('Recommended Language', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q23'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q23'].value_counts().index, y=df['Q23'].value_counts()/count*100)
plt.xlabel('Coding Time', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
q19 = []
for i in range(88,107):
    q19.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Scikit-Learn','TensorFlow','Keras','PyTorch','Spark MLlib','H2O','Fastai','Mxnet','Caret','Xgboost','mlr','Prophet','randomForest','lightgbm','catboost','CNTK','Caffe','None','Other']
df19 = pd.DataFrame(list(zip(Ind,q19)),columns=['Frameworks', '% Users'])
df19 = df19.sort_values('% Users',ascending=False)
df19
sns.barplot(x = df19['Frameworks'], y = df19['% Users'])
plt.xlabel('Frameworks', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
count = df['Q20'].count()
df['Q20'].value_counts()/count*100
ax = sns.barplot(x=df['Q20'].value_counts().index, y=df['Q20'].value_counts()/count*100)
plt.xlabel('Most used Framework', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
q21 = []
for i in range(110,123):
    q21.append(dft.iloc[i].count()/tot_count*100)

Ind = ['ggplot2','Matplotlib','Altair','Shiny','D3','Plotly','Bokeh','Seaborn','Geopltlib','Leaflet','Lattice','None','Other']
df21 = pd.DataFrame(list(zip(Ind,q21)),columns=['Visualisation Library', '% Users'])
df21 = df21.sort_values('% Users',ascending=False)
df21
sns.barplot(x = df21['Visualisation Library'], y = df21['% Users'])
plt.xlabel('Visualisation Library', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
count = df['Q22'].count()
df['Q22'].value_counts()/count*100
ax = sns.barplot(x=df['Q22'].value_counts().index, y=df['Q22'].value_counts()/count*100)
plt.xlabel('Most used Visualisation Library', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q25'].value_counts()/count*100

ax = sns.barplot(x=df['Q25'].value_counts().index, y=df['Q25'].value_counts()/count*100)
plt.xlabel('Used ML (Years)', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
q35 = []
q35.append(df['Q35_Part_1'].astype('float').dropna().mean())
q35.append(df['Q35_Part_2'].astype('float').dropna().mean())
q35.append(df['Q35_Part_3'].astype('float').dropna().mean())
q35.append(df['Q35_Part_4'].astype('float').dropna().mean())
q35.append(df['Q35_Part_5'].astype('float').dropna().mean())
q35.append(df['Q35_Part_6'].astype('float').dropna().mean())
Ind = ['Self-Taught','Online Courses','Work','University','Kaggle Competitions','Other']

df35 = pd.DataFrame(list(zip(Ind,q35)),columns=['How much spent on what?', '% Users'])
df35
labels = Ind
sizes = q35
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue','lightgreen','red']
explode = (0.1, 0, 0, 0,0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df['Q24'].value_counts()/tot_count*100
ax = sns.barplot(x=df['Q24'].value_counts().index, y=df['Q24'].value_counts()/count*100)
plt.xlabel('Time to Analyse Data', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
df['Q26'].value_counts()/count*100
ax = sns.barplot(x=df['Q26'].value_counts().index, y=df['Q26'].value_counts()/count*100)
plt.xlabel('Are you a data Scientist?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
q27 = []
for i in range(130,150):
    q27.append(dft.iloc[i].count()/tot_count*100)

Ind = ['AWS Elastic Compute Cloud','Google Compute Engine','Elastic Beanstalk','Google App','Google Kubernates','AWS Lambda','Google Cloud Functions','AWS Batch','Azure Virtual Machines','Azure Container Service','Azure Functions','Azure Event Grid','Azure Batch','Azure Kubernates Service','IBM Cloud Virtual Server','IBM Cloud Container Registry','IBM Cloud Kubernates Service','IBM Cloud Foundry','None','Other']
df27 = pd.DataFrame(list(zip(Ind,q27)),columns=['Cloud Computing Products', '% Users'])
df27 = df27.sort_values('% Users',ascending=False)
df27
sns.barplot(x = df27['Cloud Computing Products'], y = df27['% Users'])
plt.xlabel('Cloud Computing Products', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q28 = []
for i in range(151,194):
    q28.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Amazon Transcribe','Google Cloud Specch-to-text API','Amazon Rekognition','Google Cloud Vision API','Amazon Comprehend','Google Cloud Natural Language API','Amazon Translate','Google Translation API','Amazon Lex','Google Dialogflow Enterprose Edition','Amazon Rekognition Video','Google Cloud Video Intelligence API','Google Cloud AutoML','Amazon Sage Maker','Google Cloud ML Engine','DataRobot','H2O Driverless AI','Domino Datalab','SAS','Dataiku','RapidMiner','Instabase','Algorithmia','Dataversity','Cloudera','Azure ML Studio','Azure ML Workbench','Azure Cortana Intelligence Suite','Azure Bing Speech API','Azure Speaker Recognition API','Azure Computer Vision API','Azure Face API','Azure Video API','IBM Watson Studio','IBM Watson Knowledge Catalog','IBM Watson Assistant','IBM Watson Discovery','IBM Text to Speech','IBM Watson Visual Recognition','IBM Watson ML','Azure Cognitive Services','None','Other']
df28 = pd.DataFrame(list(zip(Ind,q28)),columns=['ML Products', '% Users'])
df28 = df28.sort_values('% Users',ascending=False)
df28
sns.barplot(x = df28['ML Products'][1:], y = df28['% Users'][1:])
plt.xlabel('ML Products', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q29 = []
for i in range(195,223):
    q29.append(dft.iloc[i].count()/tot_count*100)

Ind = ['AWS RD Sevice','AWS Aurora','Google CLoud SQL','Google CLoud Spanner','AWS DynamoDB','Google Cloud Datastore','Google Cloud Bigtable','AWS SimpleDB','Microsoft SQL Server','MySQL','PostgresSQL','SQLite','Oracle Database','Ingres','Microsoft Access','NexusDB','SAP IQ','Google Fusion Tables','Azure Database for MySQL','Azure Cosmos DB','Azure SQL Database','PostgresSQL','IBM Cloud Compose','IBM Cloud Compose SQL','IBM Cloud Compose PostgresSQL','IBM Cloud Db2','None','Other']
df29 = pd.DataFrame(list(zip(Ind,q29)),columns=['RDatabase Products', '% Users'])
df29 = df29.sort_values('% Users',ascending=False)
df29

sns.barplot(x = df29['RDatabase Products'], y = df29['% Users'])
plt.xlabel('RDatabase Products', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q30 = []
for i in range(224,250):
    q30.append(dft.iloc[i].count()/tot_count*100)

Ind = ['AWS Elastic MapReduce','AWS Batch','Google Cloud Dataproc','Google Cloud Dataflow','Google Cloud Dataprep','AWS Kinesis','Google Cloud Pub / Sub','AWS Athena','AWS Redshift','Google BigQuery','Teradata','Microsoft Analysis Services','Oracle Exadata','Oracle Warehouse Builder','SAP IQ','Snowflake','Databricks','Azure SQL Data Warehouse','Azure HDInsight','Azure Stream Analytics','IBM Infosphere DatStorage','IBM Cloud Analytics','IBM Cloud Streaming Analytics','None','Other']
df30 = pd.DataFrame(list(zip(Ind,q30)),columns=['Big Data & Analytics Products', '% Users'])
df30 = df30.sort_values('% Users',ascending=False)
df30
sns.barplot(x = df30['Big Data & Analytics Products'], y = df30['% Users'])
plt.xlabel('Big Data & Analytics Products', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q31 = []
for i in range(250,263):
    q31.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Audio','Categorical','Genetic','Geospatial','Image','Numerical','Sensor','Tabular','Text','Time Series','Video','Other']
df31 = pd.DataFrame(list(zip(Ind,q31)),columns=['Type of Data', '% Users'])
df31 = df31.sort_values('% Users',ascending=False)
df31
sns.barplot(x = df31['Type of Data'], y = df31['% Users'])
plt.xlabel('Type of Data', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
32
q33 = []
for i in range(265,277):
    q33.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Government Sites','University Research Group','Non-profit Research Groups','Data Aggregation Platform','Collect Own data','Publicly released Data','Google Search','Google Dataset Search','Github','Do not work with public data','Other']
df33 = pd.DataFrame(list(zip(Ind,q33)),columns=['Where do you find Data?', '% Users'])
df33 = df33.sort_values('% Users',ascending=False)
df33
sns.barplot(x = df33['Where do you find Data?'], y = df33['% Users'])
plt.xlabel('Where do you find Data?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1 = sns.distplot(df['Q34_Part_1'].astype('float').dropna(),kde=False, norm_hist=False ,bins=28,ax = ax1)
ax1.set_yticklabels(['{:,.2%}'.format(x/df['Q34_Part_1'].astype('float').dropna().count()) for x in ax1.get_yticks()])
ax1.set(xlabel='Gathering Data')
ax2 = sns.distplot(df['Q34_Part_2'].astype('float').dropna(),kde=False, norm_hist=False ,bins=28,ax = ax2)
ax2.set_yticklabels(['{:,.2%}'.format(x/df['Q34_Part_2'].astype('float').dropna().count()) for x in ax2.get_yticks()])
ax2.set(xlabel='Cleaning Data')
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1 = sns.distplot(df['Q34_Part_3'].astype('float').dropna(),kde=False, norm_hist=False ,bins=28,ax = ax1)
ax1.set_yticklabels(['{:,.2%}'.format(x/df['Q34_Part_2'].astype('float').dropna().count()) for x in ax1.get_yticks()])
ax1.set(xlabel='Visualizing Data')
ax2 = sns.distplot(df['Q34_Part_4'].astype('float').dropna(),kde=False, norm_hist=False ,bins=28,ax = ax2)
ax2.set_yticklabels(['{:,.2%}'.format(x/df['Q34_Part_2'].astype('float').dropna().count()) for x in ax2.get_yticks()])
ax2.set(xlabel='Model Selction')
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1 = sns.distplot(df['Q34_Part_5'].astype('float').dropna(),kde=False, norm_hist=False ,bins=28,ax = ax1)
ax1.set_yticklabels(['{:,.2%}'.format(x/df['Q34_Part_2'].astype('float').dropna().count()) for x in ax1.get_yticks()])
ax1.set(xlabel='Model in Production')
ax2 = sns.distplot(df['Q34_Part_6'].astype('float').dropna(),kde=False, norm_hist=False ,bins=28,ax = ax2)
#ax2.set_yticklabels(['{:,.2%}'.format(x/df['Q34_Part_2'].astype('float').dropna().count()) for x in ax2.get_yticks()])
ax2.set(xlabel='Find Insights & Communicate with Stakeholders')


q36 = []
for i in range(291,305):
    q36.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Udacity','Coursera','edX','DataCamp','DataQuest','Kaggle Learn','Fast.AI','Google Developers','Udemy','TheSchool.AI','Online University Courses','None','Other']
df36= pd.DataFrame(list(zip(Ind,q36)),columns=['Data Science Courses', '% Users'])
df36 = df36.sort_values('% Users',ascending=False)
df36
sns.barplot(x = df36['Data Science Courses'], y = df36['% Users'])
plt.xlabel('Data Science Courses', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
37
q38 = []
for i in range(307,330):
    q38.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Twitter','Hacker News','r/Machine Learning','Kaggle Forums','Fastai Forums','Siraj Raval YouTube Channel','Data Tau News Aggregator','Linear Digressions Podcast','Cloud AI Adventures (YouTube)','FiveThirtyEight.com','ArXiv & Preprints','Journal Publications','FastML Blog','KDnuggets Blog','OReilly Data Newsletter','Partially Derivative Podcast','The Data Skeptic Podcast','Medium Blog Posts','Towards Data Science Blog','Analytics Vidhya Blog','None/I do not know','Other']
df38= pd.DataFrame(list(zip(Ind,q38)),columns=['Favorite Sources', '% Users'])
df38 = df38.sort_values('% Users',ascending=False)
df38
sns.barplot(x = df38['Favorite Sources'], y = df38['% Users'])
plt.xlabel('Favorite Sources', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
df391 = df['Q39_Part_1'].value_counts()/df['Q39_Part_1'].count()*100
df391
labels = ['Slightly better','Much better','Neither better nor worse','Slightly worse','No opinion; I do not know','Much worse']
sizes = df391
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue','lightgreen','red']
explode = (0.1, 0, 0, 0,0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df392 = df['Q39_Part_2'].value_counts()/df['Q39_Part_2'].count()*100
df392
labels = ['No opinion; I do not know','Slightly better','Neither better nor worse','Much better','Slightly worse','Much worse']
sizes = df392
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue','lightgreen','red']
explode = (0.1, 0, 0, 0,0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
q34 = []
q34.append(df['Q34_Part_1'].astype('float').dropna().mean())
q34.append(df['Q34_Part_2'].astype('float').dropna().mean())
q34.append(df['Q34_Part_3'].astype('float').dropna().mean())
q34.append(df['Q34_Part_4'].astype('float').dropna().mean())
q34.append(df['Q34_Part_5'].astype('float').dropna().mean())
q34.append(df['Q34_Part_6'].astype('float').dropna().mean())
Ind = ['Gathering Data','Cleaning Data','Visualizing Data','Model Selction','Model in Production','Find Insights & Communicate with Stakeholders']

df34 = pd.DataFrame(list(zip(Ind,q34)),columns=['How much spent on what?', '% Users'])
df34
labels = Ind
sizes = q34
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue','lightgreen','red']
explode = (0, 0, 0.1, 0,0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df40 = df['Q40'].value_counts()/df['Q40'].count()*100
df40
labels = ['Independent projects are much more important than academic achievements ','Independent projects are slightly more important than academic achievements','Independent projects are equally important as academic achievements','No opinion; I do not know','Independent projects are slightly less important than academic achievements','Independent projects are much less important than academic achievements']
sizes = df40
colors = ['peachpuff', 'green', 'lightcoral', 'royalblue','lightgreen','red']
explode = (0.1, 0, 0, 0,0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df411 = df['Q41_Part_1'].value_counts()/df['Q41_Part_1'].count()*100
df411
labels = ['Very important','Slightly important','No opinion; I do not know','Not at all important']
sizes = df411
colors = ['peachpuff','royalblue','lightgreen','red']
explode = (0, 0.1, 0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df412 = df['Q41_Part_2'].value_counts()/df['Q41_Part_2'].count()*100
df412
labels = ['Very important','Slightly important','No opinion; I do not know','Not at all important']
sizes = df412
colors = ['peachpuff','royalblue','lightgreen','red']
explode = (0, 0.1, 0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df413 = df['Q41_Part_3'].value_counts()/df['Q41_Part_3'].count()*100
df413
labels = ['Very important','Slightly important','No opinion; I do not know','Not at all important']
sizes = df413
colors = ['peachpuff','royalblue','lightgreen','red']
explode = (0, 0.1, 0,0)  # explode 1st slice
font = {'weight':'normal','size'   : 25} 
plt.rc('font', **font)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
q42 = []
for i in range(336,341):
    q42.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Revenue / Buisness Goals','Accuracy Metrics','Unfair Bias Metrics','Not Applicable','Other']
df42= pd.DataFrame(list(zip(Ind,q42)),columns=['How to measure Success?', '% Users'])
df42 = df42.sort_values('% Users',ascending=False)
df42
sns.barplot(x = df42['How to measure Success?'], y = df42['% Users'])
plt.xlabel('How to measure Success?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
df['Q43'].value_counts()
ax = sns.barplot(x=df['Q43'].value_counts().index, y=df['Q43'].value_counts()/tot_count*100)
plt.xlabel('Designation', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
q44 = []
for i in range(344,350):
    q44.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Lack Of Communication','Identifying Unfairly Targeted Groups','Collecting Data on Unfairly Targeted Groups','Identifying and selecting correct Eval. metrics','No difficulty','Never performed this task']
df44= pd.DataFrame(list(zip(Ind,q44)),columns=['What Difficulty you face?', '% Users'])
df44 = df44.sort_values('% Users',ascending=False)
df44
sns.barplot(x = df44['What Difficulty you face?'], y = df44['% Users'])
plt.xlabel('What Difficulty you face?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q45 = []
for i in range(349,356):
    q45.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Very Imp.(Production Ready Models)','All Models','To determine models worth','Check if model produces required Insights','Initially Exploring new model or dataset','I do not explore & interpret models']
df45 = pd.DataFrame(list(zip(Ind,q45)),columns=['When do you explore model predictions?', '% Users'])
df45 = df45.sort_values('% Users',ascending=False)
df45
sns.barplot(x =  df45['When do you explore model predictions?'], y = df45['% Users'])
plt.xlabel('When do you explore model predictions?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
46
q47 = []
for i in range(356,372):
    q47.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Examine Individual Model Coefficient','Examine feature correlations','Examine feature importances','Plot decision boundaries','Create partial dependence plots','Dimensionality reduction techniques','Attention mapping/saliency mapping','Plot predicted vs. actual results','Print out a decision tree','Sensitivity analysis/perturbation importance','LIME functions','ELI5 functions','SHAP functions','None','Other']
df47 = pd.DataFrame(list(zip(Ind,q47)),columns=['When do you explore model predictions?', '% Users'])
df47 = df47.sort_values('% Users',ascending=False)
df47
sns.barplot(x =  df47['When do you explore model predictions?'], y = df47['% Users'])
plt.xlabel('When do you explore model predictions?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q49 = []
for i in range(373,386):
    q49.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Share Code on Github / Similar Code Repo','Share Code & Data on Github / Similar Code Repo','Share Code, Data & Environment on Hosted Service(Kaggle)','Share Code, Data & Environment using Containers(Docker)','Share Code, Data & Environment on VirtualMachine','Make sure code is well documented','Make sure code is human readable','Define all random seeds','Define relative rather absolute paths','Include file regarding all Dependencies','None/ I do not make my work easy for others to reproduce',' Other']
df49 = pd.DataFrame(list(zip(Ind,q49)),columns=['What you do to make your work easy to reproduce?', '% Users'])
df49 = df49.sort_values('% Users',ascending=False)
df49
sns.barplot(x =  df49['What you do to make your work easy to reproduce?'], y = df49['% Users'])
plt.xlabel('What you do to make your work easy to reproduce?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
q50 = []
for i in range(386,395):
    q50.append(dft.iloc[i].count()/tot_count*100)

Ind = ['Too Expensive','Too Time-consuming','Too much Tech. Knowledge required','Afraid of other not giving due credit','Not enough Incentives to share my work','Never considered making my work easier to reproduce','None',' Other']
df50 = pd.DataFrame(list(zip(Ind,q50)),columns=['What prevents you to make work easier to reuse and reproduce?', '% Users'])
df50 = df50.sort_values('% Users',ascending=False)
df50
sns.barplot(x =  df50['What prevents you to make work easier to reuse and reproduce?'], y = df50['% Users'])
plt.xlabel('What prevents you to make work easier to reuse and reproduce?', fontsize=25)
plt.ylabel('% Users', fontsize=25)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(20,12)})
sns.set(font_scale=2)  
#pylab.rcParams['ytick.major.pad']='10'
ax.xaxis.labelpad = 25
ax.yaxis.labelpad = 15
###############################################################
###############################################################







